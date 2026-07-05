import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention.
    """

    def __init__(
        self,
        ModelDimension: int,
        NumHeads: int,
        AttentionDropout: float = 0.1,
    ):
        super().__init__()

        assert ModelDimension % NumHeads == 0, \
            "ModelDimension must be divisible by NumHeads"

        self.ModelDimension = ModelDimension
        self.NumHeads       = NumHeads
        self.HeadDimension  = ModelDimension // NumHeads

        self.LinearLayerQuery  = nn.Linear(ModelDimension, ModelDimension, bias=False)  # W_Q
        self.LinearLayerKey    = nn.Linear(ModelDimension, ModelDimension, bias=False)  # W_K
        self.LinearLayerValue  = nn.Linear(ModelDimension, ModelDimension, bias=False)  # W_V
        self.LinearLayerOutput = nn.Linear(ModelDimension, ModelDimension)              # W_O

        self.AttentionDropoutLayer = nn.Dropout(AttentionDropout)

        # Store the attention weights for explainability later
        self.StoredAttentionWeights: torch.Tensor | None = None

    def forward(
        self,
        Query: torch.Tensor,
        Key: torch.Tensor, 
        Value: torch.Tensor,
        Mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        BatchSize = Query.size(0) 

        Q = self.LinearLayerQuery(Query)
        K = self.LinearLayerKey(Key)
        V = self.LinearLayerValue(Value)

        Q = Q.view(BatchSize, -1, self.NumHeads, self.HeadDimension).transpose(1, 2)
        K = K.view(BatchSize, -1, self.NumHeads, self.HeadDimension).transpose(1, 2)
        V = V.view(BatchSize, -1, self.NumHeads, self.HeadDimension).transpose(1, 2)

        ScalingFactor   = math.sqrt(self.HeadDimension)
        AttentionScores = torch.matmul(Q, K.transpose(-2, -1)) / ScalingFactor

        if Mask is not None:
            AttentionScores = AttentionScores.masked_fill(Mask == 0, float("-inf"))


        AttentionWeights = F.softmax(AttentionScores, dim=-1)
        AttentionWeights = self.AttentionDropoutLayer(AttentionWeights)

        self.StoredAttentionWeights = AttentionWeights.detach()

        ContextVector = torch.matmul(AttentionWeights, V)
        ContextVector = ContextVector.transpose(1, 2).contiguous().view(
            BatchSize, -1, self.ModelDimension
        )
        Output = self.LinearLayerOutput(ContextVector)
        return Output


class ForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network
    """

    def __init__(
        self,
        ModelDimension: int,
        HiddenDimension: int,
        ForwardDropout: float = 0.1,
    ):
        super().__init__()
        self.LinearLayerOne = nn.Linear(ModelDimension, HiddenDimension)
        self.ActivationGELU = nn.GELU()
        self.DropoutLayer   = nn.Dropout(ForwardDropout)
        self.LinearLayerTwo = nn.Linear(HiddenDimension, ModelDimension)

    def forward(self, InputTensor: torch.Tensor) -> torch.Tensor:
        Intermediate = self.LinearLayerOne(InputTensor)
        Intermediate = self.ActivationGELU(Intermediate)
        Intermediate = self.DropoutLayer(Intermediate)
        OutputTensor = self.LinearLayerTwo(Intermediate)
        return OutputTensor


class LayerNormalization(nn.Module):
    """
    Layer Normalization
    """
    def __init__(self, NormDimension: int, Epsilon: float = 1e-6):
        super().__init__()
        self.Epsilon        = Epsilon
        self.ScaleParameter = nn.Parameter(torch.ones(NormDimension))
        self.ShiftParameter = nn.Parameter(torch.zeros(NormDimension))

    def forward(self, InputTensor: torch.Tensor) -> torch.Tensor:
        Mean       = InputTensor.mean(dim=-1, keepdim=True)
        Variance   = InputTensor.var(dim=-1, keepdim=True, unbiased=False)
        Normalized = (InputTensor - Mean) / torch.sqrt(Variance + self.Epsilon)
        return self.ScaleParameter * Normalized + self.ShiftParameter


class PositionalEncoding(nn.Module):

    def __init__(self, ModelDimension: int, MaxSequenceLength: int = 5000, Dropout: float = 0.1):
        super().__init__()
        self.DropoutLayer = nn.Dropout(Dropout)

        EncodingMatrix = torch.zeros(MaxSequenceLength, ModelDimension)
        PositionVector = torch.arange(0, MaxSequenceLength).unsqueeze(1).float()
        DivisionTerms  = torch.exp(
            torch.arange(0, ModelDimension, 2).float() * (-math.log(10000.0) / ModelDimension)
        )
        EncodingMatrix[:, 0::2] = torch.sin(PositionVector * DivisionTerms)
        EncodingMatrix[:, 1::2] = torch.cos(PositionVector * DivisionTerms)
        EncodingMatrix = EncodingMatrix.unsqueeze(0)

        self.register_buffer("EncodingMatrix", EncodingMatrix)

    def forward(self, InputTensor: torch.Tensor) -> torch.Tensor:
        SequenceLength = InputTensor.size(1)
        OutputTensor   = InputTensor + self.EncodingMatrix[:, :SequenceLength, :]
        return self.DropoutLayer(OutputTensor)


class InputEmbeddingLayer(nn.Module):

    def __init__(self, NumFeatures: int, ModelDimension: int):
        super().__init__()
        self.ProjectionLayer = nn.Linear(NumFeatures, ModelDimension)
        self.ActivationReLU  = nn.ReLU()

    def forward(self, InputTensor: torch.Tensor) -> torch.Tensor:
        Projected = self.ProjectionLayer(InputTensor)
        return self.ActivationReLU(Projected)


class TwoStepMasking(nn.Module):
    """
    Implements a two-step masking strategy for time series data:
    Step 1: Random masking
    Step 2: Masking based on reconstruction error
    """

    def __init__(
        self,
        RandomMaskingRate: float = 0.15,
        AnomalyThreshold: float  = 0.75,
    ):
        super().__init__()
        self.RandomMaskingRate = RandomMaskingRate
        self.AnomalyThreshold  = AnomalyThreshold

    def StepOneRandomMasking(
        self, InputTensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        Batch, Time, Dim = InputTensor.shape

        RandomTensor = torch.rand(Batch, Time, device=InputTensor.device)
        MaskTensor   = (RandomTensor > self.RandomMaskingRate).float()
        MaskExpanded = MaskTensor.unsqueeze(-1).expand_as(InputTensor)
        MaskedTensor = InputTensor * MaskExpanded
        return MaskedTensor, MaskTensor 

    def StepTwoAnomalyMasking(
        self,
        InputTensor: torch.Tensor,
        ReconstructionError: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        Threshold    = torch.quantile(
            ReconstructionError, self.AnomalyThreshold, dim=1, keepdim=True
        )

        AnomalyMask  = (ReconstructionError < Threshold).float()
        MaskExpanded = AnomalyMask.unsqueeze(-1).expand_as(InputTensor)
        MaskedTensor = InputTensor * MaskExpanded
        return MaskedTensor, AnomalyMask

    def forward(
        self,
        InputTensor: torch.Tensor,
        ReconstructionError: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if ReconstructionError is None:
            return self.StepOneRandomMasking(InputTensor) 
        else:
            return self.StepTwoAnomalyMasking(InputTensor, ReconstructionError)





# Transformer Encoder & Decoder

class TransformerEncoderBlock(nn.Module):

    def __init__(
        self,
        ModelDimension: int,
        NumHeads: int,
        ForwardDimension: int,
        DropoutRate: float = 0.1,
    ):
        super().__init__()
        self.SelfAttention       = MultiHeadAttention(ModelDimension, NumHeads, DropoutRate)
        self.NormalizationOne    = LayerNormalization(ModelDimension)
        self.ForwardNetworkLayer = ForwardNetwork(ModelDimension, ForwardDimension, DropoutRate)
        self.NormalizationTwo    = LayerNormalization(ModelDimension)
        self.ResidualDropout     = nn.Dropout(DropoutRate)

    def forward(self, InputTensor: torch.Tensor, Mask: torch.Tensor | None = None) -> torch.Tensor:
        AttentionOutput = self.SelfAttention(InputTensor, InputTensor, InputTensor, Mask)
        AttentionOutput = self.ResidualDropout(AttentionOutput)
        AfterNormOne    = self.NormalizationOne(InputTensor + AttentionOutput)

        FFNOutput    = self.ForwardNetworkLayer(AfterNormOne)
        FFNOutput    = self.ResidualDropout(FFNOutput)
        OutputTensor = self.NormalizationTwo(AfterNormOne + FFNOutput)
        return OutputTensor


class TransformerDecoderBlock(nn.Module):

    def __init__(
        self,
        ModelDimension: int,
        NumHeads: int,
        ForwardDimension: int,
        DropoutRate: float = 0.1,
    ):
        super().__init__()
        self.MaskedSelfAttention = MultiHeadAttention(ModelDimension, NumHeads, DropoutRate)
        self.NormalizationOne    = LayerNormalization(ModelDimension)

        self.CrossAttention      = MultiHeadAttention(ModelDimension, NumHeads, DropoutRate)
        self.NormalizationTwo    = LayerNormalization(ModelDimension)

        self.ForwardNetworkLayer = ForwardNetwork(ModelDimension, ForwardDimension, DropoutRate)
        self.NormalizationThree  = LayerNormalization(ModelDimension)

        self.ResidualDropout     = nn.Dropout(DropoutRate)

    def forward(
        self,
        DecoderInput: torch.Tensor,
        EncoderOutput: torch.Tensor,
        DecoderMask: torch.Tensor | None = None,
        EncoderMask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        SelfAttOutput = self.MaskedSelfAttention(
            DecoderInput, DecoderInput, DecoderInput, DecoderMask 
        )
        SelfAttOutput = self.ResidualDropout(SelfAttOutput)
        AfterNormOne  = self.NormalizationOne(DecoderInput + SelfAttOutput)

        CrossAttOutput = self.CrossAttention(
            AfterNormOne, 
            EncoderOutput,
            EncoderOutput,
            EncoderMask
        )
        CrossAttOutput = self.ResidualDropout(CrossAttOutput)
        AfterNormTwo   = self.NormalizationTwo(AfterNormOne + CrossAttOutput)

        FFNOutput    = self.ForwardNetworkLayer(AfterNormTwo)
        FFNOutput    = self.ResidualDropout(FFNOutput)
        OutputTensor = self.NormalizationThree(AfterNormTwo + FFNOutput)
        return OutputTensor 


class TransformerEncoderStack(nn.Module):

    def __init__(self, NumLayers: int, ModelDimension: int, NumHeads: int,
                 ForwardDimension: int, DropoutRate: float = 0.1):
        super().__init__()

        self.EncoderBlockList = nn.ModuleList([
            TransformerEncoderBlock(ModelDimension, NumHeads, ForwardDimension, DropoutRate)
            for _ in range(NumLayers)
        ])
        self.FinalNormalization = LayerNormalization(ModelDimension)

    def forward(self, InputTensor: torch.Tensor, Mask: torch.Tensor | None = None) -> torch.Tensor:
        CurrentTensor = InputTensor
        for EncoderBlock in self.EncoderBlockList:
            CurrentTensor = EncoderBlock(CurrentTensor, Mask)
        return self.FinalNormalization(CurrentTensor) 


class TransformerDecoderStack(nn.Module):

    def __init__(self, NumLayers: int, ModelDimension: int, NumHeads: int,
                 ForwardDimension: int, DropoutRate: float = 0.1):
        super().__init__()
        self.DecoderBlockList = nn.ModuleList([
            TransformerDecoderBlock(ModelDimension, NumHeads, ForwardDimension, DropoutRate)
            for _ in range(NumLayers)
        ])
        self.FinalNormalization = LayerNormalization(ModelDimension)

    def forward(
        self,
        DecoderInput: torch.Tensor, 
        EncoderOutput: torch.Tensor,
        DecoderMask: torch.Tensor | None = None, 
        EncoderMask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        CurrentTensor = DecoderInput 
        for DecoderBlock in self.DecoderBlockList:
            CurrentTensor = DecoderBlock(CurrentTensor, EncoderOutput, DecoderMask, EncoderMask)
        return self.FinalNormalization(CurrentTensor)



# Generator & Discriminator

class TransformerGenerator(nn.Module):

    def __init__(
        self,
        NumFeatures: int,
        ModelDimension: int = 64,
        NumHeads: int = 4, 
        NumEncoderLayers: int = 3,
        NumDecoderLayers: int = 3,
        ForwardDimension: int = 256,
        DropoutRate: float = 0.1,
        LearningRate: float = 1e-4,
        BetaOne: float = 0.5,
        BetaTwo: float = 0.999,
        WeightDecay: float = 1e-5,
    ):
        super().__init__()

        self.InputEmbedding     = InputEmbeddingLayer(NumFeatures, ModelDimension)
        self.PositionalEncoding = PositionalEncoding(ModelDimension, Dropout=DropoutRate)
        self.EncoderStack       = TransformerEncoderStack(
            NumEncoderLayers, ModelDimension, NumHeads, ForwardDimension, DropoutRate
        )
        self.DecoderStack       = TransformerDecoderStack(
            NumDecoderLayers, ModelDimension, NumHeads, ForwardDimension, DropoutRate
        )
        self.OutputProjection   = nn.Linear(ModelDimension, NumFeatures)

        self.ReconstructionLoss = nn.MSELoss(reduction="none")
        self.GeneratorLoss      = nn.BCEWithLogitsLoss()

        self.Optimizer = optim.Adam(
            self.parameters(),
            lr=LearningRate,
            betas=(BetaOne, BetaTwo),
            weight_decay=WeightDecay, 
        )

        self.LearningRateScheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.Optimizer, T_max=100, eta_min=1e-6
        )

    def forward(self, MaskedInput: torch.Tensor) -> torch.Tensor:
        Embedded       = self.InputEmbedding(MaskedInput)
        WithPosition   = self.PositionalEncoding(Embedded)
        EncoderContext = self.EncoderStack(WithPosition)
        DecoderOutput  = self.DecoderStack(WithPosition, EncoderContext)
        Reconstruction = self.OutputProjection(DecoderOutput)
        return Reconstruction 

    def ComputeErrorPerTimestep(
        self, Reconstruction: torch.Tensor, Original: torch.Tensor
    ) -> torch.Tensor:

        ErrorTensor = self.ReconstructionLoss(Reconstruction, Original)
        return ErrorTensor.mean(dim=-1)


class TransformerDiscriminator(nn.Module):

    def __init__(
        self,
        NumFeatures: int, 
        ModelDimension: int = 64,
        NumHeads: int = 4,
        NumLayers: int = 2,
        ForwardDimension: int = 128,
        DropoutRate: float = 0.1,
        LearningRate: float = 2e-4,
        BetaOne: float = 0.5, 
        BetaTwo: float = 0.999,
        WeightDecay: float = 1e-5, 
    ):
        super().__init__()
        self.InputEmbedding      = InputEmbeddingLayer(NumFeatures, ModelDimension)
        self.PositionalEncoding  = PositionalEncoding(ModelDimension, Dropout=DropoutRate)
        self.EncoderStack        = TransformerEncoderStack( 
            NumLayers, ModelDimension, NumHeads, ForwardDimension, DropoutRate
        )
        self.ClassificationDropout = nn.Dropout(0.2)
        self.ClassificationLayer   = nn.Linear(ModelDimension, 1)

        self.DiscriminatorLoss = nn.BCEWithLogitsLoss()

        self.Optimizer = optim.Adam(
            self.parameters(), lr=LearningRate, betas=(BetaOne, BetaTwo), weight_decay=WeightDecay
        )
        self.LearningRateScheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.Optimizer, T_max=100, eta_min=1e-6
        )

    def forward(self, InputTensor: torch.Tensor) -> torch.Tensor:
        Embedded      = self.InputEmbedding(InputTensor)
        WithPosition  = self.PositionalEncoding(Embedded)
        EncoderOutput = self.EncoderStack(WithPosition)
        GlobalMean    = EncoderOutput.mean(dim=1)
        GlobalMean    = self.ClassificationDropout(GlobalMean)
        Logit         = self.ClassificationLayer(GlobalMean)
        return Logit



# AnoFormer Main Model that combines everything

class AnoFormerMainModel:
    """
    AnoFormer: A Two-Step Transformer-Based Anomaly Detection Model
    """

    def __init__(
        self,
        NumFeatures: int,
        ModelDimension: int = 64,
        NumHeads: int = 4,
        NumEncoderLayers: int = 3,
        NumDecoderLayers: int = 3,
        DiscriminatorLayers: int = 2,
        ForwardDimension: int = 256,
        DropoutRate: float = 0.1,
        GeneratorLearningRate: float = 1e-4,
        DiscriminatorLearningRate: float = 2e-4,
        MaskingRate: float = 0.15,
        AnomalyThreshold: float = 0.75,
        ReconstructionWeight: float = 10.0,
        AnomalyLimit: float | None = None,
        Device: torch.device | None = None,
    ):
        self.Device = Device if Device is not None else DEVICE

        self.Generator = TransformerGenerator(
            NumFeatures, ModelDimension, NumHeads,
            NumEncoderLayers, NumDecoderLayers,
            ForwardDimension, DropoutRate,
            GeneratorLearningRate,
        ).to(self.Device)

        self.Discriminator = TransformerDiscriminator(
            NumFeatures, ModelDimension, NumHeads,
            DiscriminatorLayers,
            ForwardDimension // 2,
            DropoutRate, DiscriminatorLearningRate,
        ).to(self.Device)

        self.Masking = TwoStepMasking(MaskingRate, AnomalyThreshold)

        self.ReconstructionWeight = ReconstructionWeight 
        self.AnomalyLimit         = AnomalyLimit
        self.NumFeatures          = NumFeatures

        self.TrainingHistory: dict[str, list[float]] = {
            "GeneratorTotalLoss":  [],
            "GeneratorGANLoss":    [],
            "GeneratorRecLoss":    [],
            "DiscriminatorLoss":   [],
        }

    def TrainingStep(
        self,
        RealData: torch.Tensor,
        ReconstructionError: torch.Tensor | None = None,
    ) -> dict[str, float]:

        RealData = RealData.to(self.Device)

        MaskedData, _ = self.Masking(RealData, ReconstructionError)

        # Discriminator Step
        self.Discriminator.Optimizer.zero_grad() 

        RealLogits = self.Discriminator(RealData) 
        RealLabels = torch.ones_like(RealLogits)

        with torch.no_grad():
            FakeReconstruction = self.Generator(MaskedData)

        FakeLogits = self.Discriminator(FakeReconstruction.detach()) 
        FakeLabels = torch.zeros_like(FakeLogits)

        DiscriminatorLoss = (
            self.Discriminator.DiscriminatorLoss(RealLogits, RealLabels) + 
            self.Discriminator.DiscriminatorLoss(FakeLogits, FakeLabels) 
        ) * 0.5 

        DiscriminatorLoss.backward()
        nn.utils.clip_grad_norm_(self.Discriminator.parameters(), max_norm=1.0)
        self.Discriminator.Optimizer.step()


        # Generator Step
        self.Generator.Optimizer.zero_grad()

        FakeReconstruction = self.Generator(MaskedData)
        GAN_Logits         = self.Discriminator(FakeReconstruction)

        GeneratorGANLoss   = self.Generator.GeneratorLoss(GAN_Logits, torch.ones_like(GAN_Logits))
        RecLoss            = self.Generator.ReconstructionLoss(FakeReconstruction, RealData).mean()
        GeneratorTotalLoss = GeneratorGANLoss + self.ReconstructionWeight * RecLoss

        GeneratorTotalLoss.backward()
        nn.utils.clip_grad_norm_(self.Generator.parameters(), max_norm=1.0)
        self.Generator.Optimizer.step() 

        return {
            "GeneratorTotalLoss": GeneratorTotalLoss.item(),
            "GeneratorGANLoss":   GeneratorGANLoss.item(),
            "GeneratorRecLoss":   RecLoss.item(),
            "DiscriminatorLoss":  DiscriminatorLoss.item(),
        }

    def TrainModel(
        self,
        TrainingLoader: DataLoader,
        NumEpochs: int = 50,
        PrintInterval: int = 5,
    ) -> None:

        self.Generator.train()
        self.Discriminator.train()

        for Epoch in range(1, NumEpochs + 1):
            EpochLosses: dict[str, list[float]] = {k: [] for k in self.TrainingHistory}

            for Batch, (WindowData, _) in enumerate(TrainingLoader):
                WindowData = WindowData.to(self.Device)

                if Epoch % 2 == 0:
                    with torch.no_grad(): 
                        MaskedTemp, _ = self.Masking(WindowData, None)
                        RecTemp       = self.Generator(MaskedTemp)
                        ErrorPerTime  = self.Generator.ComputeErrorPerTimestep(RecTemp, WindowData)
                else:
                    ErrorPerTime = None

                LossValues = self.TrainingStep(WindowData, ErrorPerTime)

                for Key, Val in LossValues.items():
                    EpochLosses[Key].append(Val)

            for Key in self.TrainingHistory:
                Average = float(np.mean(EpochLosses[Key]))
                self.TrainingHistory[Key].append(Average)

            self.Generator.LearningRateScheduler.step() 
            self.Discriminator.LearningRateScheduler.step()

            if Epoch % PrintInterval == 0 or Epoch == 1:
                print(
                    f"Epoch {Epoch:>4}/{NumEpochs} | "
                    f"G-Total={self.TrainingHistory['GeneratorTotalLoss'][-1]:.4f} | "
                    f"G-GAN={self.TrainingHistory['GeneratorGANLoss'][-1]:.4f} | "
                    f"G-Rec={self.TrainingHistory['GeneratorRecLoss'][-1]:.4f} | "
                    f"D={self.TrainingHistory['DiscriminatorLoss'][-1]:.4f}"
                )

        self.Generator.eval()
        self.Discriminator.eval()
        print("Training Completed.")

    @torch.no_grad()
    def CalculateScores(
        self,
        DataLoaderObject: DataLoader,
        NumMaskRuns: int = 5,
        Seed: int = 42,
    ) -> np.ndarray:
        """
        Calculates one stable anomaly score per ECG
        """
        self.Generator.eval()
        ScoresFromRuns: list[np.ndarray] = []

        for RunIndex in range(NumMaskRuns):
            CurrentSeed = Seed + RunIndex

            torch.manual_seed(CurrentSeed)
            np.random.seed(CurrentSeed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(CurrentSeed)

            CurrentRunScores: list[np.ndarray] = []

            for WindowData, _ in DataLoaderObject:

                WindowData = WindowData.to(self.Device)
                MaskedData, _ = self.Masking(WindowData, ReconstructionError=None)

                Reconstruction = self.Generator(MaskedData)
                ErrorPerTime = (self.Generator.ComputeErrorPerTimestep(Reconstruction, WindowData))

                BatchScores = ErrorPerTime.mean(dim=1)
                CurrentRunScores.append(BatchScores.cpu().numpy())

            CurrentRunScores = np.concatenate(CurrentRunScores,axis=0)
            ScoresFromRuns.append(CurrentRunScores)

        ScoresFromRuns = np.stack(ScoresFromRuns, axis=0,)
        StableScores = ScoresFromRuns.mean(axis=0,)
        return StableScores


    @torch.no_grad()
    def DetectAnomalies(
        self,
        TestLoader: DataLoader,
        Threshold: float | None = None,
        Percentile: float = 95.0,
        NumMaskRuns: int = 5,
        Seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates stable anomaly scores and converts them
        into binary predictions
        """
        AllScores = self.CalculateScores(
            DataLoaderObject=TestLoader,
            NumMaskRuns=NumMaskRuns,
            Seed=Seed,
        )

        # Threshold priority:
        # 1. explicitly provided threshold
        # 2. stored threshold
        # 3. percentile of current scores
        if Threshold is not None:
            UsedThreshold = float(Threshold)

        elif self.AnomalyLimit is not None:
            UsedThreshold = float(self.AnomalyLimit)
        else:
            UsedThreshold = float(np.percentile(AllScores, Percentile))

        self.AnomalyLimit = UsedThreshold

        AnomalyBinary = (AllScores >= UsedThreshold).astype(int)
        return AllScores, AnomalyBinary


    def Evaluate(
        self,
        TrueLabels: np.ndarray,
        Scores: np.ndarray,
        PredictedLabels: np.ndarray,
    ) -> dict[str, float]:
        """
        Evaluates already calculated AnoFormer scores and predictions
        """

        TrueLabels = np.asarray(TrueLabels).astype(int).reshape(-1)
        Scores = np.asarray(Scores).reshape(-1)
        PredictedLabels = np.asarray(PredictedLabels).astype(int).reshape(-1)

        if not (len(TrueLabels) == len(Scores) == len(PredictedLabels)):
            raise ValueError(
                "TrueLabels, Scores and PredictedLabels must have "
                "the same length."
            )

        TruePositive = int(((PredictedLabels == 1) & (TrueLabels == 1)).sum())
        TrueNegative = int(((PredictedLabels == 0) & (TrueLabels == 0)).sum())
        FalsePositive = int(((PredictedLabels == 1) & (TrueLabels == 0)).sum())
        FalseNegative = int(((PredictedLabels == 0) & (TrueLabels == 1)).sum())

        Precision = (TruePositive / (TruePositive + FalsePositive + 1e-9))
        Recall = (TruePositive / (TruePositive + FalseNegative + 1e-9))
        F1Score = (2 * Precision * Recall / (Precision + Recall + 1e-9))
        Accuracy = (TruePositive + TrueNegative) / (TruePositive + TrueNegative + FalsePositive + FalseNegative + 1e-9)
        FalseNegativeRate = (FalseNegative / (FalseNegative + TruePositive + 1e-9))

        ROC_AUC = roc_auc_score(TrueLabels, Scores)
        PR_AUC = average_precision_score(TrueLabels, Scores)
        ConfMatrix = confusion_matrix(TrueLabels, PredictedLabels)

        Results = {
            "Accuracy": round(Accuracy, 4),
            "Precision": round(Precision, 4),
            "Recall": round(Recall, 4),
            "F1Score": round(F1Score, 4),
            "FalseNegativeRate": round(FalseNegativeRate, 4),
            "Threshold": round(float(self.AnomalyLimit), 6),
            "TruePositive": TruePositive,
            "TrueNegative": TrueNegative,
            "FalsePositive": FalsePositive,
            "FalseNegative": FalseNegative,
            "ROC_AUC": round(ROC_AUC, 4),
            "PR_AUC": round(PR_AUC, 4),
            "ConfusionMatrix\n": ConfMatrix,
        }

        for Name, Value in Results.items():
            print(f"  {Name:<20}: {Value}")

        return Results

    def Save(self, Path: str) -> None:
        torch.save({
            "GeneratorState":     self.Generator.state_dict(),
            "DiscriminatorState": self.Discriminator.state_dict(),
            "TrainingHistory":    self.TrainingHistory,
            "AnomalyLimit":       self.AnomalyLimit,
        }, Path)
        print(f"Model saved to: {Path}")

    def Load(self, Path: str) -> None:
        State = torch.load(Path, map_location=self.Device, weights_only=False)
        self.Generator.load_state_dict(State["GeneratorState"])
        self.Discriminator.load_state_dict(State["DiscriminatorState"])
        self.TrainingHistory = State["TrainingHistory"]
        self.AnomalyLimit    = State["AnomalyLimit"]
        print(f"Model loaded from: {Path}")
