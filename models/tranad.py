import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadScaledDotProductAttention(nn.Module):
    """
    Multi-Head Self-Attention.
    """
    def __init__(
        self,
        ModelDimension: int,
        NumHeads:       int,
        DropoutRate:    float = 0.1,
    ):
        super().__init__()
        assert ModelDimension % NumHeads == 0, \
            "d_model muss durch NumHeads teilbar sein / d_model must be divisible by NumHeads"

        self.ModelDimension = ModelDimension 
        self.NumHeads       = NumHeads 
        self.HeadDimension  = ModelDimension // NumHeads 
        self.ScalingFactor  = math.sqrt(self.HeadDimension) 

        self.QueryProjection  = nn.Linear(ModelDimension, ModelDimension, bias=False)
        self.KeyProjection    = nn.Linear(ModelDimension, ModelDimension, bias=False)
        self.ValueProjection  = nn.Linear(ModelDimension, ModelDimension, bias=False)
        self.OutputProjection = nn.Linear(ModelDimension, ModelDimension, bias=False)
        self.AttentionDropout = nn.Dropout(DropoutRate) 

        self.StoredAttentionWeights: Optional[torch.Tensor] = None 

    def _SplitIntoHeads(self, Tensor: torch.Tensor) -> torch.Tensor:

        B, T, _ = Tensor.shape 
        Tensor   = Tensor.view(B, T, self.NumHeads, self.HeadDimension)
        return Tensor.transpose(1, 2)

    def forward(
        self,
        QueryInput:    torch.Tensor,
        KeyInput:      torch.Tensor,
        ValueInput:    torch.Tensor,
        AttentionMask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = QueryInput.size(0)

        Q = self._SplitIntoHeads(self.QueryProjection(QueryInput))
        K = self._SplitIntoHeads(self.KeyProjection(KeyInput))
        V = self._SplitIntoHeads(self.ValueProjection(ValueInput))

        AttentionLogits = torch.matmul(Q, K.transpose(-2, -1)) / self.ScalingFactor

        if AttentionMask is not None:
            AttentionLogits = AttentionLogits.masked_fill(AttentionMask == 0, float("-inf"))

        AttentionWeights = F.softmax(AttentionLogits, dim=-1)
        self.StoredAttentionWeights = AttentionWeights.detach()
        AttentionWeights = self.AttentionDropout(AttentionWeights)

        ContextVector = torch.matmul(AttentionWeights, V)
        ContextVector = ContextVector.transpose(1, 2).contiguous()
        ContextVector = ContextVector.view(B, -1, self.ModelDimension)

        return self.OutputProjection(ContextVector)


class PositionwiseFeedForwardNetwork(nn.Module):

    def __init__(
        self,
        ModelDimension:   int,
        ForwardDimension: int,
        DropoutRate:      float = 0.1,
    ):
        super().__init__()
        self.FullyConnectedLayer1 = nn.Linear(ModelDimension, ForwardDimension)
        self.ActivationFunction   = nn.GELU()
        self.DropoutAfterGELU     = nn.Dropout(DropoutRate)
        self.FullyConnectedLayer2 = nn.Linear(ForwardDimension, ModelDimension)
        self.DropoutAfterLinear2  = nn.Dropout(DropoutRate)

    def forward(self, InputTensor: torch.Tensor) -> torch.Tensor:
        HiddenState  = self.FullyConnectedLayer1(InputTensor)
        HiddenState  = self.ActivationFunction(HiddenState)
        HiddenState  = self.DropoutAfterGELU(HiddenState)
        OutputTensor = self.FullyConnectedLayer2(HiddenState)
        OutputTensor = self.DropoutAfterLinear2(OutputTensor)
        return OutputTensor 


class ResidualConnectionWithLayerNorm(nn.Module):

    def __init__(self, ModelDimension: int, DropoutRate: float = 0.1):
        super().__init__()
        self.LayerNormalization = nn.LayerNorm(ModelDimension) 
        self.ResidualDropout    = nn.Dropout(DropoutRate)

    def forward(self, ResidualInput: torch.Tensor, SubLayerOutput: torch.Tensor) -> torch.Tensor:
        DroppedOutput = self.ResidualDropout(SubLayerOutput) 
        return self.LayerNormalization(ResidualInput + DroppedOutput)


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(
        self,
        ModelDimension:    int, 
        MaxSequenceLength: int = 5000,
        DropoutRate:       float = 0.1,
    ):
        super().__init__()
        self.PositionalDropout = nn.Dropout(DropoutRate)

        PositionEncodingMatrix = torch.zeros(MaxSequenceLength, ModelDimension) 
        PositionIndices = torch.arange(0, MaxSequenceLength).float().unsqueeze(1)
        EvenDimIndices  = torch.arange(0, ModelDimension, 2).float()
        DivisionTerms   = torch.exp(EvenDimIndices * (-math.log(10000.0) / ModelDimension))

        PositionEncodingMatrix[:, 0::2] = torch.sin(PositionIndices * DivisionTerms)
        PositionEncodingMatrix[:, 1::2] = torch.cos(PositionIndices * DivisionTerms)
        PositionEncodingMatrix = PositionEncodingMatrix.unsqueeze(0)

        self.register_buffer("PositionEncodingBuffer", PositionEncodingMatrix)

    def forward(self, EmbeddedInput: torch.Tensor) -> torch.Tensor:
        SeqLen = EmbeddedInput.size(1)
        EncodedInput = EmbeddedInput + self.PositionEncodingBuffer[:, :SeqLen, :]
        return self.PositionalDropout(EncodedInput)


class InputEmbeddingLayer(nn.Module):

    def __init__(self, NumFeatures: int, ModelDimension: int):
        super().__init__()
        self.FeatureProjectionLayer = nn.Linear(NumFeatures, ModelDimension)
        self.EmbeddingNorm          = nn.LayerNorm(ModelDimension)

    def forward(self, RawInputTensor: torch.Tensor) -> torch.Tensor:
        ProjectedTensor = self.FeatureProjectionLayer(RawInputTensor)
        return self.EmbeddingNorm(ProjectedTensor)


class OutputProjectionLayer(nn.Module):
    def __init__(self, ModelDimension: int, NumFeatures: int):
        super().__init__()
        self.OutputLinearLayer = nn.Linear(ModelDimension, NumFeatures)

    def forward(self, TransformerOutput: torch.Tensor) -> torch.Tensor:
        return self.OutputLinearLayer(TransformerOutput)


class TranADEncoderLayer(nn.Module):

    def __init__(
        self,
        ModelDimension:   int,
        NumHeads:         int,
        ForwardDimension: int,
        DropoutRate:      float = 0.1,
    ):
        super().__init__()
        self.SelfAttentionModule   = MultiHeadScaledDotProductAttention(
            ModelDimension, NumHeads, DropoutRate
        )
        self.SelfAttentionAddNorm  = ResidualConnectionWithLayerNorm(
            ModelDimension, DropoutRate
        )
        self.FeedForwardModule     = PositionwiseFeedForwardNetwork(
            ModelDimension, ForwardDimension, DropoutRate
        )
        self.FeedForwardAddNorm    = ResidualConnectionWithLayerNorm(
            ModelDimension, DropoutRate
        )

    def forward(
        self,
        InputTensor:   torch.Tensor,
        AttentionMask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        SelfAttOutput   = self.SelfAttentionModule(
            InputTensor, InputTensor, InputTensor, AttentionMask 
        )
        AfterSelfAtt    = self.SelfAttentionAddNorm(InputTensor, SelfAttOutput)

        FeedForwardOut  = self.FeedForwardModule(AfterSelfAtt) 
        AfterFFN        = self.FeedForwardAddNorm(AfterSelfAtt, FeedForwardOut)
        return AfterFFN


class TranADEncoderStack(nn.Module):

    def __init__(
        self,
        NumLayers:        int,
        ModelDimension:   int,
        NumHeads:         int,
        ForwardDimension: int,
        DropoutRate:      float = 0.1,
    ):
        super().__init__()
        self.EncoderLayerList = nn.ModuleList([
            TranADEncoderLayer(ModelDimension, NumHeads, ForwardDimension, DropoutRate)
            for _ in range(NumLayers)
        ])
        self.FinalLayerNorm = nn.LayerNorm(ModelDimension)

    def forward(
        self,
        InputTensor:   torch.Tensor,
        AttentionMask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        CurrentTensor = InputTensor
        for EncoderLayer in self.EncoderLayerList:
            CurrentTensor = EncoderLayer(CurrentTensor, AttentionMask)
        return self.FinalLayerNorm(CurrentTensor)


class TranADDecoderLayer(nn.Module):

    def __init__(
        self,
        ModelDimension:   int,
        NumHeads:         int,
        ForwardDimension: int,
        DropoutRate:      float = 0.1,
    ):
        super().__init__()
        self.SelfAttentionModule    = MultiHeadScaledDotProductAttention(
            ModelDimension, NumHeads, DropoutRate
        )
        self.SelfAttentionAddNorm   = ResidualConnectionWithLayerNorm(ModelDimension, DropoutRate)

        self.CrossAttentionModule   = MultiHeadScaledDotProductAttention(
            ModelDimension, NumHeads, DropoutRate
        )
        self.CrossAttentionAddNorm  = ResidualConnectionWithLayerNorm(ModelDimension, DropoutRate)

        self.FeedForwardModule      = PositionwiseFeedForwardNetwork(
            ModelDimension, ForwardDimension, DropoutRate
        )
        self.FeedForwardAddNorm     = ResidualConnectionWithLayerNorm(ModelDimension, DropoutRate)

    def forward(
        self,
        TargetTensor:  torch.Tensor,
        EncoderMemory: torch.Tensor,
        TargetMask:    Optional[torch.Tensor] = None,
        MemoryMask:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        SelfAttOutput  = self.SelfAttentionModule(
            TargetTensor, TargetTensor, TargetTensor, TargetMask
        )
        AfterSelfAtt   = self.SelfAttentionAddNorm(TargetTensor, SelfAttOutput)

        CrossAttOutput = self.CrossAttentionModule(
            AfterSelfAtt, EncoderMemory, EncoderMemory, MemoryMask
        )
        AfterCrossAtt  = self.CrossAttentionAddNorm(AfterSelfAtt, CrossAttOutput)

        FeedForwardOut = self.FeedForwardModule(AfterCrossAtt)
        AfterFFN       = self.FeedForwardAddNorm(AfterCrossAtt, FeedForwardOut)
        return AfterFFN


class TranADDecoderStack(nn.Module):

    def __init__(
        self,
        NumLayers:        int,
        ModelDimension:   int,
        NumHeads:         int,
        ForwardDimension: int,
        DropoutRate:      float = 0.1,
    ):
        super().__init__()
        self.DecoderLayerList = nn.ModuleList([
            TranADDecoderLayer(ModelDimension, NumHeads, ForwardDimension, DropoutRate)
            for _ in range(NumLayers)
        ])
        self.FinalLayerNorm   = nn.LayerNorm(ModelDimension)

    def forward(
        self,
        TargetTensor:  torch.Tensor,
        EncoderMemory: torch.Tensor,
        TargetMask:    Optional[torch.Tensor] = None,
        MemoryMask:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        CurrentTensor = TargetTensor
        for DecoderLayer in self.DecoderLayerList:
            CurrentTensor = DecoderLayer(
                CurrentTensor, EncoderMemory, TargetMask, MemoryMask
            )
        return self.FinalLayerNorm(CurrentTensor)



class TranADModel(nn.Module):
    """ Core network of the TranAD model"""

    def __init__(
        self,
        NumFeatures:         int,
        ModelDimension:      int   = 64,
        NumHeads:            int   = 4,
        NumEncoderLayers:    int   = 3,
        NumDecoderLayers:    int   = 3,
        ForwardDimension:    int   = 256,
        DropoutRate:         float = 0.1,
        MaxSequenceLength:   int   = 5000,
        LearningRate:        float = 1e-4,
        WeightDecay:         float = 1e-5,
        LRSchedulerPatience: int   = 10,
        LRSchedulerFactor:   float = 0.5,
    ):
        super().__init__()
        self.NumFeatures    = NumFeatures
        self.ModelDimension = ModelDimension

        self.InputEmbeddingLayer     = InputEmbeddingLayer(NumFeatures, ModelDimension)
        self.PositionalEncodingLayer = SinusoidalPositionalEncoding(
            ModelDimension, MaxSequenceLength, DropoutRate
        )

        self.EncoderStack = TranADEncoderStack(
            NumEncoderLayers, ModelDimension, NumHeads, ForwardDimension, DropoutRate
        )

        self.DecoderInputNorm = nn.LayerNorm(ModelDimension)

        self.DecoderStack1    = TranADDecoderStack(
            NumDecoderLayers, ModelDimension, NumHeads, ForwardDimension, DropoutRate
        )
        self.OutputProjection1 = OutputProjectionLayer(ModelDimension, NumFeatures)

        self.DecoderStack2    = TranADDecoderStack(
            NumDecoderLayers, ModelDimension, NumHeads, ForwardDimension, DropoutRate
        )
        self.OutputProjection2 = OutputProjectionLayer(ModelDimension, NumFeatures)
        
        self.Optimizer = torch.optim.Adam(
            self.parameters(),
            lr           = LearningRate,
            weight_decay = WeightDecay,
        )

        self.LRScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.Optimizer,
            mode     = "min",
            factor   = LRSchedulerFactor,
            patience = LRSchedulerPatience,
        )

        self.ReconstructionLoss = nn.MSELoss(reduction="none")

        self._InitializeWeights()
        ParamCount = sum(p.numel() for p in self.parameters())
        print(
            f"TranADModel: {ParamCount:,} Parameter | "
            f"d_model={ModelDimension}, H={NumHeads}, Enc={NumEncoderLayers}, Dec×2={NumDecoderLayers}"
        )

    def _InitializeWeights(self) -> None:
        for Module in self.modules():
            if isinstance(Module, nn.Linear):
                nn.init.xavier_uniform_(Module.weight)
                if Module.bias is not None:
                    nn.init.zeros_(Module.bias)

    def _EmbedAndEncode(self, InputTensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Embedded      = self.InputEmbeddingLayer(InputTensor)
        WithPE        = self.PositionalEncodingLayer(Embedded)
        EncoderMemory = self.EncoderStack(WithPE)
        return WithPE, EncoderMemory

    def forward(
        self,
        InputTensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        EmbeddedWithPE, EncoderMemory = self._EmbedAndEncode(InputTensor)

        NormedDecInput1  = self.DecoderInputNorm(EmbeddedWithPE)
        DecoderOutput1   = self.DecoderStack1(NormedDecInput1, EncoderMemory)
        Reconstruction1  = self.OutputProjection1(DecoderOutput1)

        EmbeddedRec1     = self.InputEmbeddingLayer(Reconstruction1.detach())
        WithPERec1       = self.PositionalEncodingLayer(EmbeddedRec1)
        NormedDecInput2  = self.DecoderInputNorm(WithPERec1)

        DecoderOutput2   = self.DecoderStack2(NormedDecInput2, EncoderMemory)
        Reconstruction2  = self.OutputProjection2(DecoderOutput2)

        return Reconstruction1, Reconstruction2

    def ComputeTranADLoss(
        self,
        OriginalInput:   torch.Tensor,
        Reconstruction1: torch.Tensor,
        Reconstruction2: torch.Tensor,
        FocusScore:      float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TranAD loss function.
        L = (1/φ) · ||x - x1||² + φ · ||x - x2||²
        """
        Error1 = self.ReconstructionLoss(Reconstruction1, OriginalInput)
        Error2 = self.ReconstructionLoss(Reconstruction2, OriginalInput)

        PhiSafe        = max(FocusScore, 1e-6) 
        WeightedError1 = Error1 * (1.0 / PhiSafe)
        WeightedError2 = Error2 * float(FocusScore)
        CombinedError  = WeightedError1 + WeightedError2

        TotalLoss        = CombinedError.mean()
        ErrorPerTimestep = Error2.mean(dim=-1)
        return TotalLoss, ErrorPerTimestep


# TranAD Main Model combines everything
class TranADMainModel:
    """
    TranAD: A Transformer-based Anomaly Detection Model for Multivariate Time Series
    """

    def __init__(
        self,
        NumFeatures:         int,
        ModelDimension:      int   = 64,
        NumHeads:            int   = 4,
        NumEncoderLayers:    int   = 3,
        NumDecoderLayers:    int   = 3,
        ForwardDimension:    int   = 256,
        DropoutRate:         float = 0.1,
        LearningRate:        float = 1e-4,
        WeightDecay:         float = 1e-5,
        LRSchedulerPatience: int   = 10,
        FocusGrowthRate:     float = 1.2,
        InitialFocusScore:   float = 1.0,
        MaxFocusScore:       float = 100.0,
        AnomalyPercentile:   float = 95.0,
        Device:              Optional[torch.device] = None,
    ):
        self.Device          = Device if Device is not None else DEVICE
        self.FocusScore      = InitialFocusScore 
        self.FocusGrowthRate = FocusGrowthRate
        self.MaxFocusScore   = MaxFocusScore 
        self.AnomalyPercentile = AnomalyPercentile
        self.AnomalyThreshold  = None 

        self.Network = TranADModel(
            NumFeatures          = NumFeatures,
            ModelDimension       = ModelDimension,
            NumHeads             = NumHeads,
            NumEncoderLayers     = NumEncoderLayers,
            NumDecoderLayers     = NumDecoderLayers,
            ForwardDimension     = ForwardDimension,
            DropoutRate          = DropoutRate,
            LearningRate         = LearningRate,
            WeightDecay          = WeightDecay,
            LRSchedulerPatience  = LRSchedulerPatience,
        ).to(self.Device)

        self.TrainingHistory: Dict[str, List[float]] = {
            "TotalLoss":  [],
            "FocusScore": [],
        }

    def TrainingStep(self, WindowBatch: torch.Tensor) -> float:
        
        self.Network.train()
        WindowBatch = WindowBatch.to(self.Device)

        self.Network.Optimizer.zero_grad()
        Reconstruction1, Reconstruction2 = self.Network(WindowBatch)
        TotalLoss, _ = self.Network.ComputeTranADLoss(
            WindowBatch, Reconstruction1, Reconstruction2, self.FocusScore
        )
        TotalLoss.backward()
        nn.utils.clip_grad_norm_(self.Network.parameters(), max_norm=1.0)
        self.Network.Optimizer.step()
        return TotalLoss.item()


    def TrainModel(
        self,
        TrainingLoader: DataLoader,
        NumEpochs:      int = 50,
        PrintInterval:  int = 5,
    ) -> None:

        for Epoch in range(1, NumEpochs + 1):
            EpochLossValues: List[float] = []

            for WindowBatch, _ in TrainingLoader:
                BatchLoss = self.TrainingStep(WindowBatch)
                EpochLossValues.append(BatchLoss)

            AverageLoss = float(np.mean(EpochLossValues))
            self.TrainingHistory["TotalLoss"].append(AverageLoss)
            self.TrainingHistory["FocusScore"].append(self.FocusScore)

            self.Network.LRScheduler.step(AverageLoss)

            self.FocusScore = min(self.FocusScore * self.FocusGrowthRate, self.MaxFocusScore)

            if Epoch % PrintInterval == 0 or Epoch == 1:
                CurrentLR = self.Network.Optimizer.param_groups[0]["lr"]
                print(
                    f"Epoche {Epoch:>4}/{NumEpochs} | "
                    f"Loss={AverageLoss:.5f} | "
                    f"φ={self.FocusScore:.2f} | "
                    f"LR={CurrentLR:.2e}"
                )

        self.Network.eval()
        print("Training complete.")


    @torch.no_grad()
    def CalculateScores(
        self,
        DataLoaderObject: DataLoader,
    ) -> np.ndarray:
        """
        Calculates one anomaly score per ECG record
        """

        self.Network.eval()
        AllWindowScores: List[np.ndarray] = []

        for WindowBatch, _ in DataLoaderObject:
            WindowBatch = WindowBatch.to(
                self.Device
            )

            _, Reconstruction2 = self.Network(
                WindowBatch
            )

            WindowScores = (
                # Compute the mean squared error across features, then across time steps
                (Reconstruction2 - WindowBatch) ** 2).mean(dim=-1).mean(dim=-1)

            AllWindowScores.append(WindowScores.cpu().numpy())

        Scores = np.concatenate(AllWindowScores, axis=0)
        return Scores


    @torch.no_grad()
    def DetectAnomalies(
        self,
        TestLoader: DataLoader,
        Threshold: Optional[float] = None,
        Percentile: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates anomaly scores and applies a threshold
        """

        AllScores = self.CalculateScores(DataLoaderObject=TestLoader)

        if Threshold is not None:
            ThresholdValue = float(Threshold)

        elif self.AnomalyThreshold is not None:
            ThresholdValue = float(self.AnomalyThreshold)

        else:
            UsedPercentile = (Percentile if Percentile is not None else self.AnomalyPercentile)
            ThresholdValue = float(np.percentile( AllScores, UsedPercentile))

        self.AnomalyThreshold = ThresholdValue
        BinaryLabels = (AllScores >= ThresholdValue).astype(int)

        return AllScores, BinaryLabels

    def Evaluate(
        self,
        TrueLabels: np.ndarray,
        Scores: np.ndarray,
        PredictedLabels: np.ndarray,
    ) -> dict[str, float | int]:

        TrueLabels = np.asarray(TrueLabels).astype(int).reshape(-1)
        Scores = np.asarray(Scores).reshape(-1)
        PredictedLabels = np.asarray(PredictedLabels).astype(int).reshape(-1)

        if not (len(TrueLabels) == len(Scores) == len(PredictedLabels)):
            raise ValueError(
                "TrueLabels, Scores and PredictedLabels "
                "must have the same length."
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

        ThresholdValue = (
            round(float(self.AnomalyThreshold), 6)
            if self.AnomalyThreshold is not None
            else None
        )

        ROC_AUC = roc_auc_score(TrueLabels, Scores)
        PR_AUC = average_precision_score(TrueLabels, Scores)
        ConfMatrix = confusion_matrix(TrueLabels, PredictedLabels)

        Results = {
            "Accuracy": round(Accuracy, 4),
            "Precision": round(Precision, 4),
            "Recall": round(Recall, 4),
            "F1Score": round(F1Score, 4),
            "FalseNegativeRate": round(FalseNegativeRate, 4),
            "Threshold": ThresholdValue,
            "TruePositive": TruePositive,
            "TrueNegative": TrueNegative,
            "FalsePositive": FalsePositive,
            "FalseNegative": FalseNegative,
            "ROC_AUC": round(ROC_AUC, 4),
            "PR_AUC": round(PR_AUC, 4),
            "ConfusionMatrix\n": ConfMatrix,
        }

        for MetricName, MetricValue in Results.items():
            print(
                f"  {MetricName:<20}: "
                f"{MetricValue}"
            )

        return Results

    def Save(self, FilePath: str) -> None:
        torch.save({
            "NetworkState":      self.Network.state_dict(),
            "FocusScore":        self.FocusScore,
            "TrainingHistory":   self.TrainingHistory,
            "AnomalyThreshold":  self.AnomalyThreshold,
        }, FilePath)
        print(f"Model saved  {FilePath}")

    def Load(self, FilePath: str) -> None:
        Checkpoint = torch.load(FilePath, map_location=self.Device, weights_only=False)
        self.Network.load_state_dict(Checkpoint["NetworkState"])
        self.FocusScore       = Checkpoint["FocusScore"]
        self.TrainingHistory  = Checkpoint["TrainingHistory"]
        self.AnomalyThreshold = Checkpoint["AnomalyThreshold"]
        self.Network.eval()
        print(f"Model loaded  {FilePath}")


