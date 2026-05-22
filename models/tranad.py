import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


"""TranAD: Transformer-basierte Anomalieerkennung – Modellarchitektur und Bausteine."""

# ============================================================
# Elementare Bausteine / Elementary Building Blocks
# MultiHeadScaledDotProductAttention, PositionwiseFeedForwardNetwork,
# ResidualConnectionWithLayerNorm, SinusoidalPositionalEncoding,
# InputEmbeddingLayer, OutputProjectionLayer
# ============================================================
class MultiHeadScaledDotProductAttention(nn.Module):
    """
    Multi-Head Scaled Dot-Product Attention von Grund auf.
    / Multi-Head Scaled Dot-Product Attention from scratch.
    Formel / Formula: Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V
    """

    def __init__(
        self,
        ModelDimension: int,          # d_model – Gesamt-Attention-Dimension / Total attention dimension
        NumHeads:       int,          # H – Anzahl paralleler Köpfe / Number of parallel heads
        DropoutRate:    float = 0.1,  # Dropout auf Attention-Gewichten / Dropout on attention weights
    ):
        super().__init__()
        assert ModelDimension % NumHeads == 0, \
            "d_model muss durch NumHeads teilbar sein / d_model must be divisible by NumHeads"

        self.ModelDimension = ModelDimension              # Gesamt-Dimension speichern / Store total dimension
        self.NumHeads       = NumHeads                    # Anzahl Heads / Number of heads
        self.HeadDimension  = ModelDimension // NumHeads  # d_k = d_model / H pro Kopf / per head
        self.ScalingFactor  = math.sqrt(self.HeadDimension)  # √d_k – verhindert zu kleine Gradienten / prevents vanishing gradients

        # ── Explizite Projektionsmatrizen / Explicit projection matrices ──
        self.QueryProjection  = nn.Linear(ModelDimension, ModelDimension, bias=False)  # W_Q ∈ ℝ^{d×d}
        self.KeyProjection    = nn.Linear(ModelDimension, ModelDimension, bias=False)  # W_K ∈ ℝ^{d×d}
        self.ValueProjection  = nn.Linear(ModelDimension, ModelDimension, bias=False)  # W_V ∈ ℝ^{d×d}
        self.OutputProjection = nn.Linear(ModelDimension, ModelDimension, bias=False)  # W_O: konkatenierte Heads → d_model

        self.AttentionDropout = nn.Dropout(DropoutRate)   # Dropout auf normierten Attention-Gewichten / Dropout on normalized weights

        # Attention-Gewichte für Explainability merken / Store attention weights for explainability
        self.StoredAttentionWeights: Optional[torch.Tensor] = None  # (B, H, T_q, T_k) oder None

    def _SplitIntoHeads(self, Tensor: torch.Tensor) -> torch.Tensor:
        """
        (Batch, Seq, d_model) → (Batch, H, Seq, d_k)
        Teilt d_model in H Köpfe auf / Splits d_model into H heads.
        """
        B, T, _ = Tensor.shape                                        # Batch, Sequenz, Dimension
        Tensor   = Tensor.view(B, T, self.NumHeads, self.HeadDimension)  # (B, T, H, d_k)
        return Tensor.transpose(1, 2)                                  # (B, H, T, d_k) für Matmul / for matmul

    def forward(
        self,
        QueryInput:    torch.Tensor,                    # Q: (B, T_q, d_model)
        KeyInput:      torch.Tensor,                    # K: (B, T_k, d_model)
        ValueInput:    torch.Tensor,                    # V: (B, T_k, d_model)
        AttentionMask: Optional[torch.Tensor] = None,  # Optionale Maske (0=maskiert) / Optional mask
    ) -> torch.Tensor:                                  # Ausgabe: (B, T_q, d_model)
        B = QueryInput.size(0)                          # Batch-Größe / Batch size

        # Schritt 1: Lineare Projektion aller drei Matrizen / Step 1: Project all three
        Q = self._SplitIntoHeads(self.QueryProjection(QueryInput))    # (B, H, T_q, d_k)
        K = self._SplitIntoHeads(self.KeyProjection(KeyInput))        # (B, H, T_k, d_k)
        V = self._SplitIntoHeads(self.ValueProjection(ValueInput))    # (B, H, T_k, d_k)

        # Schritt 2: Skaliertes Skalarprodukt / Step 2: Scaled dot-product
        AttentionLogits = torch.matmul(Q, K.transpose(-2, -1)) / self.ScalingFactor  # (B, H, T_q, T_k)

        if AttentionMask is not None:                                  # Maske anwenden falls vorhanden / Apply mask if given
            AttentionLogits = AttentionLogits.masked_fill(AttentionMask == 0, float("-inf"))

        # Schritt 3: Softmax + Dropout / Step 3: Softmax + dropout
        AttentionWeights = F.softmax(AttentionLogits, dim=-1)          # (B, H, T_q, T_k) – normiert / normalized
        self.StoredAttentionWeights = AttentionWeights.detach()        # Für Explainability ablegen / Store for explainability
        AttentionWeights = self.AttentionDropout(AttentionWeights)     # Dropout auf Gewichten / Dropout on weights

        # Schritt 4: Kontextvektor berechnen & Heads zusammenführen / Step 4: Context & merge heads
        ContextVector = torch.matmul(AttentionWeights, V)              # (B, H, T_q, d_k)
        ContextVector = ContextVector.transpose(1, 2).contiguous()     # (B, T_q, H, d_k)
        ContextVector = ContextVector.view(B, -1, self.ModelDimension) # (B, T_q, d_model) konkateniert / concatenated

        # Schritt 5: Ausgabeprojektion / Step 5: Output projection
        return self.OutputProjection(ContextVector)                    # (B, T_q, d_model)

# ─────────────────────────────────────────────────────────────────────

class PositionwiseFeedForwardNetwork(nn.Module):
    """
    Positions-unabhängiges Feed-Forward-Netzwerk (FFN).
    / Position-wise Feed-Forward Network (FFN).
    Architektur / Architecture: Linear(d→d_ff) → GELU → Dropout → Linear(d_ff→d) → Dropout
    """

    def __init__(
        self,
        ModelDimension:   int,          # Eingabe- und Ausgabedimension d_model / Input and output dimension
        ForwardDimension: int,          # Innere Dimension d_ff (typisch 4×d_model) / Inner dim (typically 4×d_model)
        DropoutRate:      float = 0.1,  # Dropout-Rate / Dropout rate
    ):
        super().__init__()
        # Explizite Layer-Definitionen / Explicit layer definitions
        self.FullyConnectedLayer1 = nn.Linear(ModelDimension, ForwardDimension)  # W_1: Expansion d → d_ff
        self.ActivationFunction   = nn.GELU()                                    # GELU: glattere Gradienten als ReLU / smoother gradients
        self.DropoutAfterGELU     = nn.Dropout(DropoutRate)                      # Dropout nach Aktivierung / Dropout after activation
        self.FullyConnectedLayer2 = nn.Linear(ForwardDimension, ModelDimension)  # W_2: Projektion d_ff → d
        self.DropoutAfterLinear2  = nn.Dropout(DropoutRate)                      # Dropout nach zweiter Schicht / Dropout after second layer

    def forward(self, InputTensor: torch.Tensor) -> torch.Tensor:
        # (B, T, d_model) → expandieren → aktivieren → komprimieren → (B, T, d_model)
        HiddenState  = self.FullyConnectedLayer1(InputTensor)   # Expansion / Expansion
        HiddenState  = self.ActivationFunction(HiddenState)     # Nicht-Linearität / Non-linearity
        HiddenState  = self.DropoutAfterGELU(HiddenState)       # Regularisierung / Regularization
        OutputTensor = self.FullyConnectedLayer2(HiddenState)   # Komprimierung / Compression
        OutputTensor = self.DropoutAfterLinear2(OutputTensor)   # Zweiter Dropout / Second dropout
        return OutputTensor                                      # (B, T, d_model)

# ─────────────────────────────────────────────────────────────────────

class ResidualConnectionWithLayerNorm(nn.Module):
    """
    Add & Norm (Post-Norm): LayerNorm(x + Dropout(Sublayer(x)))
    Stabilisiert Gradienten durch Residualverbindung und Normierung.
    / Stabilizes gradients through residual connection and normalization.
    """

    def __init__(self, ModelDimension: int, DropoutRate: float = 0.1):
        super().__init__()
        self.LayerNormalization = nn.LayerNorm(ModelDimension)  # Normierung über Feature-Dimension / Norm over feature dim
        self.ResidualDropout    = nn.Dropout(DropoutRate)       # Dropout auf Sublayer-Ausgabe / Dropout on sublayer output

    def forward(self, ResidualInput: torch.Tensor, SubLayerOutput: torch.Tensor) -> torch.Tensor:
        DroppedOutput = self.ResidualDropout(SubLayerOutput)              # Dropout auf Sublayer / Dropout on sublayer
        return self.LayerNormalization(ResidualInput + DroppedOutput)     # Add & Norm

# ─────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidale Positions-Kodierung nach Vaswani et al. (2017).
    / Sinusoidal positional encoding per Vaswani et al. (2017).
    PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
    Nicht lernbar, deterministisch / Non-learnable, deterministic.
    """

    def __init__(
        self,
        ModelDimension:    int,           # d_model
        MaxSequenceLength: int = 5000,    # Maximale Sequenzlänge für vorberechnete PE / Max seq len for precomputed PE
        DropoutRate:       float = 0.1,   # Dropout nach PE-Addition / Dropout after PE addition
    ):
        super().__init__()
        self.PositionalDropout = nn.Dropout(DropoutRate)                  # Dropout / Dropout rate

        # PE-Matrix vorberechnen und als Buffer registrieren / Precompute and register as buffer
        PositionEncodingMatrix = torch.zeros(MaxSequenceLength, ModelDimension)   # (T_max, d)
        PositionIndices = torch.arange(0, MaxSequenceLength).float().unsqueeze(1)  # (T_max, 1)
        EvenDimIndices  = torch.arange(0, ModelDimension, 2).float()              # Gerade Dim / Even dims
        DivisionTerms   = torch.exp(EvenDimIndices * (-math.log(10000.0) / ModelDimension))  # 1/10000^{2i/d}

        PositionEncodingMatrix[:, 0::2] = torch.sin(PositionIndices * DivisionTerms)  # Sinus auf gerade / Sine on even
        PositionEncodingMatrix[:, 1::2] = torch.cos(PositionIndices * DivisionTerms)  # Cosinus auf ungerade / Cosine on odd
        PositionEncodingMatrix = PositionEncodingMatrix.unsqueeze(0)       # (1, T_max, d) – Batch-Dim hinzufügen / Add batch dim

        self.register_buffer("PositionEncodingBuffer", PositionEncodingMatrix)  # Kein Parameter, aber persistent / Not a param but persistent

    def forward(self, EmbeddedInput: torch.Tensor) -> torch.Tensor:
        # EmbeddedInput: (B, T, d_model)
        SeqLen = EmbeddedInput.size(1)                                   # Aktuelle Sequenzlänge / Current seq length
        EncodedInput = EmbeddedInput + self.PositionEncodingBuffer[:, :SeqLen, :]  # PE addieren / Add PE
        return self.PositionalDropout(EncodedInput)                      # Dropout auf Summe / Dropout on sum

# ─────────────────────────────────────────────────────────────────────

class InputEmbeddingLayer(nn.Module):
    """
    Projiziert Rohdaten (B, T, F) → Modell-Raum (B, T, d_model).
    / Projects raw data (B, T, F) → model space (B, T, d_model).
    Lineare Projektion + sofortige LayerNorm für stabile Einbettung.
    / Linear projection + immediate LayerNorm for stable embedding.
    """

    def __init__(self, NumFeatures: int, ModelDimension: int):
        super().__init__()
        self.FeatureProjectionLayer = nn.Linear(NumFeatures, ModelDimension)  # W_e: F → d_model
        self.EmbeddingNorm          = nn.LayerNorm(ModelDimension)            # Normierung direkt nach Projektion / Norm right after projection

    def forward(self, RawInputTensor: torch.Tensor) -> torch.Tensor:
        # (B, T, F) → lineare Projektion → Normierung → (B, T, d_model)
        ProjectedTensor = self.FeatureProjectionLayer(RawInputTensor)  # Lineare Projektion / Linear projection
        return self.EmbeddingNorm(ProjectedTensor)                     # Normiert zurückgeben / Return normalized


class OutputProjectionLayer(nn.Module):
    """
    Projiziert Transformer-Ausgabe zurück in Feature-Raum: (B, T, d_model) → (B, T, F).
    / Projects transformer output back to feature space: (B, T, d_model) → (B, T, F).
    """

    def __init__(self, ModelDimension: int, NumFeatures: int):
        super().__init__()
        self.OutputLinearLayer = nn.Linear(ModelDimension, NumFeatures)  # d_model → F

    def forward(self, TransformerOutput: torch.Tensor) -> torch.Tensor:
        return self.OutputLinearLayer(TransformerOutput)  # (B, T, d_model) → (B, T, F)


# ============================================================
# Encoder-Architektur / Encoder Architecture
# TranADEncoderLayer: ein einzelner Encoder-Block
# TranADEncoderStack: N gestapelte Encoder-Layer
# ============================================================

class TranADEncoderLayer(nn.Module):
    """
    Ein einzelner Transformer-Encoder-Layer.
    / A single Transformer encoder layer.
    Enthält / Contains:
      1. Multi-Head Self-Attention + Add & Norm
      2. Positions-weises FFN + Add & Norm
    Alle Sub-Layer werden explizit im init definiert.
    / All sub-layers explicitly defined in init.
    """

    def __init__(
        self,
        ModelDimension:   int,          # d_model
        NumHeads:         int,          # Anzahl Attention-Köpfe / Number of attention heads
        ForwardDimension: int,          # Innere FFN-Dimension d_ff / Inner FFN dimension
        DropoutRate:      float = 0.1,  # Dropout-Rate / Dropout rate
    ):
        super().__init__()
        # ── Sub-Layer 1: Self-Attention ───────────────────────────
        self.SelfAttentionModule   = MultiHeadScaledDotProductAttention(
            ModelDimension, NumHeads, DropoutRate
        )                                                  # Multi-Head Self-Attention (Q=K=V=x)
        self.SelfAttentionAddNorm  = ResidualConnectionWithLayerNorm(
            ModelDimension, DropoutRate
        )                                                  # Add & Norm nach Self-Attention / Add & Norm after self-att

        # ── Sub-Layer 2: Feed-Forward Network ────────────────────
        self.FeedForwardModule     = PositionwiseFeedForwardNetwork(
            ModelDimension, ForwardDimension, DropoutRate
        )                                                  # Positions-weises FFN / Position-wise FFN
        self.FeedForwardAddNorm    = ResidualConnectionWithLayerNorm(
            ModelDimension, DropoutRate
        )                                                  # Add & Norm nach FFN / Add & Norm after FFN

    def forward(
        self,
        InputTensor:   torch.Tensor,                    # (B, T, d_model)
        AttentionMask: Optional[torch.Tensor] = None,  # Optionale Padding-Maske / Optional padding mask
    ) -> torch.Tensor:                                  # (B, T, d_model)
        # Sub-Layer 1: Multi-Head Self-Attention
        SelfAttOutput   = self.SelfAttentionModule(
            InputTensor, InputTensor, InputTensor, AttentionMask  # Q = K = V = gleicher Eingang / Q=K=V=same input
        )                                               # Attention berechnen / Compute attention
        AfterSelfAtt    = self.SelfAttentionAddNorm(InputTensor, SelfAttOutput)  # Add & Norm

        # Sub-Layer 2: Feed-Forward Network
        FeedForwardOut  = self.FeedForwardModule(AfterSelfAtt)               # FFN anwenden / Apply FFN
        AfterFFN        = self.FeedForwardAddNorm(AfterSelfAtt, FeedForwardOut)  # Add & Norm
        return AfterFFN                                 # (B, T, d_model) – Layer-Ausgabe / Layer output

# ─────────────────────────────────────────────────────────────────────

class TranADEncoderStack(nn.Module):
    """
    Stapel von N Encoder-Layern + finale LayerNorm.
    / Stack of N encoder layers + final LayerNorm.
    Wird von beiden Phasen gemeinsam genutzt / Shared by both phases.
    """

    def __init__(
        self,
        NumLayers:        int,          # N – Anzahl Encoder-Schichten / Number of encoder layers
        ModelDimension:   int,
        NumHeads:         int,
        ForwardDimension: int,
        DropoutRate:      float = 0.1,
    ):
        super().__init__()
        # N unabhängige Encoder-Layer in ModuleList / N independent layers in ModuleList
        self.EncoderLayerList = nn.ModuleList([
            TranADEncoderLayer(ModelDimension, NumHeads, ForwardDimension, DropoutRate)
            for _ in range(NumLayers)                   # Jeder Layer hat eigene Gewichte / Each layer has own weights
        ])
        self.FinalLayerNorm = nn.LayerNorm(ModelDimension)  # Abschlussnorm / Final normalization

    def forward(
        self,
        InputTensor:   torch.Tensor,                    # (B, T, d_model)
        AttentionMask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                                  # (B, T, d_model) – Encoder-Memory
        CurrentTensor = InputTensor                     # Startpunkt / Starting point
        for EncoderLayer in self.EncoderLayerList:      # Sequenziell durch alle N Layer / Sequential through N layers
            CurrentTensor = EncoderLayer(CurrentTensor, AttentionMask)  # Layer i anwenden / Apply layer i
        return self.FinalLayerNorm(CurrentTensor)       # Finale Normierung / Final normalization



# ============================================================
# Decoder-Architektur / Decoder Architecture
# TranADDecoderLayer: ein einzelner Decoder-Block
# TranADDecoderStack: N gestapelte Decoder-Layer
# TranAD verwendet ZWEI unabhängige Decoder-Stacks.
# / TranAD uses TWO independent decoder stacks.
# ============================================================

class TranADDecoderLayer(nn.Module):
    """
    Ein einzelner Transformer-Decoder-Layer mit drei Sub-Layern.
    / A single Transformer decoder layer with three sub-layers.
    Enthält / Contains:
      1. Masked Multi-Head Self-Attention auf Zielsequenz / On target sequence
      2. Multi-Head Cross-Attention (Q=Decoder, K/V=Encoder-Memory)
      3. Positions-weises FFN
    Alle Sub-Layer explizit im init / All sub-layers explicitly in init.
    """

    def __init__(
        self,
        ModelDimension:   int,
        NumHeads:         int,
        ForwardDimension: int,
        DropoutRate:      float = 0.1,
    ):
        super().__init__()
        # ── Sub-Layer 1: Masked Self-Attention ───────────────────
        self.SelfAttentionModule    = MultiHeadScaledDotProductAttention(
            ModelDimension, NumHeads, DropoutRate
        )                                               # Self-Attention auf Zielsequenz / Self-att on target
        self.SelfAttentionAddNorm   = ResidualConnectionWithLayerNorm(ModelDimension, DropoutRate)

        # ── Sub-Layer 2: Cross-Attention ─────────────────────────
        self.CrossAttentionModule   = MultiHeadScaledDotProductAttention(
            ModelDimension, NumHeads, DropoutRate
        )                                               # Kreuz-Attention: Query=Decoder, Key/Val=Encoder / Cross-att: Q=Dec, K/V=Enc
        self.CrossAttentionAddNorm  = ResidualConnectionWithLayerNorm(ModelDimension, DropoutRate)

        # ── Sub-Layer 3: Feed-Forward Network ────────────────────
        self.FeedForwardModule      = PositionwiseFeedForwardNetwork(
            ModelDimension, ForwardDimension, DropoutRate
        )                                               # Positions-weises FFN / Position-wise FFN
        self.FeedForwardAddNorm     = ResidualConnectionWithLayerNorm(ModelDimension, DropoutRate)

    def forward(
        self,
        TargetTensor:  torch.Tensor,                    # (B, T, d_model) – Decoder-Eingabe / Decoder input
        EncoderMemory: torch.Tensor,                    # (B, T, d_model) – Encoder-Ausgabe / Encoder output
        TargetMask:    Optional[torch.Tensor] = None,  # Maske für Self-Attention / Mask for self-att
        MemoryMask:    Optional[torch.Tensor] = None,  # Maske für Cross-Attention / Mask for cross-att
    ) -> torch.Tensor:                                  # (B, T, d_model)
        # Sub-Layer 1: Selbst-Attention auf Zielsequenz / Self-attention on target sequence
        SelfAttOutput  = self.SelfAttentionModule(
            TargetTensor, TargetTensor, TargetTensor, TargetMask  # Q=K=V=Ziel / Q=K=V=target
        )
        AfterSelfAtt   = self.SelfAttentionAddNorm(TargetTensor, SelfAttOutput)  # Add & Norm

        # Sub-Layer 2: Kreuz-Attention mit Encoder-Memory / Cross-attention with encoder memory
        CrossAttOutput = self.CrossAttentionModule(
            AfterSelfAtt, EncoderMemory, EncoderMemory, MemoryMask  # Q=Decoder-Zustand, K=V=Encoder-Memory
        )
        AfterCrossAtt  = self.CrossAttentionAddNorm(AfterSelfAtt, CrossAttOutput)  # Add & Norm

        # Sub-Layer 3: Feed-Forward Network
        FeedForwardOut = self.FeedForwardModule(AfterCrossAtt)               # FFN / FFN
        AfterFFN       = self.FeedForwardAddNorm(AfterCrossAtt, FeedForwardOut)  # Add & Norm
        return AfterFFN                                 # (B, T, d_model)

# ─────────────────────────────────────────────────────────────────────

class TranADDecoderStack(nn.Module):
    """
    Stapel von N Decoder-Layern + finale LayerNorm.
    / Stack of N decoder layers + final LayerNorm.
    TranAD instanziiert ZWEI dieser Stacks (DecoderStack1, DecoderStack2).
    / TranAD instantiates TWO of these stacks (DecoderStack1, DecoderStack2).
    """

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
            for _ in range(NumLayers)                   # N unabhängige Layer / N independent layers
        ])
        self.FinalLayerNorm   = nn.LayerNorm(ModelDimension)  # Abschlussnorm / Final norm

    def forward(
        self,
        TargetTensor:  torch.Tensor,                    # (B, T, d_model)
        EncoderMemory: torch.Tensor,                    # (B, T, d_model)
        TargetMask:    Optional[torch.Tensor] = None,
        MemoryMask:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                                  # (B, T, d_model)
        CurrentTensor = TargetTensor                    # Startpunkt / Starting point
        for DecoderLayer in self.DecoderLayerList:      # Sequenziell durch alle Layer / Sequential through all layers
            CurrentTensor = DecoderLayer(
                CurrentTensor, EncoderMemory, TargetMask, MemoryMask  # Layer anwenden / Apply layer
            )
        return self.FinalLayerNorm(CurrentTensor)       # Finale Normierung / Final normalization




# ============================================================
# TranADModel: vollständiges Netzwerk 
# Zweiphasige Architektur:
#   Phase 1: Encoder → Decoder1(x_embedded) → x1
#   Phase 2: Encoder → Decoder2(x1_embedded) → x2  (adversariell)
# Loss: L = (1/φ)·||x-x1||² + φ·||x-x2||²
# φ (FocusScore) wächst mit jeder Epoche 
# ============================================================

class TranADModel(nn.Module):
    """
    Kern-Netzwerk des TranAD-Modells.
    / Core network of the TranAD model.

    Alle Layer, Optimizer, Scheduler und Loss explizit im __init__.
    / All layers, optimizer, scheduler and loss explicitly in __init__.

    Architektur / Architecture:
      InputEmbeddingLayer  → SinusoidalPositionalEncoding
      → TranADEncoderStack (gemeinsam für beide Phasen / shared)
      → TranADDecoderStack1  (Phase 1)  → OutputProjectionLayer1 → x1
      → TranADDecoderStack2  (Phase 2)  → OutputProjectionLayer2 → x2
    """

    def __init__(
        self,
        # Hier kann getuned werden / Tunable hyperparameters
        NumFeatures:         int,             # F – Anzahl Eingabe-Features / Number of input features
        ModelDimension:      int   = 64,      # d_model – Transformer-Dimension / Transformer dimension
        NumHeads:            int   = 4,       # H – Attention-Köpfe / Attention heads
        NumEncoderLayers:    int   = 3,       # N_e – Encoder-Tiefe / Encoder depth
        NumDecoderLayers:    int   = 3,       # N_d – Tiefe jedes Decoders / Depth of each decoder
        ForwardDimension:    int   = 256,     # d_ff – FFN-Innendimension / FFN inner dimension
        DropoutRate:         float = 0.1,     # Dropout-Rate / Dropout rate
        MaxSequenceLength:   int   = 5000,    # Maximale PE-Länge / Maximum PE length
        LearningRate:        float = 1e-4,    # Adam-Lernrate / Adam learning rate
        WeightDecay:         float = 1e-5,    # L2-Regularisierung / L2 regularization
        LRSchedulerPatience: int   = 10,      # Geduld für ReduceLROnPlateau / Patience for scheduler
        LRSchedulerFactor:   float = 0.5,     # Reduktionsfaktor LR / LR reduction factor
    ):
        super().__init__()
        self.NumFeatures    = NumFeatures     # Feature-Anzahl merken / Store feature count
        self.ModelDimension = ModelDimension  # d_model merken / Store d_model

        # ── Gemeinsame Einbettungs-Schichten / Shared embedding layers ──
        self.InputEmbeddingLayer     = InputEmbeddingLayer(NumFeatures, ModelDimension)
        # Projiziert F Features → d_model / Projects F features → d_model

        self.PositionalEncodingLayer = SinusoidalPositionalEncoding(
            ModelDimension, MaxSequenceLength, DropoutRate
        )                                             # Sinusoidale PE / Sinusoidal PE

        # ── Gemeinsamer Encoder (geteilt von Phase 1 & 2) / Shared encoder ──
        self.EncoderStack = TranADEncoderStack(
            NumEncoderLayers, ModelDimension, NumHeads, ForwardDimension, DropoutRate
        )                                             # Encoder für kontextuelles Gedächtnis / Encoder for contextual memory

        # ── Normierung des Decoder-Eingangs / Decoder input normalization ──
        self.DecoderInputNorm = nn.LayerNorm(ModelDimension)  # Normiert eingebetteten Decoder-Eingang / Normalizes embedded decoder input

        # ── Decoder 1 – Phase 1 (direkte Rekonstruktion / direct reconstruction) ──
        self.DecoderStack1    = TranADDecoderStack(
            NumDecoderLayers, ModelDimension, NumHeads, ForwardDimension, DropoutRate
        )                                             # Decoder für Phase 1 / Decoder for Phase 1
        self.OutputProjection1 = OutputProjectionLayer(ModelDimension, NumFeatures)
        # Projiziert d_model → F für Rekonstruktion x1 / Projects d_model → F for x1

        # ── Decoder 2 – Phase 2 (adversarielle Verfeinerung / adversarial refinement) ──
        self.DecoderStack2    = TranADDecoderStack(
            NumDecoderLayers, ModelDimension, NumHeads, ForwardDimension, DropoutRate
        )                                             # Decoder für Phase 2 / Decoder for Phase 2
        self.OutputProjection2 = OutputProjectionLayer(ModelDimension, NumFeatures)
        # Projiziert d_model → F für Rekonstruktion x2 / Projects d_model → F for x2
        
        self.Optimizer = torch.optim.Adam(
            self.parameters(),           # Alle Netzgewichte / All network weights
            lr           = LearningRate, # Initiale Lernrate / Initial learning rate
            weight_decay = WeightDecay,  # L2-Regularisierung / L2 regularization
        )                                # Adam für adaptives Lernen / Adam for adaptive learning

        self.LRScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.Optimizer,
            mode     = "min",                   # Verlust minimieren / Minimize loss
            factor   = LRSchedulerFactor,       # LR × factor bei Plateau / LR × factor on plateau
            patience = LRSchedulerPatience,     # Epochen ohne Verbesserung / Epochs without improvement
        )                                       # LR automatisch reduzieren / Auto-reduce LR

        self.ReconstructionLoss = nn.MSELoss(reduction="none")
        # Element-weiser MSE für feature-spezifische Anomalie-Scores / Element-wise MSE for per-feature scores

        # Gewichte initialisieren / Initialize weights
        self._InitializeWeights()
        ParamCount = sum(p.numel() for p in self.parameters())
        print(
            f"TranADModel: {ParamCount:,} Parameter | "
            f"d_model={ModelDimension}, H={NumHeads}, Enc={NumEncoderLayers}, Dec×2={NumDecoderLayers}"
        )

    def _InitializeWeights(self) -> None:
        """
        Xavier-Uniform für alle linearen Schichten, Null-Bias.
        / Xavier uniform for all linear layers, zero bias.
        Sorgt für ausgeglichene Anfangs-Gradienten / Ensures balanced initial gradients.
        """
        for Module in self.modules():
            if isinstance(Module, nn.Linear):          # Nur lineare Schichten / Only linear layers
                nn.init.xavier_uniform_(Module.weight) # Gewichte gleichverteilt initialisieren / Init weights
                if Module.bias is not None:
                    nn.init.zeros_(Module.bias)        # Bias = 0 / Bias = 0

    def _EmbedAndEncode(
        self, InputTensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gemeinsamer Einbettungs- und Encoder-Schritt für beide Phasen.
        / Shared embedding and encoding step for both phases.
        Rückgabe / Returns: (EmbeddedWithPE, EncoderMemory) – beide (B, T, d_model)
        """
        Embedded      = self.InputEmbeddingLayer(InputTensor)      # F → d_model
        WithPE        = self.PositionalEncodingLayer(Embedded)     # + Positions-Encoding / + positional encoding
        EncoderMemory = self.EncoderStack(WithPE)                  # Encoder-Durchlauf / Encoder pass
        return WithPE, EncoderMemory                               # Einbettung und Gedächtnis / Embedding and memory

    def forward(
        self,
        InputTensor: torch.Tensor,                                 # (B, T, F) – Eingangssequenz / Input sequence
    ) -> Tuple[torch.Tensor, torch.Tensor]:                        # (x1, x2) – beide Rekonstruktionen
        """
        Zweiphasiger TranAD-Forward-Pass / Two-phase TranAD forward pass.

        Phase 1: InputTensor → embed → encode → Decoder1 → x1
        Phase 2: x1           → embed → Decoder2(mit Encoder-Memory) → x2
        """
        # ── PHASE 1: direkte Rekonstruktion / direct reconstruction ──
        EmbeddedWithPE, EncoderMemory = self._EmbedAndEncode(InputTensor)
        # EmbeddedWithPE: (B, T, d_model), EncoderMemory: (B, T, d_model)

        NormedDecInput1  = self.DecoderInputNorm(EmbeddedWithPE)  # Decoder-Eingang normieren / Normalize decoder input
        DecoderOutput1   = self.DecoderStack1(NormedDecInput1, EncoderMemory)   # Decoder-1-Pass / Decoder-1 pass
        Reconstruction1  = self.OutputProjection1(DecoderOutput1)               # (B, T, F) → x1

        # ── PHASE 2: adversarielle Verfeinerung / adversarial refinement ──
        # x1 (detached) wird als Decoder-2-Eingabe eingebettet
        # x1 (detached) is embedded as Decoder-2 input
        EmbeddedRec1     = self.InputEmbeddingLayer(Reconstruction1.detach())   # x1 einbetten, kein Gradient durch Phase 1 / embed x1, no grad through Phase 1
        WithPERec1       = self.PositionalEncodingLayer(EmbeddedRec1)            # + PE für x1
        NormedDecInput2  = self.DecoderInputNorm(WithPERec1)                    # Decoder-2-Eingang normieren / Normalize

        DecoderOutput2   = self.DecoderStack2(NormedDecInput2, EncoderMemory)   # Decoder-2-Pass (Encoder-Memory geteilt!) / Decoder-2 pass (shared encoder memory!)
        Reconstruction2  = self.OutputProjection2(DecoderOutput2)               # (B, T, F) → x2

        return Reconstruction1, Reconstruction2  # Beide Rekonstruktionen zurückgeben / Return both reconstructions

    def ComputeTranADLoss(
        self,
        OriginalInput:   torch.Tensor,   # (B, T, F) – Original-Eingang / Original input
        Reconstruction1: torch.Tensor,   # (B, T, F) – Phase-1-Ausgabe x1
        Reconstruction2: torch.Tensor,   # (B, T, F) – Phase-2-Ausgabe x2
        FocusScore:      float,           # φ – steigt mit Epoche / increases with epoch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TranAD-Verlustfunktion / TranAD loss function.
        L = (1/φ) · ||x - x1||² + φ · ||x - x2||²

        Für kleine φ (frühe Epochen): Phase 1 dominiert → stabiles Lernen.
        For small φ (early epochs): Phase 1 dominates → stable learning.
        Für große φ (späte Epochen): Phase 2 dominiert → adversarielle Schärfung.
        For large φ (late epochs): Phase 2 dominates → adversarial sharpening.

        Rückgabe / Returns: (skalarer TotalLoss, ErrorPerTimestep (B, T) für Scoring)
        """
        Error1 = self.ReconstructionLoss(Reconstruction1, OriginalInput)   # (B, T, F) Phase-1-Fehler / Phase-1 error
        Error2 = self.ReconstructionLoss(Reconstruction2, OriginalInput)   # (B, T, F) Phase-2-Fehler / Phase-2 error

        PhiSafe        = max(FocusScore, 1e-6)                              # φ gegen Division durch Null / φ against div-by-zero
        WeightedError1 = Error1 * (1.0 / PhiSafe)                          # (1/φ) · E1 – Phase-1-Gewichtung / Phase-1 weight
        WeightedError2 = Error2 * float(FocusScore)                         # φ · E2 – Phase-2-Gewichtung / Phase-2 weight
        CombinedError  = WeightedError1 + WeightedError2                    # Gesamtfehler / Total error

        TotalLoss        = CombinedError.mean()                             # Skalarer Gesamtverlust / Scalar total loss
        ErrorPerTimestep = Error2.mean(dim=-1)                              # (B, T) – Feature-gemittelter Phase-2-Fehler / Feature-mean Phase-2 error
        return TotalLoss, ErrorPerTimestep





# ============================================================
# TranADMainModel: Orchestrierungsklasse
# Kapselt TranADModel und stellt die vollständige API bereit:
# TrainingStep, TrainModel, DetectAnomalies, Evaluate, Save, Load
# ============================================================

class TranADMainModel:
    """
    Zentrale Orchestrierungsklasse für TranAD.
    / Central orchestration class for TranAD.

    Kein nn.Module – verwaltet das TranADModel intern.
    / Not an nn.Module – manages the TranADModel internally.

    φ (FocusScore) wächst nach jeder Epoche:
    / φ (FocusScore) grows after each epoch:
    φ ← min(φ × FocusGrowthRate, MaxFocusScore)
    """

    def __init__(
        self,
        # Hier können getuned werden
        NumFeatures:         int,             # Anzahl Features / Number of features
        ModelDimension:      int   = 64,      # d_model
        NumHeads:            int   = 4,       # Attention-Köpfe / Attention heads
        NumEncoderLayers:    int   = 3,       # Encoder-Tiefe / Encoder depth
        NumDecoderLayers:    int   = 3,       # Decoder-Tiefe / Decoder depth
        ForwardDimension:    int   = 256,     # FFN-Dimension / FFN dimension
        DropoutRate:         float = 0.1,     # Dropout / Dropout rate
        LearningRate:        float = 1e-4,    # Adam-Lernrate / Adam LR
        WeightDecay:         float = 1e-5,    # L2-Regularisierung / L2 regularization
        LRSchedulerPatience: int   = 10,      # Scheduler-Geduld / Scheduler patience
        FocusGrowthRate:     float = 1.2,     # φ wächst jede Epoche um diesen Faktor / φ grows by this factor each epoch
        InitialFocusScore:   float = 1.0,     # Startwert von φ / Initial value of φ
        MaxFocusScore:       float = 100.0,   # Maximaler φ-Wert / Maximum φ value
        AnomalyPercentile:   float = 95.0,    # Perzentil für Anomalie-Schwellenwert / Percentile for threshold
        Device:              Optional[torch.device] = None,
    ):
        self.Device          = Device if Device is not None else DEVICE  # Rechengerät / Compute device
        self.FocusScore      = InitialFocusScore   # φ = 1.0 zu Beginn / φ = 1.0 initially
        self.FocusGrowthRate = FocusGrowthRate     # Wachstumsfaktor pro Epoche / Growth factor per epoch
        self.MaxFocusScore   = MaxFocusScore        # Obere Schranke für φ / Upper bound for φ
        self.AnomalyPercentile = AnomalyPercentile  # Perzentil für Schwellenwert / Percentile for threshold
        self.AnomalyThreshold  = None               # Wird bei DetectAnomalies befüllt / Filled during DetectAnomalies

        # ── TranADModel instanziieren / Instantiate TranADModel ──
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
        ).to(self.Device)                          # Netzwerk auf Gerät verschieben / Move to device

        # ── Verlaufsprotokoll / Training history log ──────────────
        self.TrainingHistory: Dict[str, List[float]] = {
            "TotalLoss":  [],    # Gesamtverlust pro Epoche / Total loss per epoch
            "FocusScore": [],    # φ-Wert pro Epoche / φ value per epoch
        }

    # ──────────────────────────────────────────────────────────────
    def TrainingStep(self, WindowBatch: torch.Tensor) -> float:
        """
        Einen vollständigen Trainingsschritt ausführen.
        / Execute one complete training step.
        Forward → TranAD-Loss → Backward → Gradient-Clipping → Optimizer-Update.
        Rückgabe / Returns: skalarer Loss-Wert / scalar loss value.
        """
        self.Network.train()                           # Trainingsmodus (Dropout aktiv) / Training mode (dropout active)
        WindowBatch = WindowBatch.to(self.Device)      # Batch auf Gerät verschieben / Move batch to device

        self.Network.Optimizer.zero_grad()             # Gradienten aus letztem Schritt löschen / Clear previous gradients
        Reconstruction1, Reconstruction2 = self.Network(WindowBatch)  # Zweiphasiger Forward-Pass / Two-phase forward
        TotalLoss, _ = self.Network.ComputeTranADLoss(
            WindowBatch, Reconstruction1, Reconstruction2, self.FocusScore
        )                                              # TranAD-Loss mit aktuellem φ / TranAD loss with current φ
        TotalLoss.backward()                           # Gradienten berechnen / Compute gradients
        nn.utils.clip_grad_norm_(self.Network.parameters(), max_norm=1.0)  # Gradient Clipping für Stabilität / Gradient clipping for stability
        self.Network.Optimizer.step()                  # Gewichte aktualisieren / Update weights
        return TotalLoss.item()                        # Skalaren Wert zurückgeben / Return scalar value

    # ──────────────────────────────────────────────────────────────
    def TrainModel(
        self,
        TrainingLoader: DataLoader,   # DataLoader mit Trainingsdaten / DataLoader with training data
        NumEpochs:      int = 50,     # Gesamtanzahl Epochen / Total number of epochs
        PrintInterval:  int = 5,      # Ausgabe-Intervall / Print interval
    ) -> None:
        """
        Vollständiges Training über NumEpochs Epochen.
        / Complete training over NumEpochs epochs.
        Nach jeder Epoche: φ-Update + LR-Scheduler-Schritt.
        / After each epoch: φ-update + LR-scheduler step.
        """
        for Epoch in range(1, NumEpochs + 1):          # Epochen 1 bis NumEpochs / Epochs 1 to NumEpochs
            EpochLossValues: List[float] = []           # Batch-Verluste dieser Epoche / Batch losses this epoch

            for WindowBatch, _ in TrainingLoader:       # Durch alle Batches / Through all batches
                BatchLoss = self.TrainingStep(WindowBatch)  # Einen Schritt / One step
                EpochLossValues.append(BatchLoss)       # Loss sammeln / Collect loss

            AverageLoss = float(np.mean(EpochLossValues))   # Epochen-Durchschnitt / Epoch average
            self.TrainingHistory["TotalLoss"].append(AverageLoss)    # Im Verlauf speichern / Store in history
            self.TrainingHistory["FocusScore"].append(self.FocusScore)  # φ speichern / Store φ

            self.Network.LRScheduler.step(AverageLoss)  # LR-Scheduler informieren / Inform LR scheduler

            # φ erhöhen: Phase-2-Gewichtung wird stärker / Increase φ: Phase-2 weight increases
            self.FocusScore = min(self.FocusScore * self.FocusGrowthRate, self.MaxFocusScore)

            if Epoch % PrintInterval == 0 or Epoch == 1:
                CurrentLR = self.Network.Optimizer.param_groups[0]["lr"]  # Aktuelle LR lesen / Read current LR
                print(
                    f"Epoche {Epoch:>4}/{NumEpochs} | "
                    f"Loss={AverageLoss:.5f} | "
                    f"φ={self.FocusScore:.2f} | "
                    f"LR={CurrentLR:.2e}"
                )

        self.Network.eval()   # Eval-Modus nach Training / Eval mode after training
        print("Training complete.")

    # ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def DetectAnomalies(
        self,
        TestLoader:  DataLoader,
        Percentile:  Optional[float] = None,  # Überschreibt self.AnomalyPercentile / Overrides self.AnomalyPercentile
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechnet Anomalie-Scores und binäre Klassifikation.
        / Computes anomaly scores and binary classification.
        Score = mittlerer Phase-2-MSE über Zeitschritte eines Fensters.
        / Score = mean Phase-2 MSE over timesteps of a window.
        Rückgabe / Returns: (AnomalyScores (N,), BinaryLabels (N,))
        """
        self.Network.eval()                             # Eval-Modus / Eval mode
        AllWindowScores: List[np.ndarray] = []          # Scores aller Fenster / Scores of all windows

        for WindowBatch, _ in TestLoader:               # Durch Test-Batches / Through test batches
            WindowBatch       = WindowBatch.to(self.Device)
            _, Reconstruction2 = self.Network(WindowBatch)   # Nur Phase-2-Rekonstruktion / Only Phase-2

            # Mittlerer MSE über Features pro Zeitschritt, dann über Zeitschritte gemittelt
            # / Mean MSE over features per timestep, then mean over timesteps
            WindowScore = ((Reconstruction2 - WindowBatch) ** 2).mean(dim=-1).mean(dim=-1)  # (B,)
            AllWindowScores.append(WindowScore.cpu().numpy())   # Auf CPU / To CPU

        AllScores    = np.concatenate(AllWindowScores, axis=0)  # (N,) – alle Fenster-Scores / all window scores
        UsedPercentile = Percentile if Percentile is not None else self.AnomalyPercentile
        ThresholdValue = float(np.percentile(AllScores, UsedPercentile))  # Schwellenwert / Threshold

        self.AnomalyThreshold = ThresholdValue          # Für spätere Nutzung speichern / Store for later use
        BinaryLabels = (AllScores > ThresholdValue).astype(int)  # 1 = anomal, 0 = normal
        return AllScores, BinaryLabels                  # Scores und Vorhersagen zurückgeben / Return scores and predictions

    # ──────────────────────────────────────────────────────────────
    def Evaluate(
        self,
        TestLoader:  DataLoader,
        TrueLabels:  np.ndarray,          # (N,) Ground-Truth-Labels / Ground-Truth labels
        Percentile:  Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Berechnet Precision, Recall, F1 ohne sklearn / ohne externe Bibliothek.
        / Computes Precision, Recall, F1 without sklearn / without external library.
        """
        AnomalyScores, PredictedLabels = self.DetectAnomalies(TestLoader, Percentile)

        # ── Konfusionsmatrix / Confusion matrix ──────────────────
        TruePositive  = int(((PredictedLabels == 1) & (TrueLabels == 1)).sum())  # Richtig positiv / True positive
        FalsePositive = int(((PredictedLabels == 1) & (TrueLabels == 0)).sum())  # Falsch positiv / False positive
        FalseNegative = int(((PredictedLabels == 0) & (TrueLabels == 1)).sum())  # Falsch negativ / False negative
        TrueNegative  = int(((PredictedLabels == 0) & (TrueLabels == 0)).sum())  # Richtig negativ / True negative

        # ── Metriken / Metrics ────────────────────────────────────
        Precision = TruePositive / (TruePositive + FalsePositive + 1e-9)  # Präzision / Precision
        Recall    = TruePositive / (TruePositive + FalseNegative + 1e-9)  # Trefferquote / Recall
        F1Score   = 2 * Precision * Recall / (Precision + Recall + 1e-9)  # F1-Maß / F1 measure

        Results = {
            "Precision":     round(Precision, 4),
            "Recall":        round(Recall, 4),
            "F1Score":       round(F1Score, 4),
            "Threshold":     round(float(self.AnomalyThreshold), 6),
            "TruePositive":  TruePositive,
            "FalsePositive": FalsePositive,
            "FalseNegative": FalseNegative,
            "TrueNegative":  TrueNegative,
        }
        print("\n Evaluation Results ──────────────────")
        for MetricName, MetricValue in Results.items():
            print(f"  {MetricName:<20}: {MetricValue}")
        return Results

    # ──────────────────────────────────────────────────────────────
    def Save(self, FilePath: str) -> None:
        """Speichert Gewichte, φ und Verlaufsprotokoll / Saves weights, φ and history."""
        torch.save({
            "NetworkState":      self.Network.state_dict(),  # Alle Netzgewichte / All network weights
            "FocusScore":        self.FocusScore,            # Aktueller φ-Wert / Current φ value
            "TrainingHistory":   self.TrainingHistory,       # Verlauf / History
            "AnomalyThreshold":  self.AnomalyThreshold,     # Gelernter Schwellenwert / Learned threshold
        }, FilePath)
        print(f"Model saved  {FilePath}")

    def Load(self, FilePath: str) -> None:
        """Lädt Gewichte, φ und Verlaufsprotokoll / Loads weights, φ and history."""
        Checkpoint = torch.load(FilePath, map_location=self.Device, weights_only=False)
        self.Network.load_state_dict(Checkpoint["NetworkState"])    # Gewichte restaurieren / Restore weights
        self.FocusScore       = Checkpoint["FocusScore"]            # φ restaurieren / Restore φ
        self.TrainingHistory  = Checkpoint["TrainingHistory"]       # Verlauf laden / Load history
        self.AnomalyThreshold = Checkpoint["AnomalyThreshold"]     # Schwellenwert laden / Load threshold
        self.Network.eval()                                         # Direkt in Eval-Modus / Directly eval mode
        print(f"Model loaded  {FilePath}")



