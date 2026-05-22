import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from typing import Optional, Tuple, List
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np


"""AnoFormer model implementation for time-series anomaly detection."""

# ============================================================
# Low-level Basiskomponenten (from scratch)
# Alle Bausteine werden manuell implementiert – kein nn.Transformer
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention (from scratch).
    Skaliertes Skalarprodukt-Attention mit h parallelen Köpfen.
    Jeder Kopf lernt unterschiedliche Beziehungen zwischen Zeitschritten.
    """

    def __init__(
        self,
        ModelDimension: int,          # Gesamt-Einbettungsdimension des Modells (z.B. 64)
        NumHeads: int,                 # Anzahl paralleler Attention-Köpfe (z.B. 4)
        AttentionDropout: float = 0.1, # Dropout-Rate auf den Attention-Gewichten
    ):
        super().__init__()
        # Sicherheitscheck: Dimension muss gleichmäßig auf Köpfe aufteilbar sein
        assert ModelDimension % NumHeads == 0, \
            "ModelDimension must be divisible by NumHeads"

        self.ModelDimension = ModelDimension           # Gesamtdimension merken
        self.NumHeads       = NumHeads                 # Kopfanzahl merken
        self.HeadDimension  = ModelDimension // NumHeads  # Dimension pro einzelnem Kopf

        # Lineare Projektionen für Query, Key, Value – ohne Bias (üblich bei Attention)
        self.LinearLayerQuery  = nn.Linear(ModelDimension, ModelDimension, bias=False)  # W_Q
        self.LinearLayerKey    = nn.Linear(ModelDimension, ModelDimension, bias=False)  # W_K
        self.LinearLayerValue  = nn.Linear(ModelDimension, ModelDimension, bias=False)  # W_V
        self.LinearLayerOutput = nn.Linear(ModelDimension, ModelDimension)               # W_O – Ausgabeprojektion

        self.AttentionDropoutLayer = nn.Dropout(AttentionDropout)  # Dropout nach Softmax

        # Attention-Gewichte zwischenspeichern → für spätere Visualisierung / Explainability
        self.StoredAttentionWeights: torch.Tensor | None = None

    def forward(
        self,
        Query: torch.Tensor,           # Anfrage-Tensor  (Batch, Zeit, Dim)
        Key: torch.Tensor,             # Schlüssel-Tensor (Batch, Zeit, Dim)
        Value: torch.Tensor,           # Wert-Tensor      (Batch, Zeit, Dim)
        Mask: torch.Tensor | None = None,  # Optionale Maske (z.B. Padding-Maske)
    ) -> torch.Tensor:
        BatchSize = Query.size(0)      # Batch-Größe aus der ersten Dimension lesen

        # Schritt 1: Lineare Projektion auf Q, K, V
        Q = self.LinearLayerQuery(Query)   # (B, T, D) – Query projiziert
        K = self.LinearLayerKey(Key)       # (B, T, D) – Key projiziert
        V = self.LinearLayerValue(Value)   # (B, T, D) – Value projiziert

        # Schritt 2: In Köpfe aufteilen → (Batch, Köpfe, Zeit, KopfDim)
        Q = Q.view(BatchSize, -1, self.NumHeads, self.HeadDimension).transpose(1, 2)
        K = K.view(BatchSize, -1, self.NumHeads, self.HeadDimension).transpose(1, 2)
        V = V.view(BatchSize, -1, self.NumHeads, self.HeadDimension).transpose(1, 2)

        # Schritt 3: Skaliertes Skalarprodukt berechnen
        ScalingFactor   = math.sqrt(self.HeadDimension)                          # √d_k – verhindert zu große Werte
        AttentionScores = torch.matmul(Q, K.transpose(-2, -1)) / ScalingFactor   # (B, H, T, T)

        # Schritt 4: Maske anwenden – maskierte Positionen auf -∞ setzen (→ Softmax ≈ 0)
        if Mask is not None:
            AttentionScores = AttentionScores.masked_fill(Mask == 0, float("-inf"))

        # Schritt 5: Softmax → Gewichte summieren sich pro Zeile zu 1
        AttentionWeights = F.softmax(AttentionScores, dim=-1)          # (B, H, T, T)
        AttentionWeights = self.AttentionDropoutLayer(AttentionWeights) # Dropout für Regularisierung

        # Gewichte für Explainability-Modul zwischenspeichern (kein Gradient nötig)
        self.StoredAttentionWeights = AttentionWeights.detach()

        # Schritt 6: Gewichtete Summe der Values berechnen → Kontextvektor
        ContextVector = torch.matmul(AttentionWeights, V)              # (B, H, T, KopfDim)
        # Köpfe wieder zusammenführen: (B, H, T, KopfDim) → (B, T, D)
        ContextVector = ContextVector.transpose(1, 2).contiguous().view(
            BatchSize, -1, self.ModelDimension
        )
        # Schritt 7: Finale lineare Ausgabeprojektion W_O
        Output = self.LinearLayerOutput(ContextVector)   # (B, T, D)
        return Output                                    # Ausgabe zurückgeben


class ForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network (from scratch).
    Wird nach jeder Attention-Schicht angewendet.
    Zwei lineare Transformationen mit GELU-Aktivierung dazwischen.
    Formel: FFN(x) = W2 · GELU(W1 · x + b1) + b2
    """

    def __init__(
        self,
        ModelDimension: int,          # Eingangs- und Ausgangsdimension (Residual-kompatibel)
        HiddenDimension: int,         # Innere (versteckte) Dimension – typisch 4× ModelDimension
        ForwardDropout: float = 0.1,  # Dropout zwischen den beiden linearen Schichten
    ):
        super().__init__()
        self.LinearLayerOne = nn.Linear(ModelDimension, HiddenDimension)  # Erste Projektion: D → 4D
        self.ActivationGELU = nn.GELU()                                    # GELU ist glatter als ReLU
        self.DropoutLayer   = nn.Dropout(ForwardDropout)                   # Regularisierung nach Aktivierung
        self.LinearLayerTwo = nn.Linear(HiddenDimension, ModelDimension)   # Rückprojektion: 4D → D

    def forward(self, InputTensor: torch.Tensor) -> torch.Tensor:
        Intermediate = self.LinearLayerOne(InputTensor)   # Erste lineare Transformation
        Intermediate = self.ActivationGELU(Intermediate)  # Nicht-Linearität einführen
        Intermediate = self.DropoutLayer(Intermediate)    # Dropout – zufällige Neuronen deaktivieren
        OutputTensor = self.LinearLayerTwo(Intermediate)  # Zweite Transformation zurück auf Modelldim.
        return OutputTensor                               # Ergebnis weitergeben


# ─────────────────────────────────────────────────────────────
class LayerNormalization(nn.Module):
    """
    Layer Normalization (from scratch, ohne nn.LayerNorm).
    Normalisiert jeden Eingabevektor unabhängig über seine letzte Dimension.
    Formel: LN(x) = γ · (x − μ) / √(σ² + ε) + β
    γ (Scale) und β (Shift) sind lernbare Parameter.
    """

    def __init__(self, NormDimension: int, Epsilon: float = 1e-6):
        super().__init__()
        self.Epsilon        = Epsilon                                      # Kleine Zahl gegen Division durch 0
        self.ScaleParameter = nn.Parameter(torch.ones(NormDimension))     # γ – mit 1 initialisiert (neutral)
        self.ShiftParameter = nn.Parameter(torch.zeros(NormDimension))    # β – mit 0 initialisiert (neutral)

    def forward(self, InputTensor: torch.Tensor) -> torch.Tensor:
        Mean       = InputTensor.mean(dim=-1, keepdim=True)                      # Mittelwert μ über letzte Dim.
        Variance   = InputTensor.var(dim=-1, keepdim=True, unbiased=False)       # Varianz σ² (Bias-Variante)
        Normalized = (InputTensor - Mean) / torch.sqrt(Variance + self.Epsilon)  # Standardisieren
        return self.ScaleParameter * Normalized + self.ShiftParameter             # Skalieren + Verschieben


# ─────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    Sinusoidale Positionskodierung (Vaswani et al. 2017).
    Da Transformer keine Reihenfolge kennen, wird die Position als Signal addiert.
    Gerade Indizes → Sinus, ungerade Indizes → Kosinus mit verschiedenen Frequenzen.
    """

    def __init__(self, ModelDimension: int, MaxSequenceLength: int = 5000, Dropout: float = 0.1):
        super().__init__()
        self.DropoutLayer = nn.Dropout(Dropout)  # Dropout nach der Positionskodierung

        # Kodierungsmatrix vorberechnen: (MaxLen, Dim) – wird nicht trainiert
        EncodingMatrix = torch.zeros(MaxSequenceLength, ModelDimension)              # Nullmatrix anlegen
        PositionVector = torch.arange(0, MaxSequenceLength).unsqueeze(1).float()     # Positionen 0…MaxLen-1
        # Frequenzterme: exp(−log(10000) · 2i/d) für jede Dimension i
        DivisionTerms  = torch.exp(
            torch.arange(0, ModelDimension, 2).float() * (-math.log(10000.0) / ModelDimension)
        )
        EncodingMatrix[:, 0::2] = torch.sin(PositionVector * DivisionTerms)  # Sinus für gerade Dim.
        EncodingMatrix[:, 1::2] = torch.cos(PositionVector * DivisionTerms)  # Kosinus für ungerade Dim.
        EncodingMatrix = EncodingMatrix.unsqueeze(0)                          # Batch-Dim hinzufügen: (1, T, D)
        # Als Buffer registrieren → wird mit Modell gespeichert, aber nicht trainiert
        self.register_buffer("EncodingMatrix", EncodingMatrix)

    def forward(self, InputTensor: torch.Tensor) -> torch.Tensor:
        SequenceLength = InputTensor.size(1)                                     # Aktuelle Sequenzlänge T
        OutputTensor   = InputTensor + self.EncodingMatrix[:, :SequenceLength, :] # Positionssignal addieren
        return self.DropoutLayer(OutputTensor)                                   # Dropout und zurückgeben


# ─────────────────────────────────────────────────────────────
class InputEmbeddingLayer(nn.Module):
    """
    Projiziert rohe Eingabe-Features auf die Modelldimension.
    Notwendig, weil die Eingabe (z.B. 3 Features) kleiner als ModelDimension ist.
    """

    def __init__(self, NumFeatures: int, ModelDimension: int):
        super().__init__()
        self.ProjectionLayer = nn.Linear(NumFeatures, ModelDimension)  # Lineare Projektion: F → D
        self.ActivationReLU  = nn.ReLU()                               # ReLU erzeugt Nicht-Linearität

    def forward(self, InputTensor: torch.Tensor) -> torch.Tensor:
        Projected = self.ProjectionLayer(InputTensor)  # Features auf Modelldimension projizieren
        return self.ActivationReLU(Projected)          # Nicht-Linearität anwenden und zurückgeben




# ============================================================
# Two-Step Masking (AnoFormer-spezifisch)
# Kernidee: Anomalien werden durch gezielte Maskierung erkannt
# ============================================================

class TwoStepMasking(nn.Module):
    """
    Zweistufige Maskierungsstrategie aus dem AnoFormer-Paper.

    Schritt 1 – Zufällige Maskierung (Training):
        Zufällige Zeitschritte werden auf 0 gesetzt. Der Encoder lernt,
        diese Lücken aus dem Kontext zu rekonstruieren.

    Schritt 2 – Anomalie-gestützte Maskierung (verfeinert):
        Zeitschritte mit hohem Rekonstruktionsfehler aus Schritt 1 werden
        gezielt maskiert → Modell konzentriert sich auf anomale Bereiche.
    """

    def __init__(
        self,
        RandomMaskingRate: float = 0.15,  # Anteil der zufällig maskierten Tokens (0–1)
        AnomalyThreshold: float  = 0.75,  # Perzentil-Grenze für Anomalie-Maskierung
    ):
        super().__init__()
        self.RandomMaskingRate = RandomMaskingRate  # Maskierungsrate speichern
        self.AnomalyThreshold  = AnomalyThreshold   # Schwellenperzentil speichern

    def StepOneRandomMasking(
        self, InputTensor: torch.Tensor        # Eingabe (Batch, Zeit, Features)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Erzeugt eine zufällige binäre Maske und wendet sie an.
        1 = sichtbar (Token bleibt), 0 = maskiert (Token auf 0 gesetzt).
        Gibt zurück: (MaskierterTensor, BinäreMaske).
        """
        Batch, Time, Dim = InputTensor.shape     # Dimensionen extrahieren
        # Gleichverteilte Zufallszahlen [0, 1) pro Token im Batch
        RandomTensor = torch.rand(Batch, Time, device=InputTensor.device)
        # Alles über der Maskierungsrate bleibt sichtbar (1), der Rest wird maskiert (0)
        MaskTensor   = (RandomTensor > self.RandomMaskingRate).float()
        # Maske auf Feature-Dimension expandieren: (B, T) → (B, T, F)
        MaskExpanded = MaskTensor.unsqueeze(-1).expand_as(InputTensor)
        # Element-weise Multiplikation: maskierte Positionen → 0
        MaskedTensor = InputTensor * MaskExpanded
        return MaskedTensor, MaskTensor          # Maskierter Tensor + verwendete Maske

    def StepTwoAnomalyMasking(
        self,
        InputTensor: torch.Tensor,               # Ursprüngliche Eingabe (B, T, F)
        ReconstructionError: torch.Tensor,        # Fehler pro Zeitschritt (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Maskiert Zeitschritte mit überdurchschnittlich hohem Rekonstruktionsfehler.
        Alles über dem AnomalyThreshold-Perzentil wird als anomal eingestuft und maskiert.
        """
        # Schwellenwert pro Batch-Element berechnen (z.B. 75. Perzentil des Fehlers)
        Threshold    = torch.quantile(
            ReconstructionError, self.AnomalyThreshold, dim=1, keepdim=True
        )
        # Token unter der Schwelle = normal (1), Token drüber = anomal/maskiert (0)
        AnomalyMask  = (ReconstructionError < Threshold).float()
        # Auf Feature-Dim expandieren: (B, T) → (B, T, F)
        MaskExpanded = AnomalyMask.unsqueeze(-1).expand_as(InputTensor)
        # Anomale Zeitschritte auf 0 setzen
        MaskedTensor = InputTensor * MaskExpanded
        return MaskedTensor, AnomalyMask         # Ergebnis und Maske zurückgeben

    def forward(
        self,
        InputTensor: torch.Tensor,
        ReconstructionError: torch.Tensor | None = None,  # None → Schritt 1 aktiv
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wählt automatisch Schritt 1 (kein Fehler) oder Schritt 2 (Fehler vorhanden)."""
        if ReconstructionError is None:
            return self.StepOneRandomMasking(InputTensor)   # Zufällige Maskierung
        else:
            return self.StepTwoAnomalyMasking(InputTensor, ReconstructionError)  # Anomalie-Maskierung





# ============================================================
# Transformer Encoder & Decoder (from scratch)
# Encoder: versteht Kontext; Decoder: rekonstruiert fehlende Teile
# ============================================================

class TransformerEncoderBlock(nn.Module):
    """
    Ein einzelner Transformer-Encoder-Block.
    Datenfluss:
      x → SelbstAttention → Residual + LayerNorm
        → ForwardNetwork  → Residual + LayerNorm → Ausgabe
    """

    def __init__(
        self,
        ModelDimension: int,      # Dimension aller Tensoren in diesem Block
        NumHeads: int,             # Anzahl Attention-Köpfe
        ForwardDimension: int,     # Verborgene Dim. im Feed-Forward-Netz (meist 4× ModelDim)
        DropoutRate: float = 0.1,  # Dropout-Rate für Attention + Residual-Verbindungen
    ):
        super().__init__()
        # Multi-Head Self-Attention: Query = Key = Value = x (Self-Attention)
        self.SelfAttention       = MultiHeadAttention(ModelDimension, NumHeads, DropoutRate)
        # Layer-Normalisierung nach erster Residual-Verbindung
        self.NormalizationOne    = LayerNormalization(ModelDimension)
        # Positionsweises Feed-Forward-Netz
        self.ForwardNetworkLayer = ForwardNetwork(ModelDimension, ForwardDimension, DropoutRate)
        # Layer-Normalisierung nach zweiter Residual-Verbindung
        self.NormalizationTwo    = LayerNormalization(ModelDimension)
        # Dropout für die Residual-Pfade (verhindert Überanpassung)
        self.ResidualDropout     = nn.Dropout(DropoutRate)

    def forward(self, InputTensor: torch.Tensor, Mask: torch.Tensor | None = None) -> torch.Tensor:
        # ── Residual-Verbindung 1: Self-Attention ─────────────
        AttentionOutput = self.SelfAttention(InputTensor, InputTensor, InputTensor, Mask)
        AttentionOutput = self.ResidualDropout(AttentionOutput)          # Dropout auf Attention-Ausgabe
        AfterNormOne    = self.NormalizationOne(InputTensor + AttentionOutput)  # Add & Norm

        # ── Residual-Verbindung 2: Feed-Forward-Netz ──────────
        FFNOutput    = self.ForwardNetworkLayer(AfterNormOne)            # FFN auf normiertem Tensor
        FFNOutput    = self.ResidualDropout(FFNOutput)                   # Dropout auf FFN-Ausgabe
        OutputTensor = self.NormalizationTwo(AfterNormOne + FFNOutput)   # Add & Norm
        return OutputTensor                                              # Fertige Block-Ausgabe


# ─────────────────────────────────────────────────────────────
class TransformerDecoderBlock(nn.Module):
    """
    Ein einzelner Transformer-Decoder-Block.
    Datenfluss:
      x → Maskierte SelbstAttention → Add & Norm
        → KreuzAttention (mit Encoder-Ausgabe) → Add & Norm
        → ForwardNetwork → Add & Norm → Ausgabe
    """

    def __init__(
        self,
        ModelDimension: int,
        NumHeads: int,
        ForwardDimension: int,
        DropoutRate: float = 0.1,
    ):
        super().__init__()
        # Maskierte Self-Attention: Decoder darf nur vergangene Positionen sehen
        self.MaskedSelfAttention = MultiHeadAttention(ModelDimension, NumHeads, DropoutRate)
        self.NormalizationOne    = LayerNormalization(ModelDimension)   # Norm nach Self-Att.

        # Kreuz-Attention: Query kommt vom Decoder, Key+Value vom Encoder
        self.CrossAttention      = MultiHeadAttention(ModelDimension, NumHeads, DropoutRate)
        self.NormalizationTwo    = LayerNormalization(ModelDimension)   # Norm nach Kreuz-Att.

        # Feed-Forward-Netz (identisch zum Encoder-Block)
        self.ForwardNetworkLayer = ForwardNetwork(ModelDimension, ForwardDimension, DropoutRate)
        self.NormalizationThree  = LayerNormalization(ModelDimension)   # Norm nach FFN

        self.ResidualDropout     = nn.Dropout(DropoutRate)              # Gemeinsamer Residual-Dropout

    def forward(
        self,
        DecoderInput: torch.Tensor,               # Decoder-Eingabe (z.B. maskiertes Signal)
        EncoderOutput: torch.Tensor,              # Encoder-Kontext (vollständige Repräsentation)
        DecoderMask: torch.Tensor | None = None,  # Kausale Maske für den Decoder
        EncoderMask: torch.Tensor | None = None,  # Padding-Maske für den Encoder
    ) -> torch.Tensor:
        # ── Schritt 1: Maskierte Self-Attention ───────────────
        SelfAttOutput = self.MaskedSelfAttention(
            DecoderInput, DecoderInput, DecoderInput, DecoderMask  # Q = K = V = DecoderInput
        )
        SelfAttOutput = self.ResidualDropout(SelfAttOutput)         # Dropout
        AfterNormOne  = self.NormalizationOne(DecoderInput + SelfAttOutput)  # Add & Norm

        # ── Schritt 2: Kreuz-Attention (Encoder ↔ Decoder) ────
        CrossAttOutput = self.CrossAttention(
            AfterNormOne,   # Query kommt vom Decoder
            EncoderOutput,  # Key kommt vom Encoder
            EncoderOutput,  # Value kommt vom Encoder
            EncoderMask
        )
        CrossAttOutput = self.ResidualDropout(CrossAttOutput)        # Dropout
        AfterNormTwo   = self.NormalizationTwo(AfterNormOne + CrossAttOutput)  # Add & Norm

        # ── Schritt 3: Feed-Forward-Netz ──────────────────────
        FFNOutput    = self.ForwardNetworkLayer(AfterNormTwo)        # Positionsweises FFN
        FFNOutput    = self.ResidualDropout(FFNOutput)               # Dropout
        OutputTensor = self.NormalizationThree(AfterNormTwo + FFNOutput)  # Add & Norm
        return OutputTensor                                          # Block-Ausgabe


# ─────────────────────────────────────────────────────────────
class TransformerEncoderStack(nn.Module):
    """
    Stapel aus N Encoder-Blöcken, gefolgt von einer abschließenden LayerNorm.
    Die Ausgabe dient als Kontext für den Decoder.
    """

    def __init__(self, NumLayers: int, ModelDimension: int, NumHeads: int,
                 ForwardDimension: int, DropoutRate: float = 0.1):
        super().__init__()
        # N identisch aufgebaute Encoder-Blöcke in einer Liste
        self.EncoderBlockList = nn.ModuleList([
            TransformerEncoderBlock(ModelDimension, NumHeads, ForwardDimension, DropoutRate)
            for _ in range(NumLayers)   # N-fache Wiederholung
        ])
        self.FinalNormalization = LayerNormalization(ModelDimension)  # Abschließende Normierung

    def forward(self, InputTensor: torch.Tensor, Mask: torch.Tensor | None = None) -> torch.Tensor:
        CurrentTensor = InputTensor                          # Startwert = Einbettung
        for EncoderBlock in self.EncoderBlockList:           # Sequenziell durch alle Blöcke
            CurrentTensor = EncoderBlock(CurrentTensor, Mask)  # Block anwenden
        return self.FinalNormalization(CurrentTensor)        # Finale Normierung und Rückgabe


# ─────────────────────────────────────────────────────────────
class TransformerDecoderStack(nn.Module):
    """
    Stapel aus N Decoder-Blöcken, gefolgt von einer abschließenden LayerNorm.
    Nimmt den Encoder-Kontext und rekonstruiert die Ausgabe.
    """

    def __init__(self, NumLayers: int, ModelDimension: int, NumHeads: int,
                 ForwardDimension: int, DropoutRate: float = 0.1):
        super().__init__()
        # N identisch aufgebaute Decoder-Blöcke
        self.DecoderBlockList = nn.ModuleList([
            TransformerDecoderBlock(ModelDimension, NumHeads, ForwardDimension, DropoutRate)
            for _ in range(NumLayers)
        ])
        self.FinalNormalization = LayerNormalization(ModelDimension)  # Abschließende Normierung

    def forward(
        self,
        DecoderInput: torch.Tensor,                # Decoder-Eingang (maskiertes Signal)
        EncoderOutput: torch.Tensor,               # Encoder-Kontext
        DecoderMask: torch.Tensor | None = None,   # Kausale Decoder-Maske
        EncoderMask: torch.Tensor | None = None,   # Encoder-Padding-Maske
    ) -> torch.Tensor:
        CurrentTensor = DecoderInput                 # Startwert
        for DecoderBlock in self.DecoderBlockList:   # Jeden Decoder-Block anwenden
            CurrentTensor = DecoderBlock(CurrentTensor, EncoderOutput, DecoderMask, EncoderMask)
        return self.FinalNormalization(CurrentTensor) # Normierung und Rückgabe






# ============================================================
# Generator & Diskriminator (GAN-Komponenten)
# Generator: rekonstruiert maskierte Zeitreihen
# Diskriminator: unterscheidet echte von rekonstruierten Zeitreihen
# ============================================================

class TransformerGenerator(nn.Module):
    """
    Generator des AnoFormer-GANs.
    Aufgabe: Maskierte Eingabe → vollständige Rekonstruktion der Zeitreihe.
    Architektur: Einbettung → Positionskodierung → Encoder → Decoder → Ausgabe.
    """

    def __init__(
        self,
        NumFeatures: int,              # Anzahl der Eingangs-Features (z.B. 3 Sensorkanäle)
        ModelDimension: int = 64,      # Interne Transformer-Dimension
        NumHeads: int = 4,             # Anzahl Attention-Köpfe
        NumEncoderLayers: int = 3,     # Tiefe des Encoders (Anzahl Blöcke)
        NumDecoderLayers: int = 3,     # Tiefe des Decoders (Anzahl Blöcke)
        ForwardDimension: int = 256,   # Verborgene Dim. im FFN (4× ModelDim empfohlen)
        DropoutRate: float = 0.1,      # Dropout-Rate überall im Generator
        LearningRate: float = 1e-4,    # Lernrate für den Adam-Optimierer
        BetaOne: float = 0.5,          # Adam β₁ – für GAN-Training niedrig wählen (0.5)
        BetaTwo: float = 0.999,        # Adam β₂ – Standardwert
        WeightDecay: float = 1e-5,     # L2-Regularisierung gegen Überanpassung
    ):
        super().__init__()
        # ── Architektur-Schichten ────────────────────────────────
        self.InputEmbedding     = InputEmbeddingLayer(NumFeatures, ModelDimension)  # Features → Dim
        self.PositionalEncoding = PositionalEncoding(ModelDimension, Dropout=DropoutRate)  # Pos. addieren
        self.EncoderStack       = TransformerEncoderStack(                   # Encoder-Stapel
            NumEncoderLayers, ModelDimension, NumHeads, ForwardDimension, DropoutRate
        )
        self.DecoderStack       = TransformerDecoderStack(                   # Decoder-Stapel
            NumDecoderLayers, ModelDimension, NumHeads, ForwardDimension, DropoutRate
        )
        # Ausgabeprojektion: Modelldim. → ursprüngliche Feature-Anzahl
        self.OutputProjection   = nn.Linear(ModelDimension, NumFeatures)

        # ── Loss-Funktionen ──────────────────────────────────────
        self.ReconstructionLoss = nn.MSELoss(reduction="none")   # MSE elementweise → für Fehler pro Token
        self.GeneratorLoss      = nn.BCEWithLogitsLoss()          # GAN-Loss: Generator will Dискriminator täuschen

        # ── Optimierer & Lernraten-Scheduler ────────────────────
        self.Optimizer = optim.Adam(
            self.parameters(),           # Alle trainierbaren Parameter des Generators
            lr=LearningRate,             # Anfangs-Lernrate
            betas=(BetaOne, BetaTwo),    # Momentumparameter für Adam
            weight_decay=WeightDecay,    # L2-Penalty auf Gewichte
        )
        # Cosine Annealing: Lernrate sinkt kosinusförmig von lr auf eta_min
        self.LearningRateScheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.Optimizer, T_max=100, eta_min=1e-6  # T_max = Anzahl Epochen bis Minimum
        )

    def forward(self, MaskedInput: torch.Tensor) -> torch.Tensor:
        Embedded       = self.InputEmbedding(MaskedInput)       # Features → interne Darstellung
        WithPosition   = self.PositionalEncoding(Embedded)       # Positionale Information hinzufügen
        EncoderContext = self.EncoderStack(WithPosition)         # Kontext aus vollständiger Sequenz
        DecoderOutput  = self.DecoderStack(WithPosition, EncoderContext)  # Rekonstruktion mit Kontext
        Reconstruction = self.OutputProjection(DecoderOutput)    # Zurück auf Feature-Raum projizieren
        return Reconstruction                                    # Rekonstruierte Zeitreihe

    def ComputeErrorPerTimestep(
        self, Reconstruction: torch.Tensor, Original: torch.Tensor
    ) -> torch.Tensor:
        """
        Berechnet den mittleren quadratischen Fehler pro Zeitschritt.
        Gemittelt über alle Features → ein Fehlerwert je Token.
        Eingabe:  (Batch, Zeit, Features)
        Ausgabe:  (Batch, Zeit)
        """
        ErrorTensor = self.ReconstructionLoss(Reconstruction, Original)  # (B, T, F) – elementweise MSE
        return ErrorTensor.mean(dim=-1)                                   # (B, T) – über Features mitteln


# ─────────────────────────────────────────────────────────────
class TransformerDiscriminator(nn.Module):
    """
    Diskriminator des AnoFormer-GANs.
    Aufgabe: Binäre Klassifikation – ist die Zeitreihe echt (1) oder generiert (0)?
    Architektur: Einbettung → Positionskodierung → Encoder → GlobalPool → Klassifikator.
    """

    def __init__(
        self,
        NumFeatures: int,              # Anzahl Eingabe-Features (muss mit Generator übereinstimmen)
        ModelDimension: int = 64,      # Interne Dimension
        NumHeads: int = 4,             # Attention-Köpfe
        NumLayers: int = 2,            # Weniger Schichten als Generator (Diskriminator ist leichter)
        ForwardDimension: int = 128,   # FFN-Dimension (halb so groß wie im Generator)
        DropoutRate: float = 0.1,      # Dropout-Rate
        LearningRate: float = 2e-4,    # Etwas höhere LR als Generator – üblich bei GANs
        BetaOne: float = 0.5,          # Adam β₁
        BetaTwo: float = 0.999,        # Adam β₂
        WeightDecay: float = 1e-5,     # L2-Regularisierung
    ):
        super().__init__()
        # ── Architektur-Schichten ────────────────────────────────
        self.InputEmbedding      = InputEmbeddingLayer(NumFeatures, ModelDimension)   # Einbettung
        self.PositionalEncoding  = PositionalEncoding(ModelDimension, Dropout=DropoutRate)  # Pos.-Kodierung
        self.EncoderStack        = TransformerEncoderStack(                            # Encoder (kein Decoder!)
            NumLayers, ModelDimension, NumHeads, ForwardDimension, DropoutRate
        )
        # Klassifikationskopf: Dropout → Linear(Dim → 1)
        self.ClassificationDropout = nn.Dropout(0.2)                   # Stärkerer Dropout vor Klassifikation
        self.ClassificationLayer   = nn.Linear(ModelDimension, 1)      # Einzelner Ausgabe-Logit

        # ── Loss-Funktion ────────────────────────────────────────
        self.DiscriminatorLoss = nn.BCEWithLogitsLoss()   # Binäre Kreuzentropie (enthält Sigmoid intern)

        # ── Optimierer & Scheduler ───────────────────────────────
        self.Optimizer = optim.Adam(
            self.parameters(), lr=LearningRate, betas=(BetaOne, BetaTwo), weight_decay=WeightDecay
        )
        self.LearningRateScheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.Optimizer, T_max=100, eta_min=1e-6
        )

    def forward(self, InputTensor: torch.Tensor) -> torch.Tensor:
        Embedded      = self.InputEmbedding(InputTensor)    # Feature-Einbettung
        WithPosition  = self.PositionalEncoding(Embedded)   # Positionskodierung hinzufügen
        EncoderOutput = self.EncoderStack(WithPosition)     # Transformer-Encoding
        # Global Average Pooling über Zeitdimension: (B, T, D) → (B, D)
        GlobalMean    = EncoderOutput.mean(dim=1)
        GlobalMean    = self.ClassificationDropout(GlobalMean)   # Dropout vor Klassifikation
        Logit         = self.ClassificationLayer(GlobalMean)     # Roher Score (kein Sigmoid hier!)
        return Logit                                             # (B, 1) – echt=1, gefälscht=0







# ============================================================
# AnoFormerMainModel (zentrale Orchestrierungsklasse)
# Verbindet Generator, Diskriminator und Two-Step Masking
# ============================================================

class AnoFormerMainModel:
    """
    Hauptklasse des AnoFormer-Systems.
    Orchestriert: Generator, Diskriminator, Two-Step Masking,
    Training, Anomalieerkennung, Evaluation sowie Speichern/Laden.
    Kein nn.Module – verwaltet zwei separate nn.Module intern.
    """

    def __init__(
        self,
        NumFeatures: int,                          # Anzahl Eingabe-Features
        ModelDimension: int = 64,                  # Transformer-Dimension
        NumHeads: int = 4,                         # Attention-Köpfe
        NumEncoderLayers: int = 3,                 # Encoder-Tiefe im Generator
        NumDecoderLayers: int = 3,                 # Decoder-Tiefe im Generator
        DiscriminatorLayers: int = 2,              # Encoder-Tiefe im Diskriminator
        ForwardDimension: int = 256,               # FFN-Dimension
        DropoutRate: float = 0.1,                  # Globale Dropout-Rate
        GeneratorLearningRate: float = 1e-4,       # Lernrate Generator
        DiscriminatorLearningRate: float = 2e-4,   # Lernrate Diskriminator (etwas höher)
        MaskingRate: float = 0.15,                 # Anteil zufällig maskierter Tokens
        AnomalyThreshold: float = 0.75,            # Perzentil für Anomalie-Maskierung in Schritt 2
        ReconstructionWeight: float = 10.0,        # Gewichtungsfaktor für Rekonstruktions-Loss
        AnomalyLimit: float | None = None,         # Fester Entscheidungsschwellenwert (optional)
        Device: torch.device | None = None,        # Rechengerät (None → DEVICE aus Zelle 1)
    ):
        self.Device = Device if Device is not None else DEVICE  # Gerät auswählen

        # ── Generator instanziieren und auf Gerät verschieben ────
        self.Generator = TransformerGenerator(
            NumFeatures, ModelDimension, NumHeads,
            NumEncoderLayers, NumDecoderLayers,
            ForwardDimension, DropoutRate,
            GeneratorLearningRate,
        ).to(self.Device)                          # Alle Parameter auf GPU/CPU verschieben

        # ── Diskriminator instanziieren ──────────────────────────
        self.Discriminator = TransformerDiscriminator(
            NumFeatures, ModelDimension, NumHeads,
            DiscriminatorLayers,
            ForwardDimension // 2,                 # Diskriminator ist bewusst kleiner als Generator
            DropoutRate, DiscriminatorLearningRate,
        ).to(self.Device)

        # ── Masking-Modul ────────────────────────────────────────
        self.Masking = TwoStepMasking(MaskingRate, AnomalyThreshold)   # Kein .to(device) nötig

        # ── Hyperparameter als Instanzattribute speichern ────────
        self.ReconstructionWeight = ReconstructionWeight  # λ in: G-Loss = GAN-Loss + λ·Rek.-Loss
        self.AnomalyLimit         = AnomalyLimit          # Schwellenwert für Anomalieklassifikation
        self.NumFeatures          = NumFeatures            # Feature-Anzahl merken

        # ── Trainingsverlauf-Protokoll initialisieren ────────────
        self.TrainingHistory: dict[str, list[float]] = {
            "GeneratorTotalLoss":  [],   # Gesamtverlust des Generators pro Epoche
            "GeneratorGANLoss":    [],   # Täuschungs-Anteil (GAN-Loss)
            "GeneratorRecLoss":    [],   # Rekonstruktions-Anteil (MSE)
            "DiscriminatorLoss":   [],   # Verlust des Diskriminators pro Epoche
        }

    # ──────────────────────────────────────────────────────────
    def TrainingStep(
        self,
        RealData: torch.Tensor,                              # Echte Zeitreihe (B, T, F)
        ReconstructionError: torch.Tensor | None = None,    # Fehler für Schritt-2-Masking
    ) -> dict[str, float]:
        """
        Führt einen vollständigen GAN-Trainingsschritt durch.
        Reihenfolge: 1) Diskriminator optimieren, 2) Generator optimieren.
        Gibt alle Einzelverluste als Dictionary zurück.
        """
        RealData = RealData.to(self.Device)              # Batch auf Rechengerät verschieben

        # ── Schritt 0: Maskierung anwenden ────────────────────────
        MaskedData, _ = self.Masking(RealData, ReconstructionError)  # Maskierte Version erzeugen

        # ════════════════════════════════════════════════════════
        # DISKRIMINATOR-SCHRITT
        # Ziel: echte Daten als echt (1) und gefälschte als falsch (0) klassifizieren
        # ════════════════════════════════════════════════════════
        self.Discriminator.Optimizer.zero_grad()          # Alte Gradienten des Diskriminators löschen

        RealLogits = self.Discriminator(RealData)         # Diskriminator auf echten Daten auswerten
        RealLabels = torch.ones_like(RealLogits)          # Ziel-Label: 1 (echt)

        # Generator-Forward ohne Gradienten (nur für Diskriminator-Training nötig)
        with torch.no_grad():
            FakeReconstruction = self.Generator(MaskedData)  # Gefälschte Rekonstruktion erzeugen

        FakeLogits = self.Discriminator(FakeReconstruction.detach())  # .detach() → kein Generator-Gradient
        FakeLabels = torch.zeros_like(FakeLogits)                      # Ziel-Label: 0 (gefälscht)

        # Diskriminator-Loss = Mittelwert aus (echt-Loss + falsch-Loss)
        DiscriminatorLoss = (
            self.Discriminator.DiscriminatorLoss(RealLogits, RealLabels) +   # Fehler auf echten Daten
            self.Discriminator.DiscriminatorLoss(FakeLogits, FakeLabels)     # Fehler auf gefälschten
        ) * 0.5                                           # Faktor 0.5: Mittelwert der beiden Terme

        DiscriminatorLoss.backward()                      # Gradienten für Diskriminator berechnen
        nn.utils.clip_grad_norm_(self.Discriminator.parameters(), max_norm=1.0)  # Gradient Clipping
        self.Discriminator.Optimizer.step()               # Diskriminator-Gewichte aktualisieren

        # ════════════════════════════════════════════════════════
        # GENERATOR-SCHRITT
        # Ziel: Diskriminator täuschen (GAN-Loss) + gute Rekonstruktion (Rec-Loss)
        # ════════════════════════════════════════════════════════
        self.Generator.Optimizer.zero_grad()              # Alte Gradienten des Generators löschen

        FakeReconstruction = self.Generator(MaskedData)   # Erneut generieren (jetzt MIT Gradient)
        GAN_Logits         = self.Discriminator(FakeReconstruction)  # Diskriminator auf Fake auswerten

        # Generator-GAN-Loss: gefälschte Daten sollen als echt (1) klassifiziert werden
        GeneratorGANLoss   = self.Generator.GeneratorLoss(GAN_Logits, torch.ones_like(GAN_Logits))
        # Rekonstruktions-Loss: gemittelter MSE zwischen Rekonstruktion und Original
        RecLoss            = self.Generator.ReconstructionLoss(FakeReconstruction, RealData).mean()
        # Kombinierter Loss: GAN-Loss + λ × Rekonstruktions-Loss
        GeneratorTotalLoss = GeneratorGANLoss + self.ReconstructionWeight * RecLoss

        GeneratorTotalLoss.backward()                     # Gradienten für Generator berechnen
        nn.utils.clip_grad_norm_(self.Generator.parameters(), max_norm=1.0)  # Gradient Clipping
        self.Generator.Optimizer.step()                   # Generator-Gewichte aktualisieren

        # ── Alle Verluste als Dictionary zurückgeben ─────────────
        return {
            "GeneratorTotalLoss": GeneratorTotalLoss.item(),   # Gesamtverlust Generator
            "GeneratorGANLoss":   GeneratorGANLoss.item(),     # GAN-Anteil
            "GeneratorRecLoss":   RecLoss.item(),              # Rekonstruktions-Anteil
            "DiscriminatorLoss":  DiscriminatorLoss.item(),    # Diskriminator-Verlust
        }

    # ──────────────────────────────────────────────────────────
    def TrainModel(
        self,
        TrainingLoader: DataLoader,    # DataLoader mit Trainings-Batches
        NumEpochs: int = 50,           # Gesamtanzahl Trainingsepochen
        PrintInterval: int = 5,        # Alle N Epochen Fortschritt ausgeben
    ) -> None:
        """
        Trainiert Generator und Diskriminator über mehrere Epochen.
        Alternierende Two-Step Masking-Strategie:
          - Ungerade Epochen → Schritt 1 (zufällige Maskierung)
          - Gerade Epochen  → Schritt 2 (anomaliegestützte Maskierung)
        """
        self.Generator.train()        # Generator in Trainingsmodus schalten (Dropout aktiv)
        self.Discriminator.train()    # Diskriminator ebenfalls in Trainingsmodus

        for Epoch in range(1, NumEpochs + 1):              # Epochen-Schleife (1-basiert)
            # Verlustlisten für diese Epoche initialisieren
            EpochLosses: dict[str, list[float]] = {k: [] for k in self.TrainingHistory}

            for Batch, (WindowData, _) in enumerate(TrainingLoader):   # Batch-Schleife
                WindowData = WindowData.to(self.Device)                 # Batch auf Gerät

                # Gerade Epoche → Schritt-2-Maskierung benötigt Fehler aus Schritt 1
                if Epoch % 2 == 0:
                    with torch.no_grad():                               # Kein Gradient nötig
                        MaskedTemp, _ = self.Masking(WindowData, None)  # Schritt 1 ausführen
                        RecTemp       = self.Generator(MaskedTemp)       # Rekonstruktion berechnen
                        ErrorPerTime  = self.Generator.ComputeErrorPerTimestep(RecTemp, WindowData)
                else:
                    ErrorPerTime = None                                  # Ungerade Epoche → Schritt 1

                # Einen vollständigen GAN-Trainingsschritt ausführen
                LossValues = self.TrainingStep(WindowData, ErrorPerTime)

                for Key, Val in LossValues.items():         # Verluste sammeln
                    EpochLosses[Key].append(Val)

            # ── Epochen-Durchschnitte berechnen und speichern ───
            for Key in self.TrainingHistory:
                Average = float(np.mean(EpochLosses[Key]))  # Mittelwert über alle Batches
                self.TrainingHistory[Key].append(Average)

            # ── Lernraten-Scheduler aktualisieren ───────────────
            self.Generator.LearningRateScheduler.step()      # Generator-LR anpassen
            self.Discriminator.LearningRateScheduler.step()  # Diskriminator-LR anpassen

            # ── Fortschritt ausgeben ─────────────────────────────
            if Epoch % PrintInterval == 0 or Epoch == 1:
                print(
                    f"Epoche {Epoch:>4}/{NumEpochs} | "
                    f"G-Total={self.TrainingHistory['GeneratorTotalLoss'][-1]:.4f} | "
                    f"G-GAN={self.TrainingHistory['GeneratorGANLoss'][-1]:.4f} | "
                    f"G-Rec={self.TrainingHistory['GeneratorRecLoss'][-1]:.4f} | "
                    f"D={self.TrainingHistory['DiscriminatorLoss'][-1]:.4f}"
                )

        self.Generator.eval()         # Generator nach Training in Auswertungsmodus
        self.Discriminator.eval()     # Diskriminator ebenfalls
        print("Training abgeschlossen.")

    # ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def DetectAnomalies(
        self,
        TestLoader: DataLoader,        # DataLoader mit Testdaten
        Threshold: float | None = None,  # Optionaler fester Schwellenwert
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Berechnet Anomalie-Scores und klassifiziert Fenster als anomal/normal.
        Score = mittlerer MSE des Generators über alle Zeitschritte eines Fensters.
        Rückgabe: (AnomalieScores, BinäreKlassifikation) je (N,)
        """
        self.Generator.eval()          # Auswertungsmodus (kein Dropout)
        AllErrors: list[np.ndarray] = []  # Fehlerliste für alle Batches

        for WindowData, _ in TestLoader:              # Durch alle Test-Batches iterieren
            WindowData     = WindowData.to(self.Device)
            MaskedTemp, _  = self.Masking(WindowData, None)   # Schritt-1-Masking anwenden
            Reconstruction = self.Generator(MaskedTemp)       # Rekonstruktion berechnen
            ErrorPerTime   = self.Generator.ComputeErrorPerTimestep(Reconstruction, WindowData)
            AllErrors.append(ErrorPerTime.cpu().numpy())      # CPU-Array sammeln

        # Alle Fehler zusammenführen → (N, T) und über Zeit mitteln → (N,)
        AllScores = np.concatenate(AllErrors, axis=0).mean(axis=1)

        # Schwellenwert bestimmen (Priorität: Argument > gespeicherter Wert > 95. Perzentil)
        UsedThreshold = (
            Threshold if Threshold is not None                    # Explizit übergeben?
            else (self.AnomalyLimit if self.AnomalyLimit is not None  # Schon gespeichert?
                  else float(np.percentile(AllScores, 95)))       # Sonst: 95. Perzentil berechnen
        )
        self.AnomalyLimit = UsedThreshold                         # Schwellenwert für spätere Nutzung merken

        AnomalyBinary = (AllScores > UsedThreshold).astype(int)  # 1 = anomal, 0 = normal
        return AllScores, AnomalyBinary                           # Scores und Labels zurückgeben

    # ──────────────────────────────────────────────────────────
    def Evaluate(
        self,
        TestLoader: DataLoader,
        TrueLabels: np.ndarray,         # Ground-Truth Anomalie-Labels (0 oder 1)
    ) -> dict[str, float]:
        """
        Evaluiert den AnoFormer anhand von Precision, Recall und F1-Score.
        Berechnet alles ohne sklearn – nur NumPy/Python.
        """
        Scores, PredictedLabels = self.DetectAnomalies(TestLoader)  # Scores + Vorhersagen holen

        # ── Konfusionsmatrix-Elemente berechnen ─────────────────
        TruePositive  = int(((PredictedLabels == 1) & (TrueLabels == 1)).sum())  # Richtig als anomal erkannt
        FalsePositive = int(((PredictedLabels == 1) & (TrueLabels == 0)).sum())  # Normal fälschlisch als anomal
        FalseNegative = int(((PredictedLabels == 0) & (TrueLabels == 1)).sum())  # Anomalie übersehen

        # ── Metriken berechnen ───────────────────────────────────
        Precision = TruePositive / (TruePositive + FalsePositive + 1e-9)  # Genauigkeit der Alarme
        Recall    = TruePositive / (TruePositive + FalseNegative + 1e-9)  # Trefferquote der Anomalien
        F1Score   = 2 * Precision * Recall / (Precision + Recall + 1e-9)  # Harmonisches Mittel

        Results = {
            "Precision":     round(Precision, 4),     # Wie viele Alarme sind korrekt?
            "Recall":        round(Recall, 4),         # Wie viele Anomalien wurden gefunden?
            "F1Score":       round(F1Score, 4),        # Ausgewogenes Gesamtmaß
            "Threshold":     round(self.AnomalyLimit, 6),  # Verwendeter Schwellenwert
            "TruePositive":  TruePositive,             # Richtig-Positiv-Anzahl
            "FalsePositive": FalsePositive,            # Falsch-Positiv-Anzahl (Fehlalarme)
            "FalseNegative": FalseNegative,            # Falsch-Negativ-Anzahl (übersehene Anomalien)
        }
        for Name, Value in Results.items():
            print(f"  {Name:<20}: {Value}")            # Alle Metriken ausgeben
        return Results

    # ──────────────────────────────────────────────────────────
    def Save(self, Path: str) -> None:
        """Speichert Modellgewichte und Trainingsverlauf in einer .pt-Datei."""
        torch.save({
            "GeneratorState":     self.Generator.state_dict(),      # Generator-Gewichte
            "DiscriminatorState": self.Discriminator.state_dict(),  # Diskriminator-Gewichte
            "TrainingHistory":    self.TrainingHistory,             # Verlaufsprotokoll
            "AnomalyLimit":       self.AnomalyLimit,                # Gespeicherter Schwellenwert
        }, Path)
        print(f"Modell gespeichert unter: {Path}")

    def Load(self, Path: str) -> None:
        """Lädt Modellgewichte und Trainingsverlauf aus einer .pt-Datei."""
        State = torch.load(Path, map_location=self.Device, weights_only=False)  # Datei laden
        self.Generator.load_state_dict(State["GeneratorState"])                  # Gewichte in Generator laden
        self.Discriminator.load_state_dict(State["DiscriminatorState"])          # Gewichte in Diskriminator
        self.TrainingHistory = State["TrainingHistory"]                           # Verlauf wiederherstellen
        self.AnomalyLimit    = State["AnomalyLimit"]                             # Schwellenwert laden
        print(f"Modell geladen von: {Path}")


