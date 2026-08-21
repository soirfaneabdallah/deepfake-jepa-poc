"""
Modèles temporels pour l'analyse des séquences vidéo dans la détection de deepfakes.
Ces modèles capturent les dépendances temporelles qui sont souvent négligées
dans les approches basées uniquement sur les frames individuelles.

Les deepfakes présentent souvent des incohérences temporelles subtiles :
- Micro-expressions faciales anormales
- Transitions non naturelles entre les frames
- Incohérences dans le mouvement des yeux et de la bouche
- Artefacts de génération qui varient dans le temps

Ces modèles sont conçus pour capturer ces anomalies temporelles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List, Dict, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# ============================================
# Classes utilitaires
# ============================================

class PositionalEncoding(nn.Module):
    """
    Encodage positionnel sinusoïdal pour les séquences temporelles.
    Permet au modèle de comprendre l'ordre des frames.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Création de l'encodage positionnel
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) ou (T, B, D) - Séquence d'entrée
            
        Returns:
            x: Séquence avec encodage positionnel
        """
        if x.dim() == 3 and x.size(1) != x.size(2):
            # Format batch_first (B, T, D)
            x = x + self.pe[:x.size(1), 0].unsqueeze(0)
        else:
            # Format seq_first (T, B, D)
            x = x + self.pe[:x.size(0)]
        
        return self.dropout(x)

class TemporalAttention(nn.Module):
    """
    Attention temporelle pour pondérer l'importance des différentes frames.
    Permet au modèle de se concentrer sur les frames les plus informatives.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Projection de sortie
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) - Séquence d'entrée
            mask: (B, T) - Masque d'attention optionnel
            
        Returns:
            attended: (B, T, D) - Séquence avec attention appliquée
        """
        B, T, D = x.shape
        
        # Projections
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calcul de l'attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        
        # Application du masque si fourni
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Application de l'attention
        attended = (attn @ v).transpose(1, 2).reshape(B, T, D)
        
        # Projection de sortie avec résidu
        output = self.layer_norm(x + self.out_proj(attended))
        
        return output

class TemporalConvBlock(nn.Module):
    """
    Bloc convolutif temporel avec dilatation.
    Capture les dépendances locales à différentes échelles.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        # Convolutions
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        
        # Normalisation et activation
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Connexion résiduelle si dimensions différentes
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
                       if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) - Séquence d'entrée
            
        Returns:
            output: (B, C_out, T) - Séquence traitée
        """
        # Première convolution
        residual = self.residual(x)
        
        # Deuxième convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Connexion résiduelle
        return x + residual

# ============================================
# Modèles temporels principaux
# ============================================

class TemporalTransformer(nn.Module):
    """
    Transformer temporel pour l'analyse des séquences de features.
    
    Avantages :
    - Capture les dépendances à long terme
    - Parallélisable pour un entraînement efficace
    - Mécanisme d'attention pour les frames importantes
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_sequence_length: int = 100,
        use_cls_token: bool = True,
        pooling: str = 'cls'  # cls, mean, max, attention
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_cls_token = use_cls_token
        self.pooling = pooling
        
        # Projection d'entrée
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Token de classification
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Encodage positionnel
        self.pos_encoding = PositionalEncoding(
            d_model=hidden_dim,
            max_len=max_sequence_length + 1,  # +1 pour le token cls
            dropout=dropout
        )
        
        # Couches Transformer
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-LN pour meilleure stabilité
            )
            for _ in range(num_layers)
        ])
        
        # Normalisation finale
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Pooling
        if pooling == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Softmax(dim=1)
            )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) - Séquence de features
            mask: (B, T) - Masque optionnel
            
        Returns:
            features: (B, hidden_dim) - Features agrégées
        """
        B, T, D = x.shape
        
        # Projection d'entrée
        x = self.input_proj(x)
        
        # Ajout du token de classification
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, D)
            
            # Extension du masque
            if mask is not None:
                mask = torch.cat([torch.ones(B, 1, device=mask.device), mask], dim=1)
        
        # Encodage positionnel
        x = self.pos_encoding(x)
        
        # Passage dans les couches Transformer
        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=(~mask.bool() if mask is not None else None))
        
        # Normalisation finale
        x = self.final_norm(x)
        
        # Pooling
        if self.pooling == 'cls' and self.use_cls_token:
            features = x[:, 0]  # Token de classification
        elif self.pooling == 'mean':
            features = x.mean(dim=1)
        elif self.pooling == 'max':
            features = x.max(dim=1)[0]
        elif self.pooling == 'attention':
            attention_weights = self.attention_pool(x)  # (B, T, 1)
            features = (x * attention_weights).sum(dim=1)
        else:
            features = x[:, -1]  # Dernière position
        
        return features

class TemporalLSTM(nn.Module):
    """
    LSTM temporel bidirectionnel pour l'analyse séquentielle.
    
    Avantages :
    - Capture les dépendances à court et long terme
    - Bidirectionnel pour le contexte complet
    - Bien adapté aux séquences de longueur variable
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        use_attention: bool = True,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Dimension de sortie
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_dim = output_dim
        
        # Layer normalization optionnelle
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = nn.Identity()
        
        # Attention optionnelle
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(output_dim, output_dim // 2),
                nn.Tanh(),
                nn.Linear(output_dim // 2, 1),
                nn.Softmax(dim=1)
            )
        
        # Projection finale
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, output_dim // 4)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_all_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, T, D) - Séquence de features
            hidden: État caché initial
            return_all_outputs: Retourner toutes les sorties
            
        Returns:
            features: (B, output_dim // 4) ou (outputs, features)
        """
        # Passage dans le LSTM
        outputs, (h_n, c_n) = self.lstm(x, hidden)
        
        # Layer normalization
        outputs = self.layer_norm(outputs)
        
        # Agrégation des sorties
        if self.use_attention:
            # Attention sur les sorties
            attention_weights = self.attention(outputs)  # (B, T, 1)
            attended = (outputs * attention_weights).sum(dim=1)  # (B, output_dim)
        else:
            # Utilisation de la dernière sortie
            attended = outputs[:, -1]  # (B, output_dim)
        
        # Projection finale
        features = self.output_proj(attended)
        
        if return_all_outputs:
            return outputs, features
        
        return features

class TemporalGRU(nn.Module):
    """
    GRU temporel avec attention pour l'analyse séquentielle.
    
    Avantages :
    - Plus léger que LSTM
    - Convergence plus rapide
    - Attention pour les frames importantes
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        use_attention: bool = True,
        attention_type: str = 'additive'  # additive, multiplicative, self
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.attention_type = attention_type
        
        # GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Dimension de sortie
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_dim = output_dim
        
        # Attention
        if use_attention:
            if attention_type == 'additive':
                self.attention = AdditiveAttention(output_dim)
            elif attention_type == 'multiplicative':
                self.attention = MultiplicativeAttention(output_dim)
            else:  # self
                self.attention = TemporalAttention(output_dim)
        
        # Projection finale
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, output_dim // 4)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) - Séquence de features
            hidden: État caché initial
            
        Returns:
            features: (B, output_dim // 4)
        """
        # Passage dans le GRU
        outputs, h_n = self.gru(x, hidden)
        
        # Agrégation des sorties
        if self.use_attention:
            attended = self.attention(outputs)
        else:
            attended = outputs[:, -1]
        
        # Projection finale
        features = self.output_proj(attended)
        
        return features

class AdditiveAttention(nn.Module):
    """
    Attention additive (Bahdanau).
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            
        Returns:
            attended: (B, D)
        """
        # Calcul des scores d'attention
        scores = self.attention(x)  # (B, T, 1)
        
        # Normalisation
        weights = self.softmax(scores)  # (B, T, 1)
        
        # Application de l'attention
        attended = (x * weights).sum(dim=1)  # (B, D)
        
        return attended

class MultiplicativeAttention(nn.Module):
    """
    Attention multiplicative (Luong).
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.scale = hidden_dim ** -0.5
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            
        Returns:
            attended: (B, D)
        """
        # Calcul des scores
        scores = (x * self.query).sum(dim=-1) * self.scale  # (B, T)
        
        # Normalisation
        weights = self.softmax(scores)  # (B, T)
        
        # Application de l'attention
        attended = (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        
        return attended

class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN).
    
    Avantages :
    - Convolutions dilatées pour un champ réceptif large
    - Parallélisable comme les CNN
    - Connexions résiduelles pour la stabilité
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: List[int] = [256, 256, 256, 256],
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        pooling: str = 'mean'  # mean, max, attention
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.pooling = pooling
        
        # Couches de convolution dilatée
        self.conv_blocks = nn.ModuleList()
        
        in_channels = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            dilation = 2 ** i
            
            self.conv_blocks.append(
                TemporalConvBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
            
            in_channels = hidden_dim
        
        # Pooling
        if pooling == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dims[-1], 1),
                nn.Softmax(dim=2)
            )
        else:
            self.attention_pool = None
        
        # Projection finale
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, hidden_dims[-1] // 4)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) - Séquence de features
            
        Returns:
            features: (B, hidden_dims[-1] // 4)
        """
        # Transposition pour les convolutions
        x = x.transpose(1, 2)  # (B, D, T)
        
        # Passage dans les blocs convolutifs
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Pooling
        if self.pooling == 'mean':
            x = x.mean(dim=2)  # (B, D)
        elif self.pooling == 'max':
            x = x.max(dim=2)[0]  # (B, D)
        elif self.pooling == 'attention':
            weights = self.attention_pool(x.transpose(1, 2))  # (B, T, 1)
            x = (x * weights.transpose(1, 2)).sum(dim=2)  # (B, D)
        else:
            x = x[:, :, -1]  # Dernière position (B, D)
        
        # Projection finale
        features = self.output_proj(x)
        
        return features

class TemporalFusion(nn.Module):
    """
    Fusion de modèles temporels multiples.
    Combine les forces de différentes architectures.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_transformer: bool = True,
        use_lstm: bool = True,
        use_gru: bool = False,
        use_tcn: bool = False,
        fusion_type: str = 'concat'  # concat, attention, weighted
    ):
        super().__init__()
        
        self.models = nn.ModuleDict()
        self.fusion_type = fusion_type
        
        # Initialisation des modèles
        if use_transformer:
            self.models['transformer'] = TemporalTransformer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=8,
                num_layers=4,
                dropout=dropout
            )
        
        if use_lstm:
            self.models['lstm'] = TemporalLSTM(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=2,
                dropout=dropout,
                bidirectional=True
            )
        
        if use_gru:
            self.models['gru'] = TemporalGRU(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=2,
                dropout=dropout,
                bidirectional=True
            )
        
        if use_tcn:
            self.models['tcn'] = TemporalConvNet(
                input_dim=input_dim,
                hidden_dims=[hidden_dim] * 4,
                dropout=dropout
            )
        
        # Fusion
        num_models = len(self.models)
        total_dim = hidden_dim * num_models
        
        if fusion_type == 'attention':
            self.fusion = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        elif fusion_type == 'weighted':
            self.fusion_weights = nn.Parameter(torch.ones(num_models) / num_models)
        else:  # concat
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) - Séquence de features
            
        Returns:
            fused_features: (B, hidden_dim)
        """
        # Passage dans chaque modèle
        outputs = []
        for name, model in self.models.items():
            output = model(x)
            outputs.append(output)
        
        # Fusion
        if self.fusion_type == 'attention':
            stacked = torch.stack(outputs, dim=1)  # (B, M, D)
            fused, _ = self.fusion(stacked, stacked, stacked)
            fused = fused.mean(dim=1)  # (B, D)
        elif self.fusion_type == 'weighted':
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = sum(w * out for w, out in zip(weights, outputs))
        else:  # concat
            concat = torch.cat(outputs, dim=1)  # (B, M*D)
            fused = self.fusion(concat)
        
        return fused

# ============================================
# Factory functions
# ============================================

def create_temporal_model(
    model_type: str = 'transformer',
    **kwargs
) -> nn.Module:
    """
    Factory pour créer un modèle temporel.
    
    Args:
        model_type: Type de modèle ('transformer', 'lstm', 'gru', 'tcn', 'fusion')
        **kwargs: Arguments spécifiques au modèle
        
    Returns:
        model: Modèle temporel initialisé
    """
    models = {
        'transformer': TemporalTransformer,
        'lstm': TemporalLSTM,
        'gru': TemporalGRU,
        'tcn': TemporalConvNet,
        'fusion': TemporalFusion
    }
    
    if model_type not in models:
        raise ValueError(f"Modèle temporel non supporté: {model_type}")
    
    return models[model_type](**kwargs)

# ============================================
# Modèles spécifiques pour la détection de deepfakes
# ============================================

class DeepfakeTemporalAnalyzer(nn.Module):
    """
    Analyseur temporel spécifique pour la détection de deepfakes.
    Combine l'analyse temporelle avec la détection d'anomalies.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        temporal_model: str = 'transformer',
        use_anomaly_head: bool = True
    ):
        super().__init__()
        
        self.use_anomaly_head = use_anomaly_head
        
        # Modèle temporel principal
        self.temporal_model = create_temporal_model(
            model_type=temporal_model,
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )
        
        # Tête de classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Tête d'anomalie (optionnelle)
        if use_anomaly_head:
            self.anomaly_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        x: torch.Tensor,
        return_anomaly_score: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, T, D) - Séquence de features temporelles
            return_anomaly_score: Retourner le score d'anomalie
            
        Returns:
            logits: (B, num_classes) ou (logits, anomaly_score)
        """
        # Features temporelles
        temporal_features = self.temporal_model(x)
        
        # Classification
        logits = self.classifier(temporal_features)
        
        if return_anomaly_score and self.use_anomaly_head:
            anomaly_score = self.anomaly_head(temporal_features)
            return logits, anomaly_score.squeeze(-1)
        
        return logits