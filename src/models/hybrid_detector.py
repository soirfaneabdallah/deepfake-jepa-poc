"""
Détecteur hybride combinant v-JEPA et analyse médico-légale.
Fusion multi-modale pour une détection robuste des deepfakes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

class MultiModalFusion(nn.Module):
    """
    Fusion multi-modale pour combiner différentes sources de features.
    """
    
    def __init__(
        self,
        input_dims: List[int],
        fusion_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dims = input_dims
        self.fusion_dim = fusion_dim
        
        # Projections des différentes modalités
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for dim in input_dims
        ])
        
        # Attention multi-modale
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion finale
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * len(input_dims), fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fusionne les différentes modalités.
        
        Args:
            features: Liste des features de différentes modalités
            
        Returns:
            fused: Features fusionnées
        """
        # Projection de chaque modalité
        projected = [
            proj(feat)
            for proj, feat in zip(self.projections, features)
        ]
        
        # Stack des features projetées
        stacked = torch.stack(projected, dim=1)  # (B, M, D)
        
        # Cross-attention entre modalités
        attended, _ = self.cross_attention(
            stacked, stacked, stacked
        )
        
        # Reshape et fusion
        B, M, D = attended.shape
        attended = attended.reshape(B, M * D)
        
        fused = self.fusion_layer(attended)
        
        return fused

class AdaptiveFusion(nn.Module):
    """
    Fusion adaptative avec attention sur les modalités.
    Apprend à pondérer les différentes sources.
    """
    
    def __init__(
        self,
        input_dims: List[int],
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.input_dims = input_dims
        
        # Réseau d'attention pour les poids
        self.attention = nn.Sequential(
            nn.Linear(sum(input_dims), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(input_dims)),
            nn.Softmax(dim=1)
        )
        
        # Projections
        self.projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim)
            for dim in input_dims
        ])
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fusionne les features avec des poids adaptatifs.
        """
        # Concatenation pour l'attention
        concat = torch.cat(features, dim=1)
        
        # Calcul des poids d'attention
        weights = self.attention(concat)  # (B, M)
        
        # Projection et pondération
        projected = [
            proj(feat) * weights[:, i].unsqueeze(1)
            for i, (proj, feat) in enumerate(zip(self.projections, features))
        ]
        
        # Somme pondérée
        fused = torch.stack(projected, dim=1).sum(dim=1)
        
        return fused

class HybridDeepfakeDetector(nn.Module):
    """
    Détecteur hybride complet combinant v-JEPA et analyse médico-légale.
    """
    
    def __init__(
        self,
        vjepa_model: nn.Module,
        forensic_analyzer: nn.Module,
        jepa_dim: int = 512,
        forensic_dim: int = 256,
        fusion_type: str = 'attention',  # attention, adaptive, concat
        num_classes: int = 2
    ):
        super().__init__()
        
        self.vjepa_model = vjepa_model
        self.forensic_analyzer = forensic_analyzer
        self.fusion_type = fusion_type
        
        # Dimensions des features
        self.jepa_dim = jepa_dim
        self.forensic_dim = forensic_dim
        
        # Fusion
        if fusion_type == 'attention':
            self.fusion = MultiModalFusion(
                input_dims=[jepa_dim, forensic_dim],
                fusion_dim=256
            )
            fusion_output_dim = 256
        elif fusion_type == 'adaptive':
            self.fusion = AdaptiveFusion(
                input_dims=[jepa_dim, forensic_dim],
                hidden_dim=128
            )
            fusion_output_dim = 128
        else:  # concat
            self.fusion = None
            fusion_output_dim = jepa_dim + forensic_dim
        
        # Classifieur final
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Détecteur d'anomalies associé
        self.anomaly_detector = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass complet du détecteur hybride.
        
        Args:
            x: (B, C, T, H, W) - Vidéo
            return_features: Retourner les features fusionnées
            
        Returns:
            logits: (B, num_classes) ou (logits, features)
        """
        # Features v-JEPA
        jepa_features = self.vjepa_model.encode(x)
        
        # Features forensiques
        forensic_features = self.forensic_analyzer(x)
        
        # Fusion
        if self.fusion_type in ['attention', 'adaptive']:
            fused_features = self.fusion([jepa_features, forensic_features])
        else:
            fused_features = torch.cat([jepa_features, forensic_features], dim=1)
        
        # Classification
        logits = self.classifier(fused_features)
        
        if return_features:
            return logits, fused_features
        
        return logits
    
    def compute_anomaly_score(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule le score d'anomalie pour la détection one-class.
        """
        if self.anomaly_detector is None:
            raise RuntimeError("Détecteur d'anomalies non initialisé")
        
        jepa_features = self.vjepa_model.encode(x)
        score = self.anomaly_detector.score(jepa_features)
        
        return score
    
    def set_anomaly_detector(self, detector: nn.Module) -> None:
        """
        Associe un détecteur d'anomalies.
        """
        self.anomaly_detector = detector
    
    def get_grad_cam_targets(self) -> List[nn.Module]:
        """
        Retourne les couches cibles pour Grad-CAM.
        """
        targets = []
        
        # Couches de l'encodeur v-JEPA
        if hasattr(self.vjepa_model, 'context_encoder'):
            targets.append(self.vjepa_model.context_encoder)
        
        # Couches du classifieur
        targets.append(self.classifier)
        
        return targets