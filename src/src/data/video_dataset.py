"""
Fonctions de perte pour l'entraînement v-JEPA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class VJEPALoss(nn.Module):
    """
    Perte v-JEPA combinant plusieurs objectifs.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        use_vicreg: bool = False,
        variance_epsilon: float = 0.0001,
        covariance_epsilon: float = 0.0001
    ):
        super().__init__()
        self.temperature = temperature
        self.use_vicreg = use_vicreg
        self.variance_epsilon = variance_epsilon
        self.covariance_epsilon = covariance_epsilon
        
    def forward(
        self,
        context_pred: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la perte totale.
        
        Args:
            context_pred: Prédictions du contexte
            target_features: Features cibles
        """
        # Perte principale (Smooth L1)
        main_loss = self.smooth_l1_loss(context_pred, target_features)
        
        if self.use_vicreg:
            # Régularisation VICReg
            variance_loss = self.variance_regularization(context_pred)
            covariance_loss = self.covariance_regularization(context_pred)
            
            total_loss = main_loss + variance_loss + covariance_loss
        else:
            total_loss = main_loss
        
        return total_loss
    
    def smooth_l1_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Smooth L1 Loss avec normalisation.
        """
        # Normalisation L2
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        
        # Smooth L1
        loss = F.smooth_l1_loss(pred, target.detach())
        
        return loss
    
    def variance_regularization(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Régularisation de la variance (VICReg).
        """
        # Calcul de la variance
        std = torch.sqrt(features.var(dim=0) + self.variance_epsilon)
        
        # Pénalité si la variance est trop faible
        loss = torch.mean(F.relu(1 - std))
        
        return loss
    
    def covariance_regularization(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Régularisation de la covariance (VICReg).
        """
        # Centrage
        features = features - features.mean(dim=0)
        
        # Matrice de covariance
        cov = (features.T @ features) / (features.size(0) - 1)
        
        # Pénalité sur les termes hors-diagonale
        off_diagonal = cov - torch.diag(torch.diag(cov))
        loss = torch.sum(off_diagonal ** 2) / features.size(1)
        
        return loss

class AnomalyDetectionLoss(nn.Module):
    """
    Perte pour la détection d'anomalies dans l'espace latent.
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        real_features: torch.Tensor,
        fake_features: torch.Tensor,
        reference_mean: torch.Tensor,
        reference_cov_inv: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la perte de détection d'anomalies.
        
        Args:
            real_features: Features des vidéos réelles
            fake_features: Features des vidéos manipulées
            reference_mean: Moyenne de la distribution des vrais visages
            reference_cov_inv: Inverse de la covariance des vrais visages
        """
        # Distance de Mahalanobis pour les vrais visages
        real_diff = real_features - reference_mean
        real_dist = torch.sqrt(real_diff @ reference_cov_inv @ real_diff.T)
        
        # Distance de Mahalanobis pour les faux visages
        fake_diff = fake_features - reference_mean
        fake_dist = torch.sqrt(fake_diff @ reference_cov_inv @ fake_diff.T)
        
        # Perte contrastive
        loss = F.relu(real_dist - fake_dist + self.margin).mean()
        
        return loss