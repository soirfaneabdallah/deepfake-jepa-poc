"""
Entraîneur spécifique pour v-JEPA.
Gère l'apprentissage auto-supervisé avec masquage spatio-temporel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from tqdm import tqdm

from .trainer import BaseTrainer, TrainingConfig, EarlyStopping

logger = logging.getLogger(__name__)

@dataclass
class JEPATrainingConfig(TrainingConfig):
    """Configuration spécifique pour l'entraînement v-JEPA."""
    # Masquage
    spatial_mask_ratio: float = 0.75
    temporal_mask_ratio: float = 0.90
    
    # EMA
    ema_decay: float = 0.998
    ema_end_decay: float = 0.9998
    ema_anneal_steps: int = 100000
    
    # Perte
    loss_type: str = 'smooth_l1'  # smooth_l1, cosine, mse
    temperature: float = 0.1
    use_vicreg: bool = False
    
    # Augmentations
    num_augmentations: int = 2

class JEPALossCalculator:
    """
    Calcul des pertes pour l'entraînement v-JEPA.
    """
    
    def __init__(self, config: JEPATrainingConfig):
        self.config = config
        
    def compute_loss(
        self,
        context_pred: torch.Tensor,
        target_features: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calcule la perte totale.
        """
        losses = {}
        
        # Perte principale
        if self.config.loss_type == 'smooth_l1':
            main_loss = F.smooth_l1_loss(
                context_pred,
                target_features.detach()
            )
        elif self.config.loss_type == 'cosine':
            context_pred_norm = F.normalize(context_pred, dim=-1)
            target_features_norm = F.normalize(target_features, dim=-1)
            main_loss = -(context_pred_norm * target_features_norm.detach()).sum(dim=-1).mean()
        else:  # mse
            main_loss = F.mse_loss(
                context_pred,
                target_features.detach()
            )
        
        losses['main'] = main_loss
        
        # Régularisation VICReg si activée
        if self.config.use_vicreg:
            variance_loss = self._variance_regularization(context_pred)
            covariance_loss = self._covariance_regularization(context_pred)
            
            losses['variance'] = variance_loss
            losses['covariance'] = covariance_loss
        
        # Perte totale
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def _variance_regularization(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Régularisation de la variance pour éviter le collapse.
        """
        std = torch.sqrt(features.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1 - std))
    
    def _covariance_regularization(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Régularisation de la covariance pour décorréler les features.
        """
        features = features - features.mean(dim=0)
        cov = (features.T @ features) / (features.size(0) - 1)
        
        # Pénalité sur les termes hors-diagonale
        off_diagonal = cov - torch.diag(torch.diag(cov))
        return torch.sum(off_diagonal ** 2) / features.size(1)

class JEPATrainer(BaseTrainer):
    """
    Entraîneur v-JEPA pour l'apprentissage auto-supervisé.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: JEPATrainingConfig,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints/jepa'
    ):
        super().__init__(model, config, device, checkpoint_dir)
        
        self.jepa_config = config
        self.loss_calculator = JEPALossCalculator(config)
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Entraîne le modèle pendant une époque.
        """
        self.model.train()
        total_losses = {'total': 0.0, 'main': 0.0}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Époque {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Extraction des données
            if isinstance(batch, dict):
                x_real = batch['frames']
            else:
                x_real = batch
            
            x_real = x_real.to(self.device)
            
            # Création des vues augmentées
            x_context = self._create_view(x_real)
            x_target = self._create_view(x_real)
            
            # Génération des masques
            context_mask, target_mask = self.model.generate_mask(
                x_context.size(0),
                self.device
            )
            
            with autocast(enabled=self.config.mixed_precision):
                # Forward pass
                context_pred, target_features = self.model(
                    x_context,
                    x_target,
                    context_mask,
                    target_mask
                )
                
                # Calcul des pertes
                losses = self.loss_calculator.compute_loss(
                    context_pred,
                    target_features,
                    context_mask,
                    target_mask
                )
                
                # Gradient accumulation
                loss = losses['total'] / self.config.gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Mise à jour de l'encodeur cible
                self.model.update_target_encoder()
                
                self.optimizer.zero_grad()
            
            # Accumulation des pertes
            for key in total_losses:
                total_losses[key] += losses.get(key, 0.0).item()
            num_batches += 1
            
            # Logging
            if batch_idx % self.config.log_frequency == 0:
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'main': f"{losses['main'].item():.4f}"
                })
        
        # Moyenne des pertes
        avg_losses = {
            key: value / num_batches
            for key, value in total_losses.items()
        }
        
        return {'loss': avg_losses['total']}
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Valide le modèle v-JEPA.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    x_real = batch['frames']
                else:
                    x_real = batch
                
                x_real = x_real.to(self.device)
                
                # Génération des masques
                context_mask, target_mask = self.model.generate_mask(
                    x_real.size(0),
                    self.device
                )
                
                # Forward pass
                context_pred, target_features = self.model(
                    x_real,
                    x_real,
                    context_mask,
                    target_mask
                )
                
                # Calcul de la perte
                losses = self.loss_calculator.compute_loss(
                    context_pred,
                    target_features
                )
                
                total_loss += losses['total'].item()
                num_batches += 1
        
        return {'loss': total_loss / num_batches}
    
    def _create_view(self, x: torch.Tensor) -> torch.Tensor:
        """
        Crée une vue augmentée de la vidéo.
        """
        # Augmentations spatiales et temporelles légères
        # (les augmentations principales sont dans le dataset)
        return x
    
    def extract_features(
        self,
        data_loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extrait les features latentes pour la détection d'anomalies.
        """
        self.model.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extraction des features"):
                if isinstance(batch, dict):
                    x = batch['frames']
                    labels = batch.get('label', None)
                else:
                    x, labels = batch
                
                x = x.to(self.device)
                
                # Extraction des features
                features = self.model.encode(x)
                all_features.append(features.cpu())
                
                if labels is not None:
                    all_labels.append(labels)
        
        features = torch.cat(all_features, dim=0)
        
        if all_labels:
            labels = torch.cat(all_labels, dim=0)
        else:
            labels = None
        
        return features, labels