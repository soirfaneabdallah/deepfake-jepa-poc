"""
Trainer v-JEPA complet avec toutes les dépendances
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from tqdm import tqdm
from pathlib import Path
import os

logger = logging.getLogger(__name__)

# === CONFIGURATIONS ===

@dataclass
class TrainingConfig:
    """Configuration de base pour l'entraînement."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    epochs: int = 50
    warmup_epochs: int = 10
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    log_frequency: int = 10
    save_frequency: int = 5
    val_frequency: int = 1
    early_stopping_patience: int = 10

@dataclass
class JEPATrainingConfig(TrainingConfig):
    """Configuration spécifique pour l'entraînement v-JEPA."""
    spatial_mask_ratio: float = 0.75
    temporal_mask_ratio: float = 0.90
    ema_decay: float = 0.998
    ema_end_decay: float = 0.9998
    ema_anneal_steps: int = 100000
    loss_type: str = 'smooth_l1'
    temperature: float = 0.1
    use_vicreg: bool = False
    num_augmentations: int = 2

# === EARLY STOPPING ===

class EarlyStopping:
    """Arrêt précoce pour éviter le sur-apprentissage."""
    
    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop

# === BASE TRAINER ===

class BaseTrainer:
    """Trainer de base."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints'
    ):
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-6
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience
        )
        
        # Métriques
        self.train_losses = []
        self.val_losses = []
        self.best_loss = float('inf')
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        raise NotImplementedError
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        raise NotImplementedError
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Boucle d'entraînement principale."""
        for epoch in range(1, self.config.epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print("-" * 50)
            
            # Entraînement
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_metrics.get('loss', 0))
            
            # Validation
            if val_loader and epoch % self.config.val_frequency == 0:
                val_metrics = self.validate(val_loader)
                self.val_losses.append(val_metrics.get('loss', 0))
                
                print(f"Train Loss: {train_metrics.get('loss', 0):.4f}")
                print(f"Val Loss: {val_metrics.get('loss', 0):.4f}")
                
                # Sauvegarde du meilleur modèle
                val_loss = val_metrics.get('loss', 0)
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"✓ Best model saved (loss: {val_loss:.4f})")
                
                # Early stopping
                if self.early_stopping(val_loss):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
            else:
                print(f"Train Loss: {train_metrics.get('loss', 0):.4f}")
            
            # Sauvegarde périodique
            if epoch % self.config.save_frequency == 0:
                self.save_checkpoint(epoch)
            
            # Scheduler
            self.scheduler.step()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Sauvegarde un checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': self.best_loss
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Charge un checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Loaded checkpoint: {checkpoint_path}")

# === JEPA LOSS CALCULATOR ===

class JEPALossCalculator:
    """Calcul des pertes pour v-JEPA."""
    
    def __init__(self, config: JEPATrainingConfig):
        self.config = config
        
    def compute_loss(
        self,
        context_pred: torch.Tensor,
        target_features: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Calcule la perte totale."""
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
        
        # Régularisation VICReg
        if self.config.use_vicreg:
            variance_loss = self._variance_regularization(context_pred)
            covariance_loss = self._covariance_regularization(context_pred)
            losses['variance'] = variance_loss
            losses['covariance'] = covariance_loss
        
        losses['total'] = sum(losses.values())
        
        return losses
    
    def _variance_regularization(self, features: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(features.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1 - std))
    
    def _covariance_regularization(self, features: torch.Tensor) -> torch.Tensor:
        features = features - features.mean(dim=0)
        cov = (features.T @ features) / (features.size(0) - 1)
        off_diagonal = cov - torch.diag(torch.diag(cov))
        return torch.sum(off_diagonal ** 2) / features.size(1)

# === JEPA TRAINER ===

class JEPATrainer(BaseTrainer):
    """Entraîneur v-JEPA."""
    
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
        """Entraîne le modèle pendant une époque."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Extraction des données
            if isinstance(batch, dict):
                x_real = batch['frames']
            elif isinstance(batch, (list, tuple)):
                x_real = batch[0]
            else:
                x_real = batch
            
            x_real = x_real.to(self.device)
            
            # Création des vues augmentées (simplifiée)
            x_context = x_real
            x_target = x_real
            
            # Génération des masques
            context_mask, target_mask = self.model.generate_mask(
                x_context.size(0),
                self.device
            )
            
            if self.scaler is not None:
                with autocast():
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
                    
                    loss = losses['total'] / self.config.gradient_accumulation_steps
                
                # Backward avec gradient accumulation
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.model.update_target_encoder()
                    self.optimizer.zero_grad()
            else:
                # Forward
                context_pred, target_features = self.model(
                    x_context,
                    x_target,
                    context_mask,
                    target_mask
                )
                
                losses = self.loss_calculator.compute_loss(
                    context_pred,
                    target_features,
                    context_mask,
                    target_mask
                )
                
                loss = losses['total'] / self.config.gradient_accumulation_steps
                
                # Backward
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                    self.optimizer.step()
                    self.model.update_target_encoder()
                    self.optimizer.zero_grad()
            
            total_loss += losses['total'].item()
            num_batches += 1
            
            if batch_idx % self.config.log_frequency == 0:
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'main': f"{losses['main'].item():.4f}"
                })
        
        return {'loss': total_loss / num_batches}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Valide le modèle."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    x_real = batch['frames']
                elif isinstance(batch, (list, tuple)):
                    x_real = batch[0]
                else:
                    x_real = batch
                
                x_real = x_real.to(self.device)
                
                context_mask, target_mask = self.model.generate_mask(
                    x_real.size(0),
                    self.device
                )
                
                context_pred, target_features = self.model(
                    x_real,
                    x_real,
                    context_mask,
                    target_mask
                )
                
                losses = self.loss_calculator.compute_loss(
                    context_pred,
                    target_features
                )
                
                total_loss += losses['total'].item()
                num_batches += 1
        
        return {'loss': total_loss / num_batches}
    
    def extract_features(
        self,
        data_loader: DataLoader
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extrait les features latentes."""
        self.model.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extraction"):
                if isinstance(batch, dict):
                    x = batch['frames']
                    labels = batch.get('label', None)
                elif isinstance(batch, (list, tuple)):
                    x, labels = batch[0], batch[1] if len(batch) > 1 else None
                else:
                    x = batch
                    labels = None
                
                x = x.to(self.device)
                
                features = self.model.encode(x)
                all_features.append(features.cpu())
                
                if labels is not None:
                    if isinstance(labels, torch.Tensor):
                        all_labels.append(labels)
                    else:
                        all_labels.append(torch.tensor(labels))
        
        features = torch.cat(all_features, dim=0)
        
        if all_labels:
            labels = torch.cat(all_labels, dim=0)
        else:
            labels = None
        
        return features, labels

# === FONCTION D'ENTRAÎNEMENT PRINCIPALE ===

def train_vjepa(
    dataset,
    config: JEPATrainingConfig = None,
    device: str = 'cuda',
    checkpoint_dir: str = './checkpoints/vjepa'
):
    """Fonction principale d'entraînement v-JEPA."""
    
    from src.models.vjepa import VJEPAModel, VJEPAConfig
    
    if config is None:
        config = JEPATrainingConfig(
            batch_size=16,
            epochs=50,
            learning_rate=1e-4,
            spatial_mask_ratio=0.75,
            temporal_mask_ratio=0.90
        )
    
    # Configuration du modèle
    model_config = VJEPAConfig(
        input_size=(224, 224),
        num_frames=8,
        embed_dim=384,
        depth=6,
        num_heads=6,
        predictor_depth=2
    )
    
    # Device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Créer le modèle
    model = VJEPAModel(model_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer
    trainer = JEPATrainer(
        model=model,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir
    )
    
    # Entraînement
    trainer.train(train_loader, val_loader)
    
    return trainer