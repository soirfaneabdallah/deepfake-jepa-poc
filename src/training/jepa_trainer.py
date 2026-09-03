"""
Module d'entraînement v-JEPA corrigé
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path
import torchvision.transforms as T
from PIL import Image
import subprocess
import sys
import importlib

# === CONFIGURATIONS ===

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    epochs: int = 50
    warmup_epochs: int = 10
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    log_frequency: int = 10
    save_frequency: int = 5
    val_frequency: int = 1
    early_stopping_patience: int = 10

@dataclass
class JEPATrainingConfig(TrainingConfig):
    spatial_mask_ratio: float = 0.75
    temporal_mask_ratio: float = 0.90
    ema_decay: float = 0.998
    ema_end_decay: float = 0.9998
    ema_anneal_steps: int = 100000
    loss_type: str = 'smooth_l1'
    temperature: float = 0.1
    use_vicreg: bool = False
    num_augmentations: int = 2
    num_frames: int = 8


# === WRAPPER POUR CONVERTIR IMAGES EN VIDEOS ===

class VideoDatasetWrapper(Dataset):
    def __init__(self, dataset, num_frames=8, transform=None):
        self.dataset = dataset
        self.num_frames = num_frames
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        if isinstance(item, tuple) and len(item) == 2:
            img, label = item
        else:
            img = item
            label = 0
        
        if isinstance(img, Image.Image):
            img_tensor = self.transform(img)
        elif isinstance(img, torch.Tensor):
            img_tensor = img
        else:
            img_tensor = self.transform(Image.fromarray(img))
        
        video = img_tensor.unsqueeze(1).repeat(1, self.num_frames, 1, 1)
        
        return video, label


# === EARLY STOPPING ===

class EarlyStopping:
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
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints'
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Déplacer le modèle sur le device
        self.model = model.to(self.device)
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-6
        )
        
        self.scaler = torch.amp.GradScaler('cuda') if (config.mixed_precision and torch.cuda.is_available()) else None
        
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)
        self.train_losses = []
        self.val_losses = []
        self.best_loss = float('inf')
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        raise NotImplementedError
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        raise NotImplementedError
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        for epoch in range(1, self.config.epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print("-" * 50)
            
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_metrics.get('loss', 0))
            
            if val_loader and epoch % self.config.val_frequency == 0:
                val_metrics = self.validate(val_loader)
                self.val_losses.append(val_metrics.get('loss', 0))
                
                print(f"Train Loss: {train_metrics.get('loss', 0):.4f}")
                print(f"Val Loss: {val_metrics.get('loss', 0):.4f}")
                
                val_loss = val_metrics.get('loss', 0)
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"Best model saved (loss: {val_loss:.4f})")
                
                if self.early_stopping(val_loss):
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                print(f"Train Loss: {train_metrics.get('loss', 0):.4f}")
            
            if epoch % self.config.save_frequency == 0:
                self.save_checkpoint(epoch)
            
            self.scheduler.step()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
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
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))


# === JEPA LOSS CALCULATOR ===

class JEPALossCalculator:
    def __init__(self, config: JEPATrainingConfig):
        self.config = config
    
    def compute_loss(self, context_pred: torch.Tensor, target_features: torch.Tensor,
                     context_mask=None, target_mask=None) -> Dict[str, torch.Tensor]:
        losses = {}
        
        if self.config.loss_type == 'smooth_l1':
            main_loss = F.smooth_l1_loss(context_pred, target_features.detach())
        elif self.config.loss_type == 'cosine':
            context_pred_norm = F.normalize(context_pred, dim=-1)
            target_features_norm = F.normalize(target_features, dim=-1)
            main_loss = -(context_pred_norm * target_features_norm.detach()).sum(dim=-1).mean()
        else:
            main_loss = F.mse_loss(context_pred, target_features.detach())
        
        losses['main'] = main_loss
        
        if self.config.use_vicreg:
            std = torch.sqrt(context_pred.var(dim=0) + 1e-4)
            variance_loss = torch.mean(F.relu(1 - std))
            losses['variance'] = variance_loss
            
            features = context_pred - context_pred.mean(dim=0)
            cov = (features.T @ features) / (features.size(0) - 1)
            off_diagonal = cov - torch.diag(torch.diag(cov))
            covariance_loss = torch.sum(off_diagonal ** 2) / features.size(1)
            losses['covariance'] = covariance_loss
        
        losses['total'] = sum(losses.values())
        return losses


# === JEPA TRAINER ===

class JEPATrainer(BaseTrainer):
    def __init__(self, model: nn.Module, config: JEPATrainingConfig,
                 device: str = 'cuda', checkpoint_dir: str = './checkpoints/jepa'):
        super().__init__(model, config, device, checkpoint_dir)
        self.jepa_config = config
        self.loss_calculator = JEPALossCalculator(config)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, dict):
                videos = batch.get('frames', batch.get('video'))
                labels = batch.get('label', None)
            elif isinstance(batch, (list, tuple)):
                videos, labels = batch[0], batch[1] if len(batch) > 1 else None
            else:
                videos = batch
                labels = None
            
            videos = videos.to(self.device)
            
            T = videos.size(2)
            split = T // 2
            x_context = videos[:, :, :split]
            x_target = videos[:, :, split:]
            
            if x_context.size(2) < self.jepa_config.num_frames:
                pad = self.jepa_config.num_frames - x_context.size(2)
                x_context = F.pad(x_context, (0, 0, 0, 0, 0, pad))
                x_target = F.pad(x_target, (0, 0, 0, 0, 0, pad))
            
            context_mask, target_mask = self.model.generate_mask(
                x_context.size(0), self.device
            )
            
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    context_pred, target_features = self.model(
                        x_context, x_target, context_mask, target_mask
                    )
                    losses = self.loss_calculator.compute_loss(
                        context_pred, target_features, context_mask, target_mask
                    )
                    loss = losses['total'] / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.model.update_target_encoder()
                    self.optimizer.zero_grad()
            else:
                context_pred, target_features = self.model(
                    x_context, x_target, context_mask, target_mask
                )
                losses = self.loss_calculator.compute_loss(
                    context_pred, target_features, context_mask, target_mask
                )
                loss = losses['total'] / self.config.gradient_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                    self.optimizer.step()
                    self.model.update_target_encoder()
                    self.optimizer.zero_grad()
            
            total_loss += losses['total'].item()
            num_batches += 1
            
            if batch_idx % self.config.log_frequency == 0:
                pbar.set_postfix({'loss': f"{losses['total'].item():.4f}"})
        
        return {'loss': total_loss / num_batches}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    videos = batch.get('frames', batch.get('video'))
                elif isinstance(batch, (list, tuple)):
                    videos = batch[0]
                else:
                    videos = batch
                
                videos = videos.to(self.device)
                
                T = videos.size(2)
                split = T // 2
                x_context = videos[:, :, :split]
                x_target = videos[:, :, split:]
                
                if x_context.size(2) < self.jepa_config.num_frames:
                    pad = self.jepa_config.num_frames - x_context.size(2)
                    x_context = F.pad(x_context, (0, 0, 0, 0, 0, pad))
                    x_target = F.pad(x_target, (0, 0, 0, 0, 0, pad))
                
                context_mask, target_mask = self.model.generate_mask(
                    x_context.size(0), self.device
                )
                
                context_pred, target_features = self.model(
                    x_context, x_target, context_mask, target_mask
                )
                
                losses = self.loss_calculator.compute_loss(
                    context_pred, target_features
                )
                
                total_loss += losses['total'].item()
                num_batches += 1
        
        return {'loss': total_loss / num_batches}


# === FONCTION PRINCIPALE D'ENTRAÎNEMENT ===

def train_vjepa(
    dataset,
    config: JEPATrainingConfig = None,
    device: str = 'cuda',
    checkpoint_dir: str = './checkpoints/vjepa'
):
    from src.models.vjepa import VJEPAModel, VJEPAConfig
    
    if config is None:
        config = JEPATrainingConfig(
            batch_size=16,
            epochs=5,
            learning_rate=1e-4,
            mixed_precision=False,
            num_frames=8
        )
    
    model_config = VJEPAConfig(
        input_size=(224, 224),
        num_frames=config.num_frames,
        embed_dim=384,
        depth=6,
        num_heads=6,
        predictor_depth=2
    )
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Préparer les données
    print("\nPreparing data...")
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if not hasattr(dataset, 'transform') or dataset.transform is None:
        dataset.transform = transform
    
    video_dataset = VideoDatasetWrapper(
        dataset,
        num_frames=config.num_frames,
        transform=transform
    )
    
    print(f"Dataset size: {len(video_dataset)}")
    
    train_size = int(0.8 * len(video_dataset))
    val_size = len(video_dataset) - train_size
    train_ds, val_ds = random_split(video_dataset, [train_size, val_size])
    
    print(f"Train: {len(train_ds)} images")
    print(f"Val: {len(val_ds)} images")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Vérifier une batch
    for videos, labels in train_loader:
        print(f"Batch video shape: {videos.shape}")
        print(f"Batch labels: {labels.shape}")
        break
    
    # Créer le modèle et le déplacer sur le device
    model = VJEPAModel(model_config)
    model = model.to(device)  # CRUCIAL: déplacer le modèle sur GPU
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model device: {next(model.parameters()).device}")
    
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


