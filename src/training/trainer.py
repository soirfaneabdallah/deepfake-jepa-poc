"""
Classe de base pour l'entraînement des modèles.
Fournit les fonctionnalités communes : logging, checkpoints, early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
from tqdm import tqdm
import wandb

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration de l'entraînement."""
    # Optimisation
    learning_rate: float = 0.0001
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    gradient_clip: float = 1.0
    
    # Entraînement
    epochs: int = 100
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    mixed_precision: bool = True
    
    # Scheduler
    scheduler_type: str = 'cosine'  # cosine, step, plateau, none
    warmup_epochs: int = 10
    min_lr: float = 0.000001
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 0.001
    
    # Checkpoints
    save_checkpoints: bool = True
    checkpoint_frequency: int = 5
    save_best: bool = True
    
    # Logging
    log_frequency: int = 10
    use_wandb: bool = False
    wandb_project: str = "deepfake-vjepa"
    
    # Validation
    validation_frequency: int = 1

class EarlyStopping:
    """
    Early stopping pour éviter le surapprentissage.
    """
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.001,
        mode: str = 'min'  # min pour loss, max pour accuracy
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(
        self,
        score: float,
        model: nn.Module
    ) -> bool:
        """
        Vérifie si l'entraînement doit s'arrêter.
        
        Returns:
            True si l'entraînement doit s'arrêter
        """
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            return False
        
        if self.mode == 'min':
            improvement = self.best_score - score > self.min_delta
        else:
            improvement = score - self.best_score > self.min_delta
        
        if improvement:
            self.best_score = score
            self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def load_best_model(self, model: nn.Module) -> None:
        """Charge le meilleur modèle."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

class CheckpointManager:
    """
    Gestion des checkpoints du modèle.
    """
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ) -> str:
        """
        Sauvegarde un checkpoint complet.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Sauvegarde standard
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, path)
        
        # Sauvegarde du meilleur modèle
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Meilleur modèle sauvegardé: {best_path}")
        
        logger.debug(f"Checkpoint sauvegardé: {path}")
        return str(path)
    
    def load_checkpoint(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ) -> Tuple[int, Dict[str, float]]:
        """
        Charge un checkpoint.
        """
        checkpoint = torch.load(path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics']
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Retourne le checkpoint le plus récent."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if not checkpoints:
            return None
        
        latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        return str(latest)

class MetricsTracker:
    """
    Suivi des métriques d'entraînement.
    """
    
    def __init__(self):
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
    def update(
        self,
        metrics: Dict[str, float],
        phase: str = 'train'
    ) -> None:
        """
        Met à jour l'historique des métriques.
        """
        for key, value in metrics.items():
            full_key = f"{phase}_{key}" if not key.startswith(phase) else key
            if full_key not in self.metrics_history:
                self.metrics_history[full_key] = []
            self.metrics_history[full_key].append(value)
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """Retourne l'historique des métriques."""
        return self.metrics_history
    
    def save(self, path: str) -> None:
        """Sauvegarde l'historique en JSON."""
        with open(path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def load(self, path: str) -> None:
        """Charge l'historique depuis JSON."""
        with open(path, 'r') as f:
            self.metrics_history = json.load(f)

class BaseTrainer:
    """
    Classe de base pour tous les entraîneurs.
    Fournit les fonctionnalités communes d'entraînement.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Gestionnaires
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.metrics_tracker = MetricsTracker()
        
        # Optimiseur
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            mode='min'
        ) if config.early_stopping else None
        
        # Mixed precision
        self.scaler = GradScaler(enabled=config.mixed_precision)
        
        # État
        self.current_epoch = 0
        self.best_metric = float('inf')
        
    def _create_optimizer(self) -> optim.Optimizer:
        """
        Crée l'optimiseur avec weight decay sélectif.
        """
        # Séparation des paramètres
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name or 'pos_embed' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        param_groups = [
            {
                'params': decay_params,
                'weight_decay': self.config.weight_decay
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0
            }
        ]
        
        return optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=self.config.eps
        )
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Crée le scheduler d'apprentissage.
        """
        if self.config.scheduler_type == 'none':
            return None
        
        total_steps = self.config.epochs
        
        if self.config.scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=self.config.min_lr
            )
        else:
            raise ValueError(f"Scheduler non supporté: {self.config.scheduler_type}")
        
        return scheduler
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Entraîne le modèle pendant une époque.
        À implémenter dans les sous-classes.
        """
        raise NotImplementedError
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Valide le modèle.
        À implémenter dans les sous-classes.
        """
        raise NotImplementedError
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Boucle d'entraînement principale.
        """
        # Reprise depuis un checkpoint
        if resume_from is not None:
            self.current_epoch, metrics = self.checkpoint_manager.load_checkpoint(
                resume_from,
                self.model,
                self.optimizer,
                self.scheduler
            )
            logger.info(f"Reprise depuis l'époque {self.current_epoch}")
        
        # Boucle d'entraînement
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            # Phase d'entraînement
            train_metrics = self.train_epoch(train_loader, epoch)
            self.metrics_tracker.update(train_metrics, phase='train')
            
            # Logging
            if epoch % self.config.log_frequency == 0:
                logger.info(f"Époque {epoch}: train_loss={train_metrics['loss']:.4f}")
                
                if self.config.use_wandb:
                    wandb.log({f'train/{k}': v for k, v in train_metrics.items()})
            
            # Phase de validation
            if val_loader is not None and epoch % self.config.validation_frequency == 0:
                val_metrics = self.validate(val_loader)
                self.metrics_tracker.update(val_metrics, phase='val')
                
                logger.info(f"Époque {epoch}: val_loss={val_metrics['loss']:.4f}")
                
                if self.config.use_wandb:
                    wandb.log({f'val/{k}': v for k, v in val_metrics.items()})
                
                # Early stopping
                if self.early_stopping is not None:
                    should_stop = self.early_stopping(
                        val_metrics['loss'],
                        self.model
                    )
                    
                    if should_stop:
                        logger.info(f"Early stopping à l'époque {epoch}")
                        self.early_stopping.load_best_model(self.model)
                        break
                
                # Sauvegarde du meilleur modèle
                if val_metrics['loss'] < self.best_metric:
                    self.best_metric = val_metrics['loss']
                    is_best = True
                else:
                    is_best = False
            else:
                val_metrics = {}
                is_best = False
            
            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                else:
                    self.scheduler.step()
            
            # Sauvegarde des checkpoints
            if self.config.save_checkpoints and epoch % self.config.checkpoint_frequency == 0:
                self.checkpoint_manager.save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    {**train_metrics, **val_metrics},
                    is_best=is_best,
                    scheduler=self.scheduler
                )
        
        # Sauvegarde finale
        if self.config.save_checkpoints:
            self.checkpoint_manager.save_checkpoint(
                self.model,
                self.optimizer,
                self.current_epoch,
                self.metrics_tracker.get_metrics(),
                is_best=True,
                scheduler=self.scheduler
            )
        
        # Sauvegarde des métriques
        self.metrics_tracker.save(
            self.checkpoint_manager.checkpoint_dir / "metrics_history.json"
        )
        
        return self.metrics_tracker.get_metrics()
    
    def predict(
        self,
        data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prédit sur un ensemble de données.
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Prédiction"):
                if isinstance(batch, dict):
                    inputs = batch['frames']
                    targets = batch.get('label', None)
                else:
                    inputs, targets = batch
                
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                predictions = outputs.argmax(dim=1).cpu().numpy()
                all_predictions.extend(predictions)
                
                if targets is not None:
                    all_targets.extend(targets.numpy())
        
        return np.array(all_predictions), np.array(all_targets)