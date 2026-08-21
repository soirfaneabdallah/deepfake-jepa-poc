"""
Entraîneur pour l'apprentissage continu.
Gère l'adaptation aux nouveaux générateurs de deepfakes sans oubli.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
import copy
from collections import deque
from tqdm import tqdm

from .trainer import BaseTrainer, TrainingConfig

logger = logging.getLogger(__name__)

@dataclass
class ContinualConfig(TrainingConfig):
    """Configuration pour l'apprentissage continu."""
    # EWC
    ewc_lambda: float = 100.0
    fisher_ema_decay: float = 0.9
    
    # Mémoire épisodique
    memory_size: int = 200
    replay_batch_size: int = 16
    
    # Tâches
    num_tasks: int = 4
    epochs_per_task: int = 10
    
    # Évaluation
    evaluate_forgetting: bool = True

class TaskManager:
    """
    Gestion des tâches d'apprentissage continu.
    """
    
    def __init__(self):
        self.tasks = []
        self.current_task = 0
        
    def add_task(
        self,
        name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ) -> None:
        """Ajoute une nouvelle tâche."""
        self.tasks.append({
            'name': name,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader
        })
    
    def get_task(self, task_id: int) -> Dict:
        """Retourne une tâche spécifique."""
        if task_id >= len(self.tasks):
            raise ValueError(f"Tâche {task_id} non trouvée")
        return self.tasks[task_id]
    
    def get_all_tasks(self) -> List[Dict]:
        """Retourne toutes les tâches."""
        return self.tasks

class ForgettingTracker:
    """
    Suivi de l'oubli catastrophique.
    """
    
    def __init__(self):
        self.performance_history = []
        self.forgetting_scores = []
        
    def update(
        self,
        task_id: int,
        performance: Dict[str, float]
    ) -> None:
        """Met à jour les performances."""
        self.performance_history.append({
            'task_id': task_id,
            'performance': performance
        })
        
    def compute_forgetting(
        self,
        current_task: int
    ) -> float:
        """
        Calcule l'oubli catastrophique.
        """
        if current_task == 0:
            return 0.0
        
        # Performance initiale sur la tâche 0
        initial_performance = self.performance_history[0]['performance']
        
        # Performance actuelle sur la tâche 0
        current_performance = self.performance_history[-1]['performance']
        
        # Différence de performance
        forgetting = initial_performance['accuracy'] - current_performance['accuracy']
        
        return max(0.0, forgetting)
    
    def get_history(self) -> List[Dict]:
        """Retourne l'historique des performances."""
        return self.performance_history

class ContinualTrainer(BaseTrainer):
    """
    Entraîneur pour l'apprentissage continu avec EWC et mémoire.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ContinualConfig,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints/continual'
    ):
        super().__init__(model, config, device, checkpoint_dir)
        
        self.continual_config = config
        
        # Gestionnaires
        self.task_manager = TaskManager()
        self.forgetting_tracker = ForgettingTracker()
        
        # Mémoire épisodique
        self.episodic_memory = deque(maxlen=config.memory_size)
        
        # Fisher information pour EWC
        self.fisher_information = {}
        self.optimal_params = {}
        
    def add_task(
        self,
        name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ) -> None:
        """Ajoute une tâche au gestionnaire."""
        self.task_manager.add_task(name, train_loader, val_loader, test_loader)
    
    def train_all_tasks(self) -> Dict[str, Any]:
        """
        Entraîne le modèle sur toutes les tâches séquentiellement.
        """
        results = []
        
        for task_id in range(len(self.task_manager.tasks)):
            logger.info(f"Entraînement sur la tâche {task_id}")
            
            # Entraînement sur la tâche courante
            task_results = self.train_task(task_id)
            results.append(task_results)
            
            # Évaluation de l'oubli
            if self.continual_config.evaluate_forgetting:
                forgetting = self.forgetting_tracker.compute_forgetting(task_id)
                logger.info(f"Oubli après tâche {task_id}: {forgetting:.4f}")
        
        return {
            'task_results': results,
            'forgetting_history': self.forgetting_tracker.get_history()
        }
    
    def train_task(self, task_id: int) -> Dict[str, float]:
        """
        Entraîne le modèle sur une tâche spécifique.
        """
        task = self.task_manager.get_task(task_id)
        train_loader = task['train_loader']
        val_loader = task['val_loader']
        
        # Calcul de la Fisher information pour EWC
        if task_id > 0:
            self._compute_fisher_information(
                self.task_manager.get_task(task_id - 1)['train_loader']
            )
        
        # Entraînement sur la tâche
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(self.continual_config.epochs_per_task):
            # Phase d'entraînement
            train_metrics = self._train_epoch_with_replay(
                train_loader,
                epoch,
                task_id
            )
            
            # Phase de validation
            val_metrics = self._validate_task(val_loader)
            
            # Sauvegarde du meilleur modèle
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_model_state = copy.deepcopy(self.model.state_dict())
            
            logger.info(
                f"Tâche {task_id}, Époque {epoch}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )
        
        # Restauration du meilleur modèle
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Mise à jour de la mémoire épisodique
        self._update_memory(train_loader)
        
        # Évaluation sur toutes les tâches vues
        performance = self._evaluate_all_tasks(task_id)
        self.forgetting_tracker.update(task_id, performance)
        
        return {
            'task_id': task_id,
            'task_name': task['name'],
            'best_val_accuracy': best_val_acc,
            'performance': performance
        }
    
    def _train_epoch_with_replay(
        self,
        train_loader: DataLoader,
        epoch: int,
        task_id: int
    ) -> Dict[str, float]:
        """
        Entraîne avec replay de la mémoire épisodique.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Tâche {task_id}, Époque {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, dict):
                x = batch['frames']
                y = batch['label']
            else:
                x, y = batch
            
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            logits = self.model(x)
            loss = nn.CrossEntropyLoss()(logits, y)
            
            # Pénalité EWC
            if task_id > 0:
                loss += self._ewc_loss()
            
            # Replay de la mémoire
            if len(self.episodic_memory) > 0:
                mem_x, mem_y = self._sample_memory()
                mem_logits = self.model(mem_x)
                mem_loss = nn.CrossEntropyLoss()(mem_logits, mem_y)
                loss += mem_loss * 0.5
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % self.config.log_frequency == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return {'loss': total_loss / num_batches}
    
    def _validate_task(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Valide sur une tâche spécifique.
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    x = batch['frames']
                    y = batch['label']
                else:
                    x, y = batch
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                logits = self.model(x)
                predictions = logits.argmax(dim=1)
                
                correct += (predictions == y).sum().item()
                total += y.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {'accuracy': accuracy}
    
    def _compute_fisher_information(self, dataloader: DataLoader) -> None:
        """
        Calcule la Fisher information pour EWC.
        """
        self.model.eval()
        
        fisher = {}
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param)
        
        num_samples = 0
        for batch in dataloader:
            if isinstance(batch, dict):
                x = batch['frames']
                y = batch['label']
            else:
                x, y = batch
            
            x = x.to(self.device)
            y = y.to(self.device)
            
            self.model.zero_grad()
            logits = self.model(x)
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
            
            num_samples += x.size(0)
            
            # Limiter le nombre d'échantillons
            if num_samples >= 100:
                break
        
        # Normalisation
        for name in fisher:
            fisher[name] /= num_samples
        
        # Mise à jour avec EMA
        if self.fisher_information:
            for name in fisher:
                fisher[name] = (
                    self.continual_config.fisher_ema_decay * self.fisher_information[name] +
                    (1 - self.continual_config.fisher_ema_decay) * fisher[name]
                )
        
        self.fisher_information = fisher
        self.optimal_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
    
    def _ewc_loss(self) -> torch.Tensor:
        """
        Calcule la pénalité EWC.
        """
        loss = 0.0
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                loss += (fisher * (param - optimal) ** 2).sum()
        
        return self.continual_config.ewc_lambda * loss
    
    def _update_memory(self, dataloader: DataLoader) -> None:
        """
        Met à jour la mémoire épisodique.
        """
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x = batch['frames']
                    y = batch['label']
                else:
                    x, y = batch
                
                # Ajout à la mémoire
                for i in range(x.size(0)):
                    self.episodic_memory.append((x[i].cpu(), y[i].cpu()))
                
                break  # Un seul batch
    
    def _sample_memory(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Échantillonne depuis la mémoire épisodique.
        """
        batch_size = min(self.continual_config.replay_batch_size, len(self.episodic_memory))
        indices = np.random.choice(len(self.episodic_memory), batch_size, replace=False)
        
        x_batch = []
        y_batch = []
        
        for idx in indices:
            x, y = self.episodic_memory[idx]
            x_batch.append(x)
            y_batch.append(y)
        
        return (
            torch.stack(x_batch).to(self.device),
            torch.tensor(y_batch).to(self.device)
        )
    
    def _evaluate_all_tasks(self, current_task: int) -> Dict[str, float]:
        """
        Évalue sur toutes les tâches vues.
        """
        performance = {}
        
        for task_id in range(current_task + 1):
            task = self.task_manager.get_task(task_id)
            val_metrics = self._validate_task(task['val_loader'])
            performance[f'task_{task_id}'] = val_metrics['accuracy']
        
        return performance