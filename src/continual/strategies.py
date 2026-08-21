"""
Stratégies d'apprentissage continu pour la détection de deepfakes.
Chaque stratégie combine différemment EWC, replay et distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import logging
import copy

from .ewc import ElasticWeightConsolidation
from .memory import DualMemory, EpisodicMemory

logger = logging.getLogger(__name__)

class ContinualStrategy(ABC):
    """
    Classe abstraite pour les stratégies d'apprentissage continu.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        
    @abstractmethod
    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10
    ) -> Dict[str, float]:
        """
        Entraîne le modèle sur une tâche.
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: int
    ) -> torch.Tensor:
        """
        Calcule la perte totale avec les régularisations.
        """
        pass

class EWCStrategy(ContinualStrategy):
    """
    Stratégie EWC pure.
    Utilise uniquement la pénalité EWC pour préserver les connaissances.
    """
    
    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 100.0,
        device: str = 'cuda'
    ):
        super().__init__(model, device)
        self.ewc = ElasticWeightConsolidation(model, lambda_ewc=ewc_lambda)
        
    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10
    ) -> Dict[str, float]:
        # Calcul de la Fisher pour la tâche précédente
        if task_id > 0:
            self.ewc.compute_fisher_information(train_loader)
        
        # Entraînement
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        best_acc = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            for batch in train_loader:
                if isinstance(batch, dict):
                    x = batch['frames']
                    y = batch['label']
                else:
                    x, y = batch
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                outputs = self.model(x)
                loss = F.cross_entropy(outputs, y)
                
                # Pénalité EWC
                if task_id > 0:
                    loss += self.ewc.penalty()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            acc = self._validate(val_loader)
            if acc > best_acc:
                best_acc = acc
        
        return {'accuracy': best_acc}
    
    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: int
    ) -> torch.Tensor:
        loss = F.cross_entropy(outputs, targets)
        
        if task_id > 0:
            loss += self.ewc.penalty()
        
        return loss
    
    def _validate(self, val_loader: DataLoader) -> float:
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
                
                outputs = self.model(x)
                predictions = outputs.argmax(dim=1)
                
                correct += (predictions == y).sum().item()
                total += y.size(0)
        
        return correct / total if total > 0 else 0.0

class ReplayStrategy(ContinualStrategy):
    """
    Stratégie de replay avec mémoire épisodique.
    Rejoue les échantillons anciens pendant l'entraînement.
    """
    
    def __init__(
        self,
        model: nn.Module,
        memory_size: int = 200,
        replay_batch_size: int = 16,
        device: str = 'cuda'
    ):
        super().__init__(model, device)
        self.memory = EpisodicMemory(capacity=memory_size)
        self.replay_batch_size = replay_batch_size
        
    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10
    ) -> Dict[str, float]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        best_acc = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            for batch in train_loader:
                if isinstance(batch, dict):
                    x = batch['frames']
                    y = batch['label']
                else:
                    x, y = batch
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Forward pass sur les données courantes
                outputs = self.model(x)
                loss = F.cross_entropy(outputs, y)
                
                # Replay de la mémoire
                if len(self.memory) > 0:
                    mem_x, mem_y = self.memory.sample(self.replay_batch_size)
                    mem_x = mem_x.to(self.device)
                    mem_y = mem_y.to(self.device)
                    
                    mem_outputs = self.model(mem_x)
                    mem_loss = F.cross_entropy(mem_outputs, mem_y)
                    loss += mem_loss * 0.5
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            acc = self._validate(val_loader)
            if acc > best_acc:
                best_acc = acc
        
        # Mise à jour de la mémoire
        self._update_memory(train_loader)
        
        return {'accuracy': best_acc}
    
    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: int
    ) -> torch.Tensor:
        loss = F.cross_entropy(outputs, targets)
        
        if len(self.memory) > 0:
            mem_x, mem_y = self.memory.sample(self.replay_batch_size)
            mem_outputs = self.model(mem_x.to(self.device))
            mem_loss = F.cross_entropy(mem_outputs, mem_y.to(self.device))
            loss += mem_loss * 0.5
        
        return loss
    
    def _update_memory(self, train_loader: DataLoader) -> None:
        """Met à jour la mémoire avec de nouveaux échantillons."""
        self.model.eval()
        
        with torch.no_grad():
            for batch in train_loader:
                if isinstance(batch, dict):
                    x = batch['frames']
                    y = batch['label']
                else:
                    x, y = batch
                
                self.memory.add_batch(x, y)
                break
    
    def _validate(self, val_loader: DataLoader) -> float:
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
                
                outputs = self.model(x)
                predictions = outputs.argmax(dim=1)
                
                correct += (predictions == y).sum().item()
                total += y.size(0)
        
        return correct / total if total > 0 else 0.0

class EWCWithReplayStrategy(ContinualStrategy):
    """
    Stratégie combinant EWC et replay.
    La plus efficace pour prévenir l'oubli catastrophique.
    """
    
    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 100.0,
        memory_size: int = 200,
        replay_batch_size: int = 16,
        device: str = 'cuda'
    ):
        super().__init__(model, device)
        self.ewc = ElasticWeightConsolidation(model, lambda_ewc=ewc_lambda)
        self.memory = EpisodicMemory(capacity=memory_size)
        self.replay_batch_size = replay_batch_size
        
    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10
    ) -> Dict[str, float]:
        # Calcul de la Fisher pour EWC
        if task_id > 0:
            self.ewc.compute_fisher_information(train_loader)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        best_acc = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            for batch in train_loader:
                if isinstance(batch, dict):
                    x = batch['frames']
                    y = batch['label']
                else:
                    x, y = batch
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                outputs = self.model(x)
                loss = F.cross_entropy(outputs, y)
                
                # Pénalité EWC
                if task_id > 0:
                    loss += self.ewc.penalty()
                
                # Replay
                if len(self.memory) > 0:
                    mem_x, mem_y = self.memory.sample(self.replay_batch_size)
                    mem_x = mem_x.to(self.device)
                    mem_y = mem_y.to(self.device)
                    
                    mem_outputs = self.model(mem_x)
                    mem_loss = F.cross_entropy(mem_outputs, mem_y)
                    loss += mem_loss * 0.5
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            acc = self._validate(val_loader)
            if acc > best_acc:
                best_acc = acc
        
        # Mise à jour de la mémoire
        self._update_memory(train_loader)
        
        return {'accuracy': best_acc}
    
    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: int
    ) -> torch.Tensor:
        loss = F.cross_entropy(outputs, targets)
        
        if task_id > 0:
            loss += self.ewc.penalty()
        
        if len(self.memory) > 0:
            mem_x, mem_y = self.memory.sample(self.replay_batch_size)
            mem_outputs = self.model(mem_x.to(self.device))
            mem_loss = F.cross_entropy(mem_outputs, mem_y.to(self.device))
            loss += mem_loss * 0.5
        
        return loss
    
    def _update_memory(self, train_loader: DataLoader) -> None:
        self.model.eval()
        
        with torch.no_grad():
            for batch in train_loader:
                if isinstance(batch, dict):
                    x = batch['frames']
                    y = batch['label']
                else:
                    x, y = batch
                
                self.memory.add_batch(x, y)
                break
    
    def _validate(self, val_loader: DataLoader) -> float:
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
                
                outputs = self.model(x)
                predictions = outputs.argmax(dim=1)
                
                correct += (predictions == y).sum().item()
                total += y.size(0)
        
        return correct / total if total > 0 else 0.0

class LwFStrategy(ContinualStrategy):
    """
    Learning without Forgetting (LwF).
    Utilise la distillation pour préserver les connaissances.
    """
    
    def __init__(
        self,
        model: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5,
        device: str = 'cuda'
    ):
        super().__init__(model, device)
        self.temperature = temperature
        self.alpha = alpha
        self.old_model = None
        
    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10
    ) -> Dict[str, float]:
        # Sauvegarde du modèle précédent
        if task_id > 0:
            self.old_model = copy.deepcopy(self.model)
            self.old_model.eval()
            for param in self.old_model.parameters():
                param.requires_grad = False
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        best_acc = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            for batch in train_loader:
                if isinstance(batch, dict):
                    x = batch['frames']
                    y = batch['label']
                else:
                    x, y = batch
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                outputs = self.model(x)
                loss = F.cross_entropy(outputs, y)
                
                # Distillation
                if self.old_model is not None:
                    with torch.no_grad():
                        old_outputs = self.old_model(x)
                    
                    # Distillation loss
                    soft_targets = F.softmax(old_outputs / self.temperature, dim=1)
                    soft_outputs = F.log_softmax(outputs / self.temperature, dim=1)
                    distill_loss = F.kl_div(soft_outputs, soft_targets, reduction='batchmean')
                    
                    loss = self.alpha * loss + (1 - self.alpha) * distill_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            acc = self._validate(val_loader)
            if acc > best_acc:
                best_acc = acc
        
        return {'accuracy': best_acc}
    
    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: int
    ) -> torch.Tensor:
        loss = F.cross_entropy(outputs, targets)
        
        if self.old_model is not None:
            with torch.no_grad():
                old_outputs = self.old_model(outputs)
            
            soft_targets = F.softmax(old_outputs / self.temperature, dim=1)
            soft_outputs = F.log_softmax(outputs / self.temperature, dim=1)
            distill_loss = F.kl_div(soft_outputs, soft_targets, reduction='batchmean')
            
            loss = self.alpha * loss + (1 - self.alpha) * distill_loss
        
        return loss
    
    def _validate(self, val_loader: DataLoader) -> float:
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
                
                outputs = self.model(x)
                predictions = outputs.argmax(dim=1)
                
                correct += (predictions == y).sum().item()
                total += y.size(0)
        
        return correct / total if total > 0 else 0.0

class ProgressiveNetworkStrategy(ContinualStrategy):
    """
    Progressive Neural Networks.
    Ajoute de nouvelles colonnes pour chaque tâche.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        super().__init__(model, device)
        self.task_models = []
        self.adapters = []
        
    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10
    ) -> Dict[str, float]:
        # Création d'un nouveau modèle pour la tâche
        task_model = copy.deepcopy(self.model)
        task_model = task_model.to(self.device)
        
        # Gel des modèles précédents
        for prev_model in self.task_models:
            for param in prev_model.parameters():
                param.requires_grad = False
        
        # Entraînement du nouveau modèle
        optimizer = torch.optim.Adam(task_model.parameters(), lr=0.0001)
        best_acc = 0.0
        
        for epoch in range(epochs):
            task_model.train()
            
            for batch in train_loader:
                if isinstance(batch, dict):
                    x = batch['frames']
                    y = batch['label']
                else:
                    x, y = batch
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Forward pass avec connexions latérales
                outputs = task_model(x)
                
                # Ajout des features des modèles précédents
                for prev_model in self.task_models:
                    with torch.no_grad():
                        prev_outputs = prev_model(x)
                    outputs = outputs + prev_outputs * 0.1
                
                loss = F.cross_entropy(outputs, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            acc = self._validate(task_model, val_loader)
            if acc > best_acc:
                best_acc = acc
        
        # Sauvegarde du modèle de la tâche
        self.task_models.append(task_model)
        
        return {'accuracy': best_acc}
    
    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: int
    ) -> torch.Tensor:
        return F.cross_entropy(outputs, targets)
    
    def _validate(self, model: nn.Module, val_loader: DataLoader) -> float:
        model.eval()
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
                
                outputs = model(x)
                predictions = outputs.argmax(dim=1)
                
                correct += (predictions == y).sum().item()
                total += y.size(0)
        
        return correct / total if total > 0 else 0.0

def create_strategy(
    strategy_type: str,
    model: nn.Module,
    **kwargs
) -> ContinualStrategy:
    """
    Factory pour créer une stratégie d'apprentissage continu.
    """
    strategies = {
        'ewc': EWCStrategy,
        'replay': ReplayStrategy,
        'ewc_replay': EWCWithReplayStrategy,
        'lwf': LwFStrategy,
        'progressive': ProgressiveNetworkStrategy
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Stratégie non supportée: {strategy_type}")
    
    return strategies[strategy_type](model, **kwargs)