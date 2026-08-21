"""
Elastic Weight Consolidation (EWC) et ses variantes avancées.
Empêche l'oubli catastrophique en pénalisant les changements de paramètres importants.

Variantes implémentées :
- EWC classique (Fisher Information)
- Online EWC (mise à jour continue)
- SI (Synaptic Intelligence)
- MAS (Memory Aware Synapses)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import logging
import copy

logger = logging.getLogger(__name__)

class ElasticWeightConsolidation:
    """
    EWC classique avec Fisher Information Matrix.
    
    La Fisher Information mesure l'importance de chaque paramètre
    pour la tâche précédente. Les paramètres importants sont
    pénalisés s'ils changent trop.
    """
    
    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 100.0,
        fisher_ema_decay: float = 0.9,
        num_samples_fisher: int = 100
    ):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_ema_decay = fisher_ema_decay
        self.num_samples_fisher = num_samples_fisher
        
        # Stockage des informations
        self.fisher_information: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        
    def compute_fisher_information(
        self,
        dataloader: DataLoader,
        loss_fn: Optional[callable] = None
    ) -> None:
        """
        Calcule la Fisher Information Matrix.
        
        Args:
            dataloader: DataLoader pour la tâche courante
            loss_fn: Fonction de perte personnalisée
        """
        self.model.eval()
        
        # Initialisation de la Fisher
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)
        
        # Calcul de la Fisher
        num_samples = 0
        for batch in dataloader:
            if num_samples >= self.num_samples_fisher:
                break
            
            # Extraction des données
            if isinstance(batch, dict):
                inputs = batch['frames']
                targets = batch.get('label', None)
            else:
                inputs, targets = batch
            
            inputs = inputs.to(next(self.model.parameters()).device)
            
            # Forward pass
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            # Calcul de la perte
            if loss_fn is not None:
                loss = loss_fn(outputs, targets)
            elif targets is not None:
                targets = targets.to(next(self.model.parameters()).device)
                loss = F.cross_entropy(outputs, targets)
            else:
                # Pour l'apprentissage auto-supervisé
                loss = outputs.abs().sum()
            
            # Backward pass
            loss.backward()
            
            # Accumulation de la Fisher
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
            
            num_samples += inputs.size(0)
        
        # Normalisation
        for name in fisher:
            fisher[name] /= max(num_samples, 1)
            
            # Mise à jour EMA si la Fisher existe déjà
            if name in self.fisher_information:
                fisher[name] = (
                    self.fisher_ema_decay * self.fisher_information[name] +
                    (1 - self.fisher_ema_decay) * fisher[name]
                )
        
        # Sauvegarde
        self.fisher_information = fisher
        self.optimal_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        logger.info(f"Fisher Information calculée sur {num_samples} échantillons")
    
    def penalty(self) -> torch.Tensor:
        """
        Calcule la pénalité EWC.
        
        Returns:
            penalty: Pénalité à ajouter à la perte
        """
        if not self.fisher_information:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_information and name in self.optimal_params:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                
                # Pénalité quadratique
                penalty += (fisher * (param - optimal) ** 2).sum()
        
        return self.lambda_ewc * penalty
    
    def save(self, path: str) -> None:
        """Sauvegarde l'état EWC."""
        torch.save({
            'fisher_information': self.fisher_information,
            'optimal_params': self.optimal_params,
            'lambda_ewc': self.lambda_ewc
        }, path)
    
    def load(self, path: str) -> None:
        """Charge l'état EWC."""
        checkpoint = torch.load(path)
        self.fisher_information = checkpoint['fisher_information']
        self.optimal_params = checkpoint['optimal_params']
        self.lambda_ewc = checkpoint['lambda_ewc']

class OnlineEWC(ElasticWeightConsolidation):
    """
    Online EWC avec mise à jour continue de la Fisher.
    Adapté pour l'apprentissage continu en ligne.
    """
    
    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 100.0,
        gamma: float = 0.9  # Facteur de décroissance
    ):
        super().__init__(model, lambda_ewc)
        self.gamma = gamma
        
    def update_fisher(
        self,
        batch: Dict[str, torch.Tensor],
        loss_fn: Optional[callable] = None
    ) -> None:
        """
        Met à jour la Fisher Information en ligne.
        """
        self.model.train()
        
        # Extraction des données
        if isinstance(batch, dict):
            inputs = batch['frames']
            targets = batch.get('label', None)
        else:
            inputs, targets = batch
        
        inputs = inputs.to(next(self.model.parameters()).device)
        
        # Forward pass
        self.model.zero_grad()
        outputs = self.model(inputs)
        
        if loss_fn is not None:
            loss = loss_fn(outputs, targets)
        elif targets is not None:
            targets = targets.to(next(self.model.parameters()).device)
            loss = F.cross_entropy(outputs, targets)
        else:
            loss = outputs.abs().sum()
        
        # Backward pass
        loss.backward()
        
        # Mise à jour de la Fisher
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in self.fisher_information:
                    self.fisher_information[name] = torch.zeros_like(param)
                    self.optimal_params[name] = param.data.clone()
                
                # Mise à jour avec décroissance
                self.fisher_information[name] = (
                    self.gamma * self.fisher_information[name] +
                    param.grad.data ** 2
                )
    
    def penalty(self) -> torch.Tensor:
        """
        Calcule la pénalité Online EWC.
        """
        if not self.fisher_information:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                penalty += (fisher * (param - optimal) ** 2).sum()
        
        return self.lambda_ewc * penalty

class SIEWC(ElasticWeightConsolidation):
    """
    Synaptic Intelligence (SI).
    Mesure l'importance des paramètres basée sur leur contribution
    à la diminution de la perte pendant l'entraînement.
    """
    
    def __init__(
        self,
        model: nn.Module,
        lambda_si: float = 1.0,
        xi: float = 0.1  # Facteur d'amortissement
    ):
        super().__init__(model, lambda_si)
        self.xi = xi
        
        # Suivi des paramètres
        self.previous_params = {}
        self.importance = {}
        self.omega = {}
        
    def update_importance(
        self,
        loss: torch.Tensor
    ) -> None:
        """
        Met à jour l'importance des paramètres.
        """
        # Calcul des gradients
        grads = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads[name] = param.grad.data.clone()
                
                if name not in self.previous_params:
                    self.previous_params[name] = param.data.clone()
                    self.importance[name] = torch.zeros_like(param)
                    self.omega[name] = torch.zeros_like(param)
                
                # Accumulation de l'importance
                delta = param.data - self.previous_params[name]
                self.importance[name] += grads[name] * delta
                self.previous_params[name] = param.data.clone()
    
    def consolidate(self) -> None:
        """
        Consolide l'importance après l'entraînement.
        """
        for name in self.importance:
            # Calcul de l'oméga
            delta = self.model.state_dict()[name] - self.previous_params[name]
            self.omega[name] += self.importance[name] / (delta ** 2 + self.xi)
            
            # Réinitialisation pour la prochaine tâche
            self.importance[name] = torch.zeros_like(self.importance[name])
    
    def penalty(self) -> torch.Tensor:
        """
        Calcule la pénalité SI.
        """
        if not self.omega:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if name in self.omega:
                omega = self.omega[name]
                optimal = self.optimal_params.get(name, param.data.clone())
                penalty += (omega * (param - optimal) ** 2).sum()
        
        return self.lambda_ewc * penalty

class MASEWC(ElasticWeightConsolidation):
    """
    Memory Aware Synapses (MAS).
    Mesure l'importance basée sur la sensibilité de la sortie
    aux changements de paramètres.
    """
    
    def __init__(
        self,
        model: nn.Module,
        lambda_mas: float = 1.0
    ):
        super().__init__(model, lambda_mas)
        
        self.importance = {}
        
    def compute_importance(
        self,
        dataloader: DataLoader
    ) -> None:
        """
        Calcule l'importance des paramètres basée sur MAS.
        """
        self.model.eval()
        
        # Initialisation
        importance = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                importance[name] = torch.zeros_like(param)
        
        # Calcul de l'importance
        num_samples = 0
        for batch in dataloader:
            if num_samples >= 100:
                break
            
            if isinstance(batch, dict):
                inputs = batch['frames']
            else:
                inputs = batch
            
            inputs = inputs.to(next(self.model.parameters()).device)
            inputs.requires_grad = True
            
            # Forward pass
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            # Sensibilité de la sortie
            outputs_norm = outputs.norm(dim=-1).sum()
            outputs_norm.backward()
            
            # Accumulation de l'importance
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    importance[name] += param.grad.data.abs()
            
            num_samples += inputs.size(0)
        
        # Normalisation
        for name in importance:
            importance[name] /= max(num_samples, 1)
        
        self.importance = importance
        self.optimal_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
    
    def penalty(self) -> torch.Tensor:
        """
        Calcule la pénalité MAS.
        """
        if not self.importance:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if name in self.importance:
                importance = self.importance[name]
                optimal = self.optimal_params[name]
                penalty += (importance * (param - optimal) ** 2).sum()
        
        return self.lambda_ewc * penalty

def create_ewc(
    method: str = 'ewc',
    model: nn.Module = None,
    **kwargs
) -> ElasticWeightConsolidation:
    """
    Factory pour créer une méthode EWC.
    """
    methods = {
        'ewc': ElasticWeightConsolidation,
        'online_ewc': OnlineEWC,
        'si': SIEWC,
        'mas': MASEWC
    }
    
    if method not in methods:
        raise ValueError(f"Méthode EWC non supportée: {method}")
    
    if model is None:
        raise ValueError("Modèle requis pour créer EWC")
    
    return methods[method](model, **kwargs)