"""
Cartes de saillance pour l'explicabilité des modèles.
Implémente plusieurs méthodes basées sur les gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class SaliencyMap:
    """
    Carte de saillance standard basée sur les gradients.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Génère la carte de saillance.
        
        Args:
            input_tensor: (1, C, T, H, W) ou (1, C, H, W)
            target_class: Classe cible
            normalize: Normaliser la carte
            
        Returns:
            saliency: Carte de saillance
        """
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Carte de saillance = valeur absolue des gradients
        saliency = input_tensor.grad.abs()
        
        # Agrégation sur les canaux
        saliency = saliency.max(dim=1, keepdim=True)[0]
        
        # Agrégation temporelle si vidéo
        if saliency.dim() == 5:
            saliency = saliency.mean(dim=2)
        
        saliency = saliency.squeeze(0).squeeze(0)
        
        # Normalisation
        if normalize:
            saliency = self._normalize(saliency)
        
        return saliency
    
    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalise entre 0 et 1."""
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        if tensor_max - tensor_min > 1e-8:
            tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        
        return tensor

class IntegratedGradients:
    """
    Integrated Gradients pour l'attribution des features.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 50
    ):
        self.model = model
        self.num_steps = num_steps
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Génère les attributions Integrated Gradients.
        
        Args:
            input_tensor: (1, C, T, H, W) ou (1, C, H, W)
            target_class: Classe cible
            baseline: Baseline (par défaut : zéros)
            
        Returns:
            attributions: Carte d'attribution
        """
        # Baseline
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        # Forward pass pour déterminer la classe
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Génération des interpolations
        alphas = torch.linspace(0, 1, self.num_steps).to(input_tensor.device)
        
        # Accumulation des gradients
        integrated_gradients = torch.zeros_like(input_tensor)
        
        for alpha in alphas:
            # Interpolation
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated = interpolated.clone().requires_grad_(True)
            
            # Forward pass
            output = self.model(interpolated)
            
            # Backward pass
            self.model.zero_grad()
            output[0, target_class].backward()
            
            # Accumulation
            integrated_gradients += interpolated.grad
        
        # Moyenne
        integrated_gradients /= self.num_steps
        
        # Multiplication par la différence
        attributions = integrated_gradients * (input_tensor - baseline)
        
        # Agrégation
        attributions = attributions.abs().max(dim=1, keepdim=True)[0]
        
        if attributions.dim() == 5:
            attributions = attributions.mean(dim=2)
        
        attributions = attributions.squeeze(0).squeeze(0)
        
        # Normalisation
        attributions = self._normalize(attributions)
        
        return attributions
    
    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalise entre 0 et 1."""
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        if tensor_max - tensor_min > 1e-8:
            tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        
        return tensor

class SmoothGrad:
    """
    SmoothGrad pour réduire le bruit dans les cartes de saillance.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 50,
        noise_level: float = 0.1
    ):
        self.model = model
        self.num_samples = num_samples
        self.noise_level = noise_level
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Génère la carte SmoothGrad.
        """
        # Forward pass initial
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Accumulation des cartes de saillance bruitées
        saliency_maps = []
        
        for _ in range(self.num_samples):
            # Ajout de bruit
            noise = torch.randn_like(input_tensor) * self.noise_level
            noisy_input = input_tensor + noise
            noisy_input = noisy_input.clone().requires_grad_(True)
            
            # Forward pass
            output = self.model(noisy_input)
            
            # Backward pass
            self.model.zero_grad()
            output[0, target_class].backward()
            
            # Carte de saillance
            saliency = noisy_input.grad.abs().max(dim=1, keepdim=True)[0]
            
            if saliency.dim() == 5:
                saliency = saliency.mean(dim=2)
            
            saliency_maps.append(saliency.squeeze(0).squeeze(0))
        
        # Moyenne des cartes
        smooth_saliency = torch.stack(saliency_maps).mean(dim=0)
        
        # Normalisation
        smooth_saliency = self._normalize(smooth_saliency)
        
        return smooth_saliency
    
    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalise entre 0 et 1."""
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        if tensor_max - tensor_min > 1e-8:
            tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        
        return tensor

class GuidedBackpropagation:
    """
    Guided Backpropagation pour des cartes de saillance plus nettes.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """
        Enregistre les hooks pour guider la rétropropagation.
        """
        def relu_hook(module, grad_input, grad_output):
            # Masquer les gradients négatifs
            if isinstance(grad_input, tuple):
                grad_input = list(grad_input)
                for i in range(len(grad_input)):
                    if grad_input[i] is not None:
                        grad_input[i] = F.relu(grad_input[i])
                grad_input = tuple(grad_input)
            return grad_input
        
        def forward_hook(module, input, output):
            # Masquer les activations négatives
            return F.relu(output)
        
        # Enregistrement des hooks sur les ReLU
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                self.hooks.append(
                    module.register_forward_hook(forward_hook)
                )
                self.hooks.append(
                    module.register_backward_hook(relu_hook)
                )
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Génère la carte Guided Backpropagation.
        """
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Carte de saillance
        guided_saliency = input_tensor.grad.abs().max(dim=1, keepdim=True)[0]
        
        if guided_saliency.dim() == 5:
            guided_saliency = guided_saliency.mean(dim=2)
        
        guided_saliency = guided_saliency.squeeze(0).squeeze(0)
        
        # Normalisation
        guided_saliency = self._normalize(guided_saliency)
        
        return guided_saliency
    
    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalise entre 0 et 1."""
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        if tensor_max - tensor_min > 1e-8:
            tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        
        return tensor
    
    def remove_hooks(self) -> None:
        """Supprime les hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __del__(self):
        """Destructeur."""
        self.remove_hooks()