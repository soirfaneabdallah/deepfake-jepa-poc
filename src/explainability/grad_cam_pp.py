"""
Grad-CAM++ : Version améliorée de Grad-CAM.
Utilise une pondération des gradients plus sophistiquée.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
import logging

from .grad_cam import GradCAM

logger = logging.getLogger(__name__)

class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ avec pondération explicite des gradients.
    
    Améliorations par rapport à Grad-CAM standard :
    - Meilleure localisation des objets multiples
    - Pondération des gradients par leur importance relative
    - Plus robuste aux activations négatives
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        normalize: bool = True
    ):
        super().__init__(model, target_layer, normalize)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        return_raw: bool = False
    ) -> torch.Tensor:
        """
        Génère la carte Grad-CAM++.
        """
        if self.target_layer is None:
            raise RuntimeError("Aucune couche cible définie")
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Calcul des poids Grad-CAM++
        weights = self._compute_weights()
        
        # Application des poids
        if self.activations.dim() == 5:
            cam = torch.sum(weights.view(1, -1, 1, 1, 1) * self.activations, dim=1)
            cam = cam.mean(dim=1)
        else:
            cam = torch.sum(weights.view(1, -1, 1, 1) * self.activations, dim=1)
        
        # ReLU
        cam = F.relu(cam)
        
        # Redimensionnement
        cam = F.interpolate(
            cam.unsqueeze(0),
            size=input_tensor.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)
        
        # Normalisation
        if self.normalize:
            cam = self._normalize_cam(cam)
        
        if return_raw:
            return cam, self.activations, self.gradients
        
        return cam
    
    def _compute_weights(self) -> torch.Tensor:
        """
        Calcule les poids Grad-CAM++.
        
        Formule : w_k = sum_i sum_j alpha_ij^k * ReLU(grad_ij^k)
        où alpha_ij^k est calculé à partir des gradients de second ordre
        """
        gradients = self.gradients
        activations = self.activations
        
        # Aplatissement spatial
        if gradients.dim() == 5:
            # (1, C, T, H, W) -> (1, C, T*H*W)
            gradients_flat = gradients.view(1, gradients.size(1), -1)
            activations_flat = activations.view(1, activations.size(1), -1)
        else:
            # (1, C, H, W) -> (1, C, H*W)
            gradients_flat = gradients.view(1, gradients.size(1), -1)
            activations_flat = activations.view(1, activations.size(1), -1)
        
        # Calcul des coefficients alpha
        grad_power_2 = gradients_flat ** 2
        grad_power_3 = grad_power_2 * gradients_flat
        
        # Somme sur les positions spatiales
        sum_grad_2 = grad_power_2.sum(dim=2, keepdim=True)
        sum_grad_3 = grad_power_3.sum(dim=2, keepdim=True)
        sum_activations = activations_flat.sum(dim=2, keepdim=True)
        
        # Éviter la division par zéro
        epsilon = 1e-8
        
        # Calcul des alpha
        alpha_numerator = sum_grad_2
        alpha_denominator = 2 * sum_grad_2 + sum_activations * sum_grad_3 + epsilon
        alpha = alpha_numerator / alpha_denominator
        
        # Application des alpha aux gradients
        weighted_gradients = alpha * F.relu(gradients_flat)
        
        # Somme sur les positions spatiales pour obtenir les poids
        weights = weighted_gradients.sum(dim=2)
        
        return weights.squeeze(0)

class GradCAMPlusPlusExplainer:
    """
    Interface haut niveau pour Grad-CAM++.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        self.grad_cam_pp = GradCAMPlusPlus(model, target_layer)
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        compare_with_gradcam: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Génère une explication Grad-CAM++.
        
        Args:
            input_tensor: Tenseur d'entrée
            target_class: Classe cible
            compare_with_gradcam: Comparer avec Grad-CAM standard
            
        Returns:
            Dict contenant les cartes d'explication
        """
        input_tensor = input_tensor.to(self.device)
        
        # Grad-CAM++
        cam_pp = self.grad_cam_pp.generate(input_tensor, target_class)
        
        results = {'grad_cam_pp': cam_pp}
        
        # Comparaison avec Grad-CAM standard
        if compare_with_gradcam:
            grad_cam = GradCAM(self.model, self.grad_cam_pp.target_layer)
            cam_standard = grad_cam.generate(input_tensor, target_class)
            results['grad_cam'] = cam_standard
        
        return results