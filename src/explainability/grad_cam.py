"""
Grad-CAM (Gradient-weighted Class Activation Mapping) pour la détection de deepfakes.
Visualise les régions importantes pour la décision du modèle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class GradCAM:
    """
    Implémentation standard de Grad-CAM.
    
    Grad-CAM utilise les gradients de la classe cible par rapport aux
    feature maps de la dernière couche convolutionnelle pour générer
    une carte de chaleur localisant les régions importantes.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        normalize: bool = True
    ):
        self.model = model
        self.target_layer = target_layer
        self.normalize = normalize
        
        # Stockage des activations et gradients
        self.activations = None
        self.gradients = None
        
        # Hooks
        self.forward_handle = None
        self.backward_handle = None
        
        # Trouver la couche cible automatiquement
        if target_layer is None:
            self.target_layer = self._find_target_layer()
        
        if self.target_layer is not None:
            self._register_hooks()
    
    def _find_target_layer(self) -> Optional[nn.Module]:
        """
        Trouve automatiquement la dernière couche convolutionnelle.
        """
        last_conv_layer = None
        
        def find_conv(module):
            nonlocal last_conv_layer
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                last_conv_layer = module
        
        self.model.apply(find_conv)
        
        if last_conv_layer is None:
            logger.warning("Aucune couche convolutionnelle trouvée")
        
        return last_conv_layer
    
    def _register_hooks(self) -> None:
        """
        Enregistre les hooks pour capturer les activations et gradients.
        """
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_backward_hook(backward_hook)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        return_raw: bool = False
    ) -> torch.Tensor:
        """
        Génère la carte Grad-CAM.
        
        Args:
            input_tensor: (1, C, T, H, W) ou (1, C, H, W)
            target_class: Classe cible (None = classe prédite)
            return_raw: Retourner la carte brute
            
        Returns:
            cam: (H, W) - Carte de chaleur normalisée
        """
        if self.target_layer is None:
            raise RuntimeError("Aucune couche cible définie")
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Détermination de la classe cible
        if target_class is None:
            if isinstance(output, tuple):
                output = output[0]
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Vérification des gradients
        if self.gradients is None:
            raise RuntimeError("Gradients non capturés")
        
        # Calcul des poids
        weights = torch.mean(self.gradients, dim=(2, 3, 4) if self.gradients.dim() == 5 else (2, 3))
        
        # Application des poids aux activations
        if self.activations.dim() == 5:  # 3D (vidéo)
            cam = torch.sum(weights.view(1, -1, 1, 1, 1) * self.activations, dim=1)
            # Agrégation temporelle
            cam = cam.mean(dim=1)  # (1, H, W)
        else:  # 2D (image)
            cam = torch.sum(weights.view(1, -1, 1, 1) * self.activations, dim=1)
        
        # ReLU
        cam = F.relu(cam)
        
        # Redimensionnement
        if input_tensor.dim() == 5:
            target_size = input_tensor.shape[-2:]
        else:
            target_size = input_tensor.shape[-2:]
        
        cam = F.interpolate(
            cam.unsqueeze(0),
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)
        
        # Normalisation
        if self.normalize:
            cam = self._normalize_cam(cam)
        
        if return_raw:
            return cam, self.activations, self.gradients
        
        return cam
    
    def _normalize_cam(self, cam: torch.Tensor) -> torch.Tensor:
        """
        Normalise la carte de chaleur entre 0 et 1.
        """
        cam_min = cam.min()
        cam_max = cam.max()
        
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        
        return cam
    
    def remove_hooks(self) -> None:
        """
        Supprime les hooks.
        """
        if self.forward_handle is not None:
            self.forward_handle.remove()
        if self.backward_handle is not None:
            self.backward_handle.remove()
    
    def __del__(self):
        """Destructeur pour nettoyer les hooks."""
        self.remove_hooks()

class GradCAMExplainer:
    """
    Interface haut niveau pour l'explication Grad-CAM.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        self.grad_cam = GradCAM(model, target_layer)
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        overlay: bool = True,
        alpha: float = 0.5
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Génère une explication pour une entrée.
        
        Args:
            input_tensor: Tenseur d'entrée
            target_class: Classe cible
            overlay: Superposer sur l'image
            alpha: Transparence de la superposition
            
        Returns:
            cam ou (cam, overlay)
        """
        input_tensor = input_tensor.to(self.device)
        cam = self.grad_cam.generate(input_tensor, target_class)
        
        if overlay:
            # Conversion en image
            if input_tensor.dim() == 5:
                image = input_tensor[0, :, input_tensor.size(2)//2]  # Frame centrale
            else:
                image = input_tensor[0]
            
            overlay_img = self._overlay(image, cam, alpha)
            return cam, overlay_img
        
        return cam
    
    def _overlay(
        self,
        image: torch.Tensor,
        cam: torch.Tensor,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Superpose la carte de chaleur sur l'image.
        """
        # Normalisation de l'image
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())
        
        # Conversion de la carte en couleur
        cam = cam.cpu().numpy()
        cam_colored = cv2.applyColorMap(
            (cam * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        cam_colored = cam_colored / 255.0
        
        # Superposition
        overlay = alpha * cam_colored + (1 - alpha) * image
        
        return torch.from_numpy(overlay).permute(2, 0, 1)

class TemporalGradCAM(GradCAM):
    """
    Grad-CAM temporel pour les vidéos.
    Analyse l'importance des différentes frames.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        temporal_aggregation: str = 'mean'  # mean, max, weighted
    ):
        super().__init__(model, target_layer)
        self.temporal_aggregation = temporal_aggregation
    
    def generate_temporal(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        return_all_frames: bool = False
    ) -> torch.Tensor:
        """
        Génère la carte Grad-CAM temporelle.
        
        Args:
            input_tensor: (1, C, T, H, W)
            target_class: Classe cible
            return_all_frames: Retourner la carte pour chaque frame
            
        Returns:
            cam: (T, H, W) ou (H, W) selon return_all_frames
        """
        if input_tensor.dim() != 5:
            raise ValueError("Input doit être 5D pour Grad-CAM temporel")
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Calcul des poids
        weights = torch.mean(self.gradients, dim=(2, 3, 4))
        
        # Application des poids
        cam = torch.sum(weights.view(1, -1, 1, 1, 1) * self.activations, dim=1)
        # (1, T, H', W')
        
        # Redimensionnement
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # (T, H, W)
        
        # Normalisation par frame
        for t in range(cam.size(0)):
            cam[t] = self._normalize_cam(cam[t])
        
        if return_all_frames:
            return cam
        
        # Agrégation temporelle
        if self.temporal_aggregation == 'mean':
            cam = cam.mean(dim=0)
        elif self.temporal_aggregation == 'max':
            cam = cam.max(dim=0)[0]
        else:  # weighted
            frame_importance = cam.mean(dim=(1, 2))
            weights = F.softmax(frame_importance, dim=0)
            cam = (cam * weights.view(-1, 1, 1)).sum(dim=0)
        
        return cam
    
    def get_frame_importance(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Calcule l'importance de chaque frame.
        """
        temporal_cam = self.generate_temporal(
            input_tensor,
            target_class,
            return_all_frames=True
        )
        
        # Importance moyenne par frame
        importance = temporal_cam.mean(dim=(1, 2))
        
        return importance