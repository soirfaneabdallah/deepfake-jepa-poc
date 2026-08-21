"""
Augmentations spatio-temporelles pour v-JEPA.
Optimisées pour l'apprentissage auto-supervisé sur vidéos.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import random
from typing import Tuple, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

class SpatialAugmentation:
    """
    Augmentations spatiales pour les frames vidéo.
    Appliquées indépendamment à chaque frame.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        crop_scale: Tuple[float, float] = (0.3, 1.0),
        crop_ratio: Tuple[float, float] = (0.75, 1.33),
        horizontal_flip_prob: float = 0.5,
        color_jitter_strength: float = 0.4,
        gaussian_blur_prob: float = 0.5,
        solarization_prob: float = 0.2
    ):
        self.image_size = image_size
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.horizontal_flip_prob = horizontal_flip_prob
        self.color_jitter_strength = color_jitter_strength
        self.gaussian_blur_prob = gaussian_blur_prob
        self.solarization_prob = solarization_prob
        
        # Transformations
        self.color_jitter = T.ColorJitter(
            brightness=0.8 * color_jitter_strength,
            contrast=0.8 * color_jitter_strength,
            saturation=0.8 * color_jitter_strength,
            hue=0.2 * color_jitter_strength
        )
        
        self.gaussian_blur = T.GaussianBlur(
            kernel_size=23,
            sigma=(0.1, 2.0)
        )
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applique les augmentations spatiales.
        
        Args:
            frames: (C, T, H, W)
            
        Returns:
            augmented: (C, T, H, W)
        """
        C, T, H, W = frames.shape
        
        # Reshape pour traitement par frame
        frames = frames.permute(1, 0, 2, 3)  # (T, C, H, W)
        
        # Random resized crop
        frames = self._random_resized_crop(frames)
        
        # Flip horizontal
        if random.random() < self.horizontal_flip_prob:
            frames = torch.flip(frames, dims=[-1])
        
        # Color jitter
        if random.random() < 0.8:
            frames = self.color_jitter(frames)
        
        # Gaussian blur
        if random.random() < self.gaussian_blur_prob:
            frames = self.gaussian_blur(frames)
        
        # Solarization
        if random.random() < self.solarization_prob:
            frames = self._solarize(frames)
        
        # Normalisation
        frames = self._normalize(frames)
        
        # Reshape retour
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        
        return frames
    
    def _random_resized_crop(
        self,
        frames: torch.Tensor
    ) -> torch.Tensor:
        """
        Applique un crop aléatoire redimensionné.
        """
        T, C, H, W = frames.shape
        
        # Paramètres du crop
        scale = random.uniform(*self.crop_scale)
        ratio = random.uniform(*self.crop_ratio)
        
        # Calcul des dimensions du crop
        crop_h = int(H * scale)
        crop_w = int(crop_h * ratio)
        
        # Position du crop
        top = random.randint(0, max(0, H - crop_h))
        left = random.randint(0, max(0, W - crop_w))
        
        # Application du crop
        frames = frames[:, :, top:top+crop_h, left:left+crop_w]
        
        # Redimensionnement
        frames = F.interpolate(
            frames,
            size=self.image_size,
            mode='bilinear',
            align_corners=False
        )
        
        return frames
    
    def _solarize(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applique la solarisation.
        """
        threshold = random.uniform(0.5, 1.0)
        return torch.where(frames < threshold, frames, 1 - frames)
    
    def _normalize(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Normalise les frames.
        """
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        return (frames - mean) / std

class TemporalAugmentation:
    """
    Augmentations temporelles pour les clips vidéo.
    """
    
    def __init__(
        self,
        frame_dropout_prob: float = 0.1,
        temporal_crop_prob: float = 0.5,
        reverse_playback_prob: float = 0.1,
        speed_perturbation_range: Tuple[float, float] = (0.5, 2.0)
    ):
        self.frame_dropout_prob = frame_dropout_prob
        self.temporal_crop_prob = temporal_crop_prob
        self.reverse_playback_prob = reverse_playback_prob
        self.speed_perturbation_range = speed_perturbation_range
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applique les augmentations temporelles.
        
        Args:
            frames: (C, T, H, W)
            
        Returns:
            augmented: (C, T, H, W)
        """
        # Frame dropout
        if random.random() < self.frame_dropout_prob:
            frames = self._frame_dropout(frames)
        
        # Temporal crop
        if random.random() < self.temporal_crop_prob:
            frames = self._temporal_crop(frames)
        
        # Reverse playback
        if random.random() < self.reverse_playback_prob:
            frames = torch.flip(frames, dims=[1])
        
        # Speed perturbation
        if random.random() < 0.3:
            frames = self._speed_perturbation(frames)
        
        return frames
    
    def _frame_dropout(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Supprime aléatoirement des frames.
        """
        C, T, H, W = frames.shape
        
        # Masque de dropout
        keep_prob = 1 - self.frame_dropout_prob
        mask = torch.rand(T) < keep_prob
        
        # Garder au moins une frame
        if not mask.any():
            mask[0] = True
        
        # Application du masque
        frames = frames[:, mask]
        
        # Padding si nécessaire
        if frames.size(1) < T:
            pad_size = T - frames.size(1)
            frames = F.pad(frames, (0, 0, 0, 0, 0, pad_size), mode='replicate')
        
        return frames
    
    def _temporal_crop(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Crop temporel aléatoire.
        """
        C, T, H, W = frames.shape
        
        # Taille du crop
        crop_size = random.randint(T // 2, T)
        
        # Position du crop
        start = random.randint(0, T - crop_size)
        
        # Application du crop
        frames = frames[:, start:start+crop_size]
        
        # Redimensionnement temporel
        frames = F.interpolate(
            frames.permute(1, 0, 2, 3),
            size=T,
            mode='nearest'
        ).permute(1, 0, 2, 3)
        
        return frames
    
    def _speed_perturbation(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Perturbation de la vitesse de lecture.
        """
        C, T, H, W = frames.shape
        
        # Facteur de vitesse
        speed = random.uniform(*self.speed_perturbation_range)
        
        # Nouveaux indices
        indices = torch.linspace(0, T - 1, int(T * speed)).long()
        indices = torch.clamp(indices, 0, T - 1)
        
        # Rééchantillonnage
        frames = frames[:, indices]
        
        # Redimensionnement si nécessaire
        if frames.size(1) != T:
            frames = F.interpolate(
                frames.permute(1, 0, 2, 3),
                size=T,
                mode='nearest'
            ).permute(1, 0, 2, 3)
        
        return frames

class VJEPAAugmentation:
    """
    Augmentations complètes pour v-JEPA.
    Combine les augmentations spatiales et temporelles.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        crop_scale: Tuple[float, float] = (0.4, 1.0),
        color_jitter_strength: float = 0.5,
        frame_dropout_prob: float = 0.1
    ):
        self.spatial_aug = SpatialAugmentation(
            image_size=image_size,
            crop_scale=crop_scale,
            color_jitter_strength=color_jitter_strength
        )
        
        self.temporal_aug = TemporalAugmentation(
            frame_dropout_prob=frame_dropout_prob
        )
    
    def __call__(
        self,
        frames: torch.Tensor,
        apply_temporal: bool = True
    ) -> torch.Tensor:
        """
        Applique les augmentations complètes.
        """
        # Augmentations spatiales
        frames = self.spatial_aug(frames)
        
        # Augmentations temporelles
        if apply_temporal:
            frames = self.temporal_aug(frames)
        
        return frames
    
    def create_views(
        self,
        frames: torch.Tensor,
        num_views: int = 2
    ) -> List[torch.Tensor]:
        """
        Crée plusieurs vues augmentées du même clip.
        """
        return [self(frames) for _ in range(num_views)]

def create_vjepa_augmentations(config: Dict) -> VJEPAAugmentation:
    """
    Crée les augmentations v-JEPA à partir de la configuration.
    """
    aug_config = config['vjepa']['augmentations']
    
    return VJEPAAugmentation(
        image_size=tuple(config['data']['preprocessing']['image_size']),
        crop_scale=tuple(aug_config['spatial']['random_resized_crop']['scale']),
        color_jitter_strength=aug_config['spatial']['color_jitter']['brightness'],
        frame_dropout_prob=aug_config['temporal']['frame_dropout']
    )

# Augmentations spécifiques pour les vrais visages
class RealFaceAugmentation(VJEPAAugmentation):
    """
    Augmentations spécifiques pour les visages authentiques.
    Plus conservatives pour préserver les caractéristiques naturelles.
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            crop_scale=(0.5, 1.0),  # Crops plus larges
            color_jitter_strength=0.3,  # Moins de perturbation couleur
            frame_dropout_prob=0.05,  # Moins de dropout temporel
            **kwargs
        )

# Augmentations pour les tests (pas d'augmentation)
class TestAugmentation:
    """
    Pas d'augmentation pour l'évaluation.
    Juste le redimensionnement et la normalisation.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        C, T, H, W = frames.shape
        
        # Reshape pour traitement par frame
        frames = frames.permute(1, 0, 2, 3)
        
        # Redimensionnement
        frames = F.interpolate(
            frames,
            size=self.image_size,
            mode='bilinear',
            align_corners=False
        )
        
        # Normalisation
        frames = self.normalize(frames)
        
        # Reshape retour
        frames = frames.permute(1, 0, 2, 3)
        
        return frames