"""
Module de prétraitement des vidéos et images pour la détection de deepfakes.
Fournit des outils pour le nettoyage, l'alignement et la normalisation des visages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict, Union, Callable
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration du prétraitement."""
    image_size: Tuple[int, int] = (224, 224)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    face_margin: int = 20
    min_face_size: int = 50
    align_eyes: bool = True
    normalize_illumination: bool = True
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    denoise: bool = False
    denoise_strength: int = 10
    sharpen: bool = False
    sharpen_strength: float = 1.0

class VideoPreprocessor:
    """
    Préprocesseur vidéo complet.
    Gère le pipeline de prétraitement des clips vidéo de visages.
    """
    
    def __init__(
        self,
        config: PreprocessingConfig = PreprocessingConfig()
    ):
        self.config = config
        
        self.face_aligner = FaceAligner(
            margin=config.face_margin,
            align_eyes=config.align_eyes
        )
        
        self.frame_normalizer = FrameNormalizer(
            mean=config.mean,
            std=config.std,
            normalize_illumination=config.normalize_illumination,
            use_clahe=config.use_clahe,
            clahe_clip_limit=config.clahe_clip_limit,
            clahe_grid_size=config.clahe_grid_size
        )
        
        self.denoiser = Denoiser(strength=config.denoise_strength) if config.denoise else None
        self.sharpener = Sharpener(strength=config.sharpen_strength) if config.sharpen else None
    
    def preprocess(
        self,
        frames: torch.Tensor,
        face_detector: Optional[object] = None,
        landmarks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Prétraite un clip vidéo de visages."""
        if frames.dim() != 4:
            raise ValueError(f"Expected 4D tensor (C, T, H, W), got {frames.dim()}D")
        
        C, T, H, W = frames.shape
        
        if face_detector is not None:
            frames = face_detector.extract_faces(
                frames,
                margin=self.config.face_margin,
                align=self.config.align_eyes
            )
        
        if landmarks is not None:
            frames = self.face_aligner.align_frames(frames, landmarks)
        
        if frames.shape[-2:] != self.config.image_size:
            frames = F.interpolate(
                frames,
                size=self.config.image_size,
                mode='bilinear',
                align_corners=False
            )
        
        if self.denoiser is not None:
            frames = self.denoiser(frames)
        
        if self.sharpener is not None:
            frames = self.sharpener(frames)
        
        frames = self.frame_normalizer(frames)
        
        return frames
    
    def preprocess_batch(
        self,
        batch: Dict[str, torch.Tensor],
        face_detector: Optional[object] = None
    ) -> Dict[str, torch.Tensor]:
        """Prétraite un batch complet."""
        processed_batch = {}
        
        for key, frames in batch.items():
            if isinstance(frames, torch.Tensor) and frames.dim() == 4:
                processed_batch[key] = self.preprocess(frames, face_detector)
            else:
                processed_batch[key] = frames
        
        return processed_batch

    def preprocess_image(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image, str],
        face_detector: Optional[object] = None,
        landmarks: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """
        Prétraite une image unique.
        
        Args:
            image: Image en entrée (tensor, numpy, PIL ou chemin)
            face_detector: Détecteur de visages
            landmarks: Landmarks faciaux
            
        Returns:
            processed: (C, H, W) - Image prétraitée
        """
        # Chargement de l'image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if isinstance(image, np.ndarray):
            if image.dtype != np.float32:
                image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        if image.dim() == 3:
            image = image.unsqueeze(1)  # (C, 1, H, W)
        
        # Prétraitement
        processed = self.preprocess(image, face_detector, landmarks)
        
        return processed.squeeze(1)  # (C, H, W)
    
    def preprocess_images(
        self,
        images: List[Union[torch.Tensor, np.ndarray, Image.Image, str]],
        face_detector: Optional[object] = None
    ) -> torch.Tensor:
        """
        Prétraite une liste d'images.
        
        Args:
            images: Liste d'images
            face_detector: Détecteur de visages
            
        Returns:
            processed: (N, C, H, W) - Images prétraitées
        """
        processed_list = []
        
        for img in images:
            processed = self.preprocess_image(img, face_detector)
            processed_list.append(processed)
        
        return torch.stack(processed_list)

class FaceAligner:
    """Alignement des visages basé sur les landmarks."""
    
    def __init__(
        self,
        margin: int = 20,
        align_eyes: bool = True,
        target_eye_distance: float = 60.0
    ):
        self.margin = margin
        self.align_eyes = align_eyes
        self.target_eye_distance = target_eye_distance
        
        self.reference_points = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]
        ], dtype=np.float32)
    
    def align_face(
        self,
        face: np.ndarray,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """Aligne un seul visage."""
        if landmarks is None or len(landmarks) < 5:
            return face
        
        if len(landmarks) == 68:
            key_points = landmarks[[36, 45, 30, 48, 54]]
        else:
            key_points = landmarks[:5]
        
        transform_matrix = self._compute_transform_matrix(
            key_points.astype(np.float32),
            self.reference_points
        )
        
        aligned = cv2.warpAffine(
            face,
            transform_matrix,
            (face.shape[1], face.shape[0]),
            flags=cv2.INTER_CUBIC
        )
        
        return aligned
    
    def align_frames(
        self,
        frames: torch.Tensor,
        landmarks: torch.Tensor
    ) -> torch.Tensor:
        """Aligne un clip vidéo complet."""
        C, T, H, W = frames.shape
        
        frames_np = frames.permute(1, 2, 3, 0).numpy()
        frames_np = (frames_np * 255).astype(np.uint8)
        
        aligned_frames = []
        for t in range(T):
            if landmarks is not None and t < len(landmarks):
                aligned = self.align_face(frames_np[t], landmarks[t])
            else:
                aligned = frames_np[t]
            aligned_frames.append(aligned)
        
        aligned_frames = np.stack(aligned_frames)
        aligned_frames = torch.from_numpy(aligned_frames).permute(3, 0, 1, 2).float() / 255.0
        
        return aligned_frames
    
    def align_image(
        self,
        image: np.ndarray,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Aligne une image unique.
        
        Args:
            image: (H, W, C) - Image
            landmarks: (5, 2) ou (68, 2) - Landmarks faciaux
            
        Returns:
            aligned: (H, W, C) - Image alignée
        """
        return self.align_face(image, landmarks)
    
    def align_batch(
        self,
        images: np.ndarray,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Aligne un batch d'images.
        
        Args:
            images: (N, H, W, C) - Batch d'images
            landmarks: (N, 5, 2) ou (N, 68, 2) - Landmarks par image
            
        Returns:
            aligned: (N, H, W, C) - Images alignées
        """
        aligned_images = []
        for i in range(len(images)):
            aligned = self.align_face(images[i], landmarks[i] if landmarks is not None else None)
            aligned_images.append(aligned)
        
        return np.stack(aligned_images)
    
    def _compute_transform_matrix(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray
    ) -> np.ndarray:
        """Calcule la matrice de transformation affine."""
        if self.align_eyes:
            src_eyes = src_points[:2]
            dst_eyes = dst_points[:2]
            
            src_angle = np.arctan2(
                src_eyes[1, 1] - src_eyes[0, 1],
                src_eyes[1, 0] - src_eyes[0, 0]
            )
            dst_angle = np.arctan2(
                dst_eyes[1, 1] - dst_eyes[0, 1],
                dst_eyes[1, 0] - dst_eyes[0, 0]
            )
            
            angle = np.degrees(dst_angle - src_angle)
            
            src_dist = np.linalg.norm(src_eyes[1] - src_eyes[0])
            dst_dist = np.linalg.norm(dst_eyes[1] - dst_eyes[0])
            scale = dst_dist / src_dist if src_dist > 0 else 1.0
            
            src_center = src_eyes.mean(axis=0)
            dst_center = dst_eyes.mean(axis=0)
            
            transform = cv2.getRotationMatrix2D(
                tuple(src_center),
                angle,
                scale
            )
            
            transform[0, 2] += dst_center[0] - src_center[0]
            transform[1, 2] += dst_center[1] - src_center[1]
        else:
            transform = cv2.estimateAffinePartial2D(
                src_points,
                dst_points
            )[0]
        
        return transform

class FrameNormalizer:
    """Normalisation des frames pour la cohérence des features."""
    
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        normalize_illumination: bool = True,
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8)
    ):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.normalize_illumination = normalize_illumination
        self.use_clahe = use_clahe
        
        self.clahe = None
        if use_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=clahe_grid_size
            )
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Normalise un clip vidéo."""
        if self.use_clahe and self.clahe is not None:
            frames = self._apply_clahe(frames)
        
        frames = self._normalize_batch(frames)
        
        frames = (frames - self.mean.to(frames.device)) / self.std.to(frames.device)
        
        return frames
    
    def normalize_image(
        self,
        image: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Normalise une image unique.
        
        Args:
            image: (H, W, C) ou (C, H, W) - Image
            
        Returns:
            normalized: (C, H, W) - Image normalisée
        """
        if isinstance(image, np.ndarray):
            if image.dtype != np.float32:
                image = image.astype(np.float32) / 255.0
            if image.shape[2] == 3:  # (H, W, C)
                image = torch.from_numpy(image).permute(2, 0, 1)
            else:
                image = torch.from_numpy(image)
        
        if image.dim() == 3:
            image = image.unsqueeze(1)  # (C, 1, H, W)
        
        normalized = self(image)
        
        return normalized.squeeze(1)  # (C, H, W)
    
    def denormalize(
        self,
        frames: torch.Tensor,
        clamp: bool = True
    ) -> torch.Tensor:
        """
        Dénormalise un clip vidéo.
        
        Args:
            frames: (C, T, H, W) - Clip normalisé
            clamp: Clamper les valeurs entre 0 et 1
            
        Returns:
            denormalized: (C, T, H, W) - Clip dénormalisé
        """
        denormalized = frames * self.std.to(frames.device) + self.mean.to(frames.device)
        
        if clamp:
            denormalized = torch.clamp(denormalized, 0, 1)
        
        return denormalized
    
    def _apply_clahe(self, frames: torch.Tensor) -> torch.Tensor:
        """Applique CLAHE pour améliorer le contraste local."""
        C, T, H, W = frames.shape
        
        frames_np = frames.permute(1, 2, 3, 0).numpy()
        frames_np = (frames_np * 255).astype(np.uint8)
        
        processed = []
        for t in range(T):
            frame = frames_np[t]
            
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            l_channel = self.clahe.apply(l_channel)
            
            lab = cv2.merge([l_channel, a_channel, b_channel])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            processed.append(frame)
        
        processed = np.stack(processed)
        processed = torch.from_numpy(processed).permute(3, 0, 1, 2).float() / 255.0
        
        return processed
    
    def _normalize_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """Normalise chaque frame individuellement."""
        if not self.normalize_illumination:
            return frames
        
        mean = frames.mean(dim=(2, 3), keepdim=True)
        std = frames.std(dim=(2, 3), keepdim=True) + 1e-8
        
        normalized = (frames - mean) / std
        
        alpha = 0.7
        return alpha * normalized + (1 - alpha) * frames

class Denoiser:
    """Débruitage des frames pour réduire le bruit de capteur."""
    
    def __init__(
        self,
        strength: int = 10,
        method: str = 'fastnlmeans'
    ):
        self.strength = strength
        self.method = method
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Débruite un clip vidéo."""
        C, T, H, W = frames.shape
        
        frames_np = frames.permute(1, 2, 3, 0).numpy()
        frames_np = (frames_np * 255).astype(np.uint8)
        
        denoised = []
        for t in range(T):
            frame = frames_np[t]
            
            if self.method == 'fastnlmeans':
                frame = cv2.fastNlMeansDenoisingColored(
                    frame,
                    None,
                    self.strength,
                    self.strength,
                    7,
                    21
                )
            elif self.method == 'bilateral':
                frame = cv2.bilateralFilter(
                    frame,
                    -1,
                    self.strength,
                    self.strength
                )
            else:
                frame = cv2.GaussianBlur(
                    frame,
                    (5, 5),
                    self.strength / 10
                )
            
            denoised.append(frame)
        
        denoised = np.stack(denoised)
        denoised = torch.from_numpy(denoised).permute(3, 0, 1, 2).float() / 255.0
        
        return denoised
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Débruite une image unique.
        
        Args:
            image: (H, W, C) - Image
            
        Returns:
            denoised: (H, W, C) - Image débruirée
        """
        if self.method == 'fastnlmeans':
            return cv2.fastNlMeansDenoisingColored(
                image,
                None,
                self.strength,
                self.strength,
                7,
                21
            )
        elif self.method == 'bilateral':
            return cv2.bilateralFilter(
                image,
                -1,
                self.strength,
                self.strength
            )
        else:
            return cv2.GaussianBlur(
                image,
                (5, 5),
                self.strength / 10
            )

class Sharpener:
    """Amélioration de la netteté des frames."""
    
    def __init__(
        self,
        strength: float = 1.0,
        method: str = 'unsharp'
    ):
        self.strength = strength
        self.method = method
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Améliore la netteté d'un clip vidéo."""
        C, T, H, W = frames.shape
        
        frames_np = frames.permute(1, 2, 3, 0).numpy()
        frames_np = (frames_np * 255).astype(np.uint8)
        
        sharpened = []
        for t in range(T):
            frame = frames_np[t]
            
            if self.method == 'unsharp':
                blurred = cv2.GaussianBlur(frame, (0, 0), 3)
                frame = cv2.addWeighted(
                    frame,
                    1 + self.strength,
                    blurred,
                    -self.strength,
                    0
                )
            else:
                laplacian = cv2.Laplacian(frame, cv2.CV_64F)
                frame = cv2.addWeighted(
                    frame,
                    1,
                    laplacian.astype(np.uint8),
                    -self.strength,
                    0
                )
            
            sharpened.append(frame)
        
        sharpened = np.stack(sharpened)
        sharpened = torch.from_numpy(sharpened).permute(3, 0, 1, 2).float() / 255.0
        
        return sharpened
    
    def sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        Améliore la netteté d'une image unique.
        
        Args:
            image: (H, W, C) - Image
            
        Returns:
            sharpened: (H, W, C) - Image affinée
        """
        if self.method == 'unsharp':
            blurred = cv2.GaussianBlur(image, (0, 0), 3)
            return cv2.addWeighted(
                image,
                1 + self.strength,
                blurred,
                -self.strength,
                0
            )
        else:
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            return cv2.addWeighted(
                image,
                1,
                laplacian.astype(np.uint8),
                -self.strength,
                0
            )

class TemporalSmoother:
    """Lissage temporel pour réduire les variations brusques."""
    
    def __init__(
        self,
        window_size: int = 5,
        method: str = 'moving_average'
    ):
        self.window_size = window_size
        self.method = method
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Lisse un clip vidéo temporellement."""
        if self.method == 'moving_average':
            return self._moving_average(frames)
        elif self.method == 'exponential':
            return self._exponential_smoothing(frames)
        else:
            return self._median_filter(frames)
    
    def _moving_average(self, frames: torch.Tensor) -> torch.Tensor:
        C, T, H, W = frames.shape
        
        pad_size = self.window_size // 2
        padded = F.pad(frames, (0, 0, 0, 0, pad_size, pad_size), mode='replicate')
        
        smoothed = []
        for t in range(T):
            window = padded[:, t:t+self.window_size]
            smoothed.append(window.mean(dim=1, keepdim=True))
        
        return torch.cat(smoothed, dim=1)
    
    def _exponential_smoothing(self, frames: torch.Tensor) -> torch.Tensor:
        alpha = 0.3
        smoothed = [frames[:, 0:1]]
        
        for t in range(1, frames.size(1)):
            smoothed.append(
                alpha * frames[:, t:t+1] + (1 - alpha) * smoothed[-1]
            )
        
        return torch.cat(smoothed, dim=1)
    
    def _median_filter(self, frames: torch.Tensor) -> torch.Tensor:
        C, T, H, W = frames.shape
        
        pad_size = self.window_size // 2
        padded = F.pad(frames, (0, 0, 0, 0, pad_size, pad_size), mode='replicate')
        
        smoothed = []
        for t in range(T):
            window = padded[:, t:t+self.window_size]
            smoothed.append(window.median(dim=1, keepdim=True)[0])
        
        return torch.cat(smoothed, dim=1)

class VideoPreprocessorPipeline:
    """Pipeline complet de prétraitement vidéo."""
    
    def __init__(
        self,
        config: PreprocessingConfig = PreprocessingConfig(),
        use_temporal_smoothing: bool = False
    ):
        self.config = config
        self.preprocessor = VideoPreprocessor(config)
        self.temporal_smoother = TemporalSmoother(
            window_size=5,
            method='moving_average'
        ) if use_temporal_smoothing else None
    
    def __call__(
        self,
        frames: torch.Tensor,
        face_detector: Optional[object] = None,
        landmarks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Applique le pipeline complet de prétraitement."""
        processed = self.preprocessor.preprocess(
            frames,
            face_detector,
            landmarks
        )
        
        if self.temporal_smoother is not None:
            processed = self.temporal_smoother(processed)
        
        return processed
    
    def preprocess_image(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image, str],
        face_detector: Optional[object] = None,
        landmarks: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """
        Prétraite une image unique.
        
        Args:
            image: Image en entrée
            face_detector: Détecteur de visages
            landmarks: Landmarks faciaux
            
        Returns:
            processed: (C, H, W) - Image prétraitée
        """
        return self.preprocessor.preprocess_image(image, face_detector, landmarks)
    
    def preprocess_images(
        self,
        images: List[Union[torch.Tensor, np.ndarray, Image.Image, str]],
        face_detector: Optional[object] = None
    ) -> torch.Tensor:
        """
        Prétraite une liste d'images.
        
        Args:
            images: Liste d'images
            face_detector: Détecteur de visages
            
        Returns:
            processed: (N, C, H, W) - Images prétraitées
        """
        return self.preprocessor.preprocess_images(images, face_detector)
    
    def preprocess_batch(
        self,
        batch: Dict[str, torch.Tensor],
        face_detector: Optional[object] = None
    ) -> Dict[str, torch.Tensor]:
        """Prétraite un batch complet."""
        processed_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dim() == 4:
                processed_batch[key] = self(value, face_detector)
            else:
                processed_batch[key] = value
        
        return processed_batch

# Fonctions utilitaires
def create_preprocessor(
    image_size: Tuple[int, int] = (224, 224),
    use_clahe: bool = True,
    normalize_illumination: bool = True,
    denoise: bool = False,
    sharpen: bool = False,
    use_temporal_smoothing: bool = False,
    **kwargs
) -> VideoPreprocessorPipeline:
    """Factory pour créer un pipeline de prétraitement."""
    config = PreprocessingConfig(
        image_size=image_size,
        use_clahe=use_clahe,
        normalize_illumination=normalize_illumination,
        denoise=denoise,
        sharpen=sharpen,
        **kwargs
    )
    
    return VideoPreprocessorPipeline(
        config=config,
        use_temporal_smoothing=use_temporal_smoothing
    )

def preprocess_image_simple(
    image: Union[torch.Tensor, np.ndarray, Image.Image, str],
    image_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """
    Fonction simple pour prétraiter une image rapidement.
    
    Args:
        image: Image en entrée
        image_size: Taille de sortie
        
    Returns:
        processed: (C, H, W) - Image prétraitée
    """
    preprocessor = create_preprocessor(image_size=image_size)
    return preprocessor.preprocess_image(image)