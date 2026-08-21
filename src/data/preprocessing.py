"""
Module de prétraitement des vidéos pour la détection de deepfakes.
Fournit des outils pour le nettoyage, l'alignement et la normalisation des visages.

Ce module est crucial car la qualité du prétraitement impacte directement
les performances de détection. Les visages doivent être :
- Correctement détectés et alignés
- Normalisés pour une cohérence des features
- Augmentés pour la robustesse
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
        
        # Initialisation des composants
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
        
        # Filtres optionnels
        self.denoiser = Denoiser(strength=config.denoise_strength) if config.denoise else None
        self.sharpener = Sharpener(strength=config.sharpen_strength) if config.sharpen else None
    
    def preprocess(
        self,
        frames: torch.Tensor,
        face_detector: Optional[object] = None,
        landmarks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Prétraite un clip vidéo de visages.
        
        Args:
            frames: (C, T, H, W) - Clip vidéo
            face_detector: Détecteur de visages optionnel
            landmarks: Landmarks faciaux optionnels
            
        Returns:
            processed: (C, T, H', W') - Clip prétraité
        """
        # Vérification des dimensions
        if frames.dim() != 4:
            raise ValueError(f"Expected 4D tensor (C, T, H, W), got {frames.dim()}D")
        
        C, T, H, W = frames.shape
        
        # Étape 1 : Détection et extraction des visages
        if face_detector is not None:
            frames = face_detector.extract_faces(
                frames,
                margin=self.config.face_margin,
                align=self.config.align_eyes
            )
        
        # Étape 2 : Alignement des visages
        if landmarks is not None:
            frames = self.face_aligner.align_frames(frames, landmarks)
        
        # Étape 3 : Redimensionnement
        if frames.shape[-2:] != self.config.image_size:
            frames = F.interpolate(
                frames,
                size=self.config.image_size,
                mode='bilinear',
                align_corners=False
            )
        
        # Étape 4 : Débruitage optionnel
        if self.denoiser is not None:
            frames = self.denoiser(frames)
        
        # Étape 5 : Netteté optionnelle
        if self.sharpener is not None:
            frames = self.sharpener(frames)
        
        # Étape 6 : Normalisation
        frames = self.frame_normalizer(frames)
        
        return frames
    
    def preprocess_batch(
        self,
        batch: Dict[str, torch.Tensor],
        face_detector: Optional[object] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prétraite un batch complet.
        
        Args:
            batch: Dictionnaire contenant les frames
            face_detector: Détecteur de visages
            
        Returns:
            processed_batch: Batch prétraité
        """
        processed_batch = {}
        
        for key, frames in batch.items():
            if isinstance(frames, torch.Tensor) and frames.dim() == 4:
                processed_batch[key] = self.preprocess(frames, face_detector)
            else:
                processed_batch[key] = frames
        
        return processed_batch

class FaceAligner:
    """
    Alignement des visages basé sur les landmarks.
    Aligne les yeux horizontalement et centre le visage.
    """
    
    def __init__(
        self,
        margin: int = 20,
        align_eyes: bool = True,
        target_eye_distance: float = 60.0
    ):
        self.margin = margin
        self.align_eyes = align_eyes
        self.target_eye_distance = target_eye_distance
        
        # Points de référence pour l'alignement
        self.reference_points = np.array([
            [30.2946, 51.6963],  # Œil gauche
            [65.5318, 51.5014],  # Œil droit
            [48.0252, 71.7366],  # Nez
            [33.5493, 92.3655],  # Bouche gauche
            [62.7299, 92.2041]   # Bouche droite
        ], dtype=np.float32)
    
    def align_face(
        self,
        face: np.ndarray,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Aligne un seul visage.
        
        Args:
            face: (H, W, C) - Image du visage
            landmarks: (5, 2) ou (68, 2) - Landmarks faciaux
            
        Returns:
            aligned: (H, W, C) - Visage aligné
        """
        if landmarks is None or len(landmarks) < 5:
            return face
        
        # Extraction des points clés (yeux, nez, bouche)
        if len(landmarks) == 68:
            # Format 68 points
            key_points = landmarks[[36, 45, 30, 48, 54]]
        else:
            # Format 5 points
            key_points = landmarks[:5]
        
        # Calcul de la transformation affine
        transform_matrix = self._compute_transform_matrix(
            key_points.astype(np.float32),
            self.reference_points
        )
        
        # Application de la transformation
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
        """
        Aligne un clip vidéo complet.
        
        Args:
            frames: (C, T, H, W) - Clip vidéo
            landmarks: (T, 5, 2) ou (T, 68, 2) - Landmarks par frame
            
        Returns:
            aligned_frames: (C, T, H, W) - Clip aligné
        """
        C, T, H, W = frames.shape
        
        # Conversion en numpy
        frames_np = frames.permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
        frames_np = (frames_np * 255).astype(np.uint8)
        
        # Alignement de chaque frame
        aligned_frames = []
        for t in range(T):
            if landmarks is not None and t < len(landmarks):
                aligned = self.align_face(frames_np[t], landmarks[t])
            else:
                aligned = frames_np[t]
            aligned_frames.append(aligned)
        
        # Conversion en tensor
        aligned_frames = np.stack(aligned_frames)  # (T, H, W, C)
        aligned_frames = torch.from_numpy(aligned_frames).permute(3, 0, 1, 2).float() / 255.0
        
        return aligned_frames
    
    def _compute_transform_matrix(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray
    ) -> np.ndarray:
        """
        Calcule la matrice de transformation affine.
        """
        if self.align_eyes:
            # Alignement basé sur les yeux uniquement
            src_eyes = src_points[:2]
            dst_eyes = dst_points[:2]
            
            # Calcul de l'angle de rotation
            src_angle = np.arctan2(
                src_eyes[1, 1] - src_eyes[0, 1],
                src_eyes[1, 0] - src_eyes[0, 0]
            )
            dst_angle = np.arctan2(
                dst_eyes[1, 1] - dst_eyes[0, 1],
                dst_eyes[1, 0] - dst_eyes[0, 0]
            )
            
            angle = np.degrees(dst_angle - src_angle)
            
            # Calcul de l'échelle
            src_dist = np.linalg.norm(src_eyes[1] - src_eyes[0])
            dst_dist = np.linalg.norm(dst_eyes[1] - dst_eyes[0])
            scale = dst_dist / src_dist if src_dist > 0 else 1.0
            
            # Centre des yeux
            src_center = src_eyes.mean(axis=0)
            dst_center = dst_eyes.mean(axis=0)
            
            # Matrice de transformation
            transform = cv2.getRotationMatrix2D(
                tuple(src_center),
                angle,
                scale
            )
            
            # Translation
            transform[0, 2] += dst_center[0] - src_center[0]
            transform[1, 2] += dst_center[1] - src_center[1]
        else:
            # Alignement basé sur tous les points (similarité)
            transform = cv2.estimateAffinePartial2D(
                src_points,
                dst_points
            )[0]
        
        return transform

class FrameNormalizer:
    """
    Normalisation des frames pour la cohérence des features.
    Applique la normalisation par lots et l'égalisation d'histogramme.
    """
    
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
        
        # CLAHE pour l'égalisation d'histogramme
        self.clahe = None
        if use_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=clahe_grid_size
            )
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Normalise un clip vidéo.
        
        Args:
            frames: (C, T, H, W) - Clip vidéo
            
        Returns:
            normalized: (C, T, H, W) - Clip normalisé
        """
        # Étape 1 : Égalisation d'histogramme (CLAHE)
        if self.use_clahe and self.clahe is not None:
            frames = self._apply_clahe(frames)
        
        # Étape 2 : Normalisation par batch
        frames = self._normalize_batch(frames)
        
        # Étape 3 : Normalisation standard
        frames = (frames - self.mean.to(frames.device)) / self.std.to(frames.device)
        
        return frames
    
    def _apply_clahe(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applique CLAHE pour améliorer le contraste local.
        """
        C, T, H, W = frames.shape
        
        # Conversion en numpy
        frames_np = frames.permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
        frames_np = (frames_np * 255).astype(np.uint8)
        
        # Application de CLAHE sur le canal de luminance
        processed = []
        for t in range(T):
            frame = frames_np[t]
            
            # Conversion en LAB
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Application de CLAHE sur le canal L
            l_channel = self.clahe.apply(l_channel)
            
            # Fusion des canaux
            lab = cv2.merge([l_channel, a_channel, b_channel])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            processed.append(frame)
        
        # Conversion en tensor
        processed = np.stack(processed)  # (T, H, W, C)
        processed = torch.from_numpy(processed).permute(3, 0, 1, 2).float() / 255.0
        
        return processed
    
    def _normalize_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Normalise chaque frame individuellement.
        """
        if not self.normalize_illumination:
            return frames
        
        # Calcul des statistiques par frame
        mean = frames.mean(dim=(2, 3), keepdim=True)
        std = frames.std(dim=(2, 3), keepdim=True) + 1e-8
        
        # Normalisation
        normalized = (frames - mean) / std
        
        # Mélange avec l'original pour éviter la sur-normalisation
        alpha = 0.7
        return alpha * normalized + (1 - alpha) * frames

class Denoiser:
    """
    Débruitage des frames pour réduire le bruit de capteur.
    """
    
    def __init__(
        self,
        strength: int = 10,
        method: str = 'fastnlmeans'  # fastnlmeans, bilateral, gaussian
    ):
        self.strength = strength
        self.method = method
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Débruite un clip vidéo.
        """
        C, T, H, W = frames.shape
        
        # Conversion en numpy
        frames_np = frames.permute(1, 2, 3, 0).numpy()
        frames_np = (frames_np * 255).astype(np.uint8)
        
        # Débruitage de chaque frame
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
            else:  # gaussian
                frame = cv2.GaussianBlur(
                    frame,
                    (5, 5),
                    self.strength / 10
                )
            
            denoised.append(frame)
        
        # Conversion en tensor
        denoised = np.stack(denoised)
        denoised = torch.from_numpy(denoised).permute(3, 0, 1, 2).float() / 255.0
        
        return denoised

class Sharpener:
    """
    Amélioration de la netteté des frames.
    """
    
    def __init__(
        self,
        strength: float = 1.0,
        method: str = 'unsharp'  # unsharp, laplacian
    ):
        self.strength = strength
        self.method = method
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Améliore la netteté d'un clip vidéo.
        """
        C, T, H, W = frames.shape
        
        # Conversion en numpy
        frames_np = frames.permute(1, 2, 3, 0).numpy()
        frames_np = (frames_np * 255).astype(np.uint8)
        
        # Application de la netteté
        sharpened = []
        for t in range(T):
            frame = frames_np[t]
            
            if self.method == 'unsharp':
                # Masque flou
                blurred = cv2.GaussianBlur(frame, (0, 0), 3)
                frame = cv2.addWeighted(
                    frame,
                    1 + self.strength,
                    blurred,
                    -self.strength,
                    0
                )
            else:  # laplacian
                # Filtre laplacien
                laplacian = cv2.Laplacian(frame, cv2.CV_64F)
                frame = cv2.addWeighted(
                    frame,
                    1,
                    laplacian.astype(np.uint8),
                    -self.strength,
                    0
                )
            
            sharpened.append(frame)
        
        # Conversion en tensor
        sharpened = np.stack(sharpened)
        sharpened = torch.from_numpy(sharpened).permute(3, 0, 1, 2).float() / 255.0
        
        return sharpened

class TemporalSmoother:
    """
    Lissage temporel pour réduire les variations brusques.
    Utile pour stabiliser les features entre frames consécutives.
    """
    
    def __init__(
        self,
        window_size: int = 5,
        method: str = 'moving_average'  # moving_average, exponential, median
    ):
        self.window_size = window_size
        self.method = method
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Lisse un clip vidéo temporellement.
        
        Args:
            frames: (C, T, H, W)
            
        Returns:
            smoothed: (C, T, H, W)
        """
        if self.method == 'moving_average':
            return self._moving_average(frames)
        elif self.method == 'exponential':
            return self._exponential_smoothing(frames)
        else:  # median
            return self._median_filter(frames)
    
    def _moving_average(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Lissage par moyenne mobile.
        """
        C, T, H, W = frames.shape
        
        # Padding temporel
        pad_size = self.window_size // 2
        padded = F.pad(frames, (0, 0, 0, 0, pad_size, pad_size), mode='replicate')
        
        # Application de la moyenne mobile
        smoothed = []
        for t in range(T):
            window = padded[:, t:t+self.window_size]
            smoothed.append(window.mean(dim=1, keepdim=True))
        
        return torch.cat(smoothed, dim=1)
    
    def _exponential_smoothing(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Lissage exponentiel.
        """
        alpha = 0.3
        smoothed = [frames[:, 0:1]]
        
        for t in range(1, frames.size(1)):
            smoothed.append(
                alpha * frames[:, t:t+1] + (1 - alpha) * smoothed[-1]
            )
        
        return torch.cat(smoothed, dim=1)
    
    def _median_filter(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Filtre médian temporel.
        """
        C, T, H, W = frames.shape
        
        # Padding temporel
        pad_size = self.window_size // 2
        padded = F.pad(frames, (0, 0, 0, 0, pad_size, pad_size), mode='replicate')
        
        # Application du filtre médian
        smoothed = []
        for t in range(T):
            window = padded[:, t:t+self.window_size]
            smoothed.append(window.median(dim=1, keepdim=True)[0])
        
        return torch.cat(smoothed, dim=1)

class VideoPreprocessorPipeline:
    """
    Pipeline complet de prétraitement vidéo.
    Enchaîne toutes les étapes de manière cohérente.
    """
    
    def __init__(
        self,
        config: PreprocessingConfig = PreprocessingConfig(),
        use_temporal_smoothing: bool = False
    ):
        self.config = config
        
        # Initialisation des composants
        self.preprocessor = VideoPreprocessor(config)
        
        # Lissage temporel optionnel
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
        """
        Applique le pipeline complet de prétraitement.
        
        Args:
            frames: (C, T, H, W) - Clip vidéo brut
            face_detector: Détecteur de visages
            landmarks: Landmarks faciaux
            
        Returns:
            processed: (C, T, H', W') - Clip prétraité
        """
        # Prétraitement de base
        processed = self.preprocessor.preprocess(
            frames,
            face_detector,
            landmarks
        )
        
        # Lissage temporel optionnel
        if self.temporal_smoother is not None:
            processed = self.temporal_smoother(processed)
        
        return processed
    
    def preprocess_batch(
        self,
        batch: Dict[str, torch.Tensor],
        face_detector: Optional[object] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prétraite un batch complet.
        """
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
    """
    Factory pour créer un pipeline de prétraitement.
    
    Args:
        image_size: Taille des images
        use_clahe: Utiliser CLAHE
        normalize_illumination: Normaliser l'illumination
        denoise: Débruiter
        sharpen: Améliorer la netteté
        use_temporal_smoothing: Lissage temporel
        
    Returns:
        pipeline: Pipeline de prétraitement configuré
    """
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