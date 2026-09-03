"""
Module de prétraitement des vidéos et images pour la détection de deepfakes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict, Union, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, random_split, Subset
import random

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
    def __init__(self, config: PreprocessingConfig = PreprocessingConfig()):
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

    def preprocess(self, frames: torch.Tensor, face_detector: Optional[object] = None,
                   landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        if frames.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {frames.dim()}D")
        
        if face_detector is not None:
            frames = face_detector.extract_faces(frames, margin=self.config.face_margin, align=self.config.align_eyes)
        
        if landmarks is not None:
            frames = self.face_aligner.align_frames(frames, landmarks)
        
        if frames.shape[-2:] != self.config.image_size:
            frames = F.interpolate(frames, size=self.config.image_size, mode='bilinear', align_corners=False)
        
        if self.denoiser is not None:
            frames = self.denoiser(frames)
        
        if self.sharpener is not None:
            frames = self.sharpener(frames)
        
        frames = self.frame_normalizer(frames)
        return frames

    def preprocess_image(self, image: Union[torch.Tensor, np.ndarray, Image.Image, str],
                         face_detector: Optional[object] = None,
                         landmarks: Optional[np.ndarray] = None) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(image, np.ndarray):
            if image.dtype != np.float32:
                image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
        if image.dim() == 3:
            image = image.unsqueeze(1)
        processed = self.preprocess(image, face_detector, landmarks)
        return processed.squeeze(1)

    def preprocess_images(self, images: List[Union[torch.Tensor, np.ndarray, Image.Image, str]],
                          face_detector: Optional[object] = None) -> torch.Tensor:
        processed_list = []
        for img in images:
            processed = self.preprocess_image(img, face_detector)
            processed_list.append(processed)
        return torch.stack(processed_list)

    def preprocess_batch(self, batch: Dict[str, torch.Tensor],
                         face_detector: Optional[object] = None) -> Dict[str, torch.Tensor]:
        processed_batch = {}
        for key, frames in batch.items():
            if isinstance(frames, torch.Tensor) and frames.dim() == 4:
                processed_batch[key] = self.preprocess(frames, face_detector)
            else:
                processed_batch[key] = frames
        return processed_batch

    def normalize_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Normalise un batch avec les statistiques ImageNet."""
        return (batch - torch.tensor(self.config.mean).view(1, 3, 1, 1).to(batch.device)) / \
               torch.tensor(self.config.std).view(1, 3, 1, 1).to(batch.device)

    def denormalize_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Dénormalise un batch."""
        mean = torch.tensor(self.config.mean).view(1, 3, 1, 1).to(batch.device)
        std = torch.tensor(self.config.std).view(1, 3, 1, 1).to(batch.device)
        return batch * std + mean


class FaceAligner:
    def __init__(self, margin: int = 20, align_eyes: bool = True, target_eye_distance: float = 60.0):
        self.margin = margin
        self.align_eyes = align_eyes
        self.target_eye_distance = target_eye_distance
        self.reference_points = np.array([
            [30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
            [33.5493, 92.3655], [62.7299, 92.2041]
        ], dtype=np.float32)

    def align_face(self, face: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        if landmarks is None or len(landmarks) < 5:
            return face
        if len(landmarks) == 68:
            key_points = landmarks[[36, 45, 30, 48, 54]]
        else:
            key_points = landmarks[:5]
        transform_matrix = self._compute_transform_matrix(key_points.astype(np.float32), self.reference_points)
        aligned = cv2.warpAffine(face, transform_matrix, (face.shape[1], face.shape[0]), flags=cv2.INTER_CUBIC)
        return aligned

    def align_frames(self, frames: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
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

    def _compute_transform_matrix(self, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
        if self.align_eyes:
            src_eyes = src_points[:2]
            dst_eyes = dst_points[:2]
            src_angle = np.arctan2(src_eyes[1, 1] - src_eyes[0, 1], src_eyes[1, 0] - src_eyes[0, 0])
            dst_angle = np.arctan2(dst_eyes[1, 1] - dst_eyes[0, 1], dst_eyes[1, 0] - dst_eyes[0, 0])
            angle = np.degrees(dst_angle - src_angle)
            src_dist = np.linalg.norm(src_eyes[1] - src_eyes[0])
            dst_dist = np.linalg.norm(dst_eyes[1] - dst_eyes[0])
            scale = dst_dist / src_dist if src_dist > 0 else 1.0
            src_center = src_eyes.mean(axis=0)
            dst_center = dst_eyes.mean(axis=0)
            transform = cv2.getRotationMatrix2D(tuple(src_center), angle, scale)
            transform[0, 2] += dst_center[0] - src_center[0]
            transform[1, 2] += dst_center[1] - src_center[1]
        else:
            transform = cv2.estimateAffinePartial2D(src_points, dst_points)[0]
        return transform


class FrameNormalizer:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 normalize_illumination=True, use_clahe=True, clahe_clip_limit=2.0,
                 clahe_grid_size=(8, 8)):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.normalize_illumination = normalize_illumination
        self.use_clahe = use_clahe
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size) if use_clahe else None

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if self.use_clahe and self.clahe is not None:
            frames = self._apply_clahe(frames)
        frames = self._normalize_batch(frames)
        frames = (frames - self.mean.to(frames.device)) / self.std.to(frames.device)
        return frames

    def _apply_clahe(self, frames: torch.Tensor) -> torch.Tensor:
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
        if not self.normalize_illumination:
            return frames
        mean = frames.mean(dim=(2, 3), keepdim=True)
        std = frames.std(dim=(2, 3), keepdim=True) + 1e-8
        normalized = (frames - mean) / std
        alpha = 0.7
        return alpha * normalized + (1 - alpha) * frames


class Denoiser:
    def __init__(self, strength: int = 10, method: str = 'fastnlmeans'):
        self.strength = strength
        self.method = method

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        C, T, H, W = frames.shape
        frames_np = frames.permute(1, 2, 3, 0).numpy()
        frames_np = (frames_np * 255).astype(np.uint8)
        denoised = []
        for t in range(T):
            frame = frames_np[t]
            if self.method == 'fastnlmeans':
                frame = cv2.fastNlMeansDenoisingColored(frame, None, self.strength, self.strength, 7, 21)
            elif self.method == 'bilateral':
                frame = cv2.bilateralFilter(frame, -1, self.strength, self.strength)
            else:
                frame = cv2.GaussianBlur(frame, (5, 5), self.strength / 10)
            denoised.append(frame)
        denoised = np.stack(denoised)
        denoised = torch.from_numpy(denoised).permute(3, 0, 1, 2).float() / 255.0
        return denoised


class Sharpener:
    def __init__(self, strength: float = 1.0, method: str = 'unsharp'):
        self.strength = strength
        self.method = method

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        C, T, H, W = frames.shape
        frames_np = frames.permute(1, 2, 3, 0).numpy()
        frames_np = (frames_np * 255).astype(np.uint8)
        sharpened = []
        for t in range(T):
            frame = frames_np[t]
            if self.method == 'unsharp':
                blurred = cv2.GaussianBlur(frame, (0, 0), 3)
                frame = cv2.addWeighted(frame, 1 + self.strength, blurred, -self.strength, 0)
            else:
                laplacian = cv2.Laplacian(frame, cv2.CV_64F)
                frame = cv2.addWeighted(frame, 1, laplacian.astype(np.uint8), -self.strength, 0)
            sharpened.append(frame)
        sharpened = np.stack(sharpened)
        sharpened = torch.from_numpy(sharpened).permute(3, 0, 1, 2).float() / 255.0
        return sharpened


class TemporalSmoother:
    def __init__(self, window_size: int = 5, method: str = 'moving_average'):
        self.window_size = window_size
        self.method = method

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
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
            smoothed.append(alpha * frames[:, t:t+1] + (1 - alpha) * smoothed[-1])
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
    """Pipeline complet de prétraitement avec méthodes de normalisation et split."""

    def __init__(self, config: PreprocessingConfig = PreprocessingConfig(),
                 use_temporal_smoothing: bool = False):
        self.config = config
        self.preprocessor = VideoPreprocessor(config)
        self.temporal_smoother = TemporalSmoother(window_size=5, method='moving_average') if use_temporal_smoothing else None

    def __call__(self, frames: torch.Tensor, face_detector: Optional[object] = None,
                 landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        processed = self.preprocessor.preprocess(frames, face_detector, landmarks)
        if self.temporal_smoother is not None:
            processed = self.temporal_smoother(processed)
        return processed

    def preprocess_image(self, image: Union[torch.Tensor, np.ndarray, Image.Image, str],
                         face_detector: Optional[object] = None,
                         landmarks: Optional[np.ndarray] = None) -> torch.Tensor:
        return self.preprocessor.preprocess_image(image, face_detector, landmarks)

    def preprocess_images(self, images: List[Union[torch.Tensor, np.ndarray, Image.Image, str]],
                          face_detector: Optional[object] = None) -> torch.Tensor:
        return self.preprocessor.preprocess_images(images, face_detector)

    # === METHODES DE NORMALISATION ===

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalise les données avec les statistiques ImageNet.
        Args:
            data: (N, C, H, W) ou (C, H, W)
        Returns:
            normalized: Données normalisées
        """
        mean = torch.tensor(self.config.mean).view(1, 3, 1, 1).to(data.device)
        std = torch.tensor(self.config.std).view(1, 3, 1, 1).to(data.device)
        return (data - mean) / std

    def denormalize(self, data: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        """
        Dénormalise les données.
        Args:
            data: Données normalisées
            clamp: Clamper entre 0 et 1
        Returns:
            denormalized: Données dénormalisées
        """
        mean = torch.tensor(self.config.mean).view(1, 3, 1, 1).to(data.device)
        std = torch.tensor(self.config.std).view(1, 3, 1, 1).to(data.device)
        denormalized = data * std + mean
        if clamp:
            denormalized = torch.clamp(denormalized, 0, 1)
        return denormalized

    def normalize_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Normalise un batch."""
        return self.normalize(batch)

    def denormalize_batch(self, batch: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        """Dénormalise un batch."""
        return self.denormalize(batch, clamp)

    def normalize_dataset(self, dataset: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
        """
        Applique la normalisation à tout un dataset.
        Args:
            dataset: Dataset à normaliser
        Returns:
            normalized_dataset: Dataset normalisé
        """
        class NormalizedDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, normalizer):
                self.dataset = dataset
                self.normalizer = normalizer

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                data, label = self.dataset[idx]
                if isinstance(data, torch.Tensor):
                    data = self.normalizer.normalize(data)
                return data, label

        return NormalizedDataset(dataset, self)

    # === METHODES DE SPLIT ===

    def split_dataset(self, dataset: torch.utils.data.Dataset,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1,
                      shuffle: bool = True,
                      seed: int = 42) -> Dict[str, torch.utils.data.Dataset]:
        """
        Divise un dataset en train/val/test.
        Args:
            dataset: Dataset à diviser
            train_ratio: Ratio pour l'entraînement (0.7-0.9)
            val_ratio: Ratio pour la validation (0.05-0.15)
            test_ratio: Ratio pour le test (0.05-0.15)
            shuffle: Mélanger avant de diviser
            seed: Seed pour reproductibilité
        Returns:
            Dict avec 'train', 'val', 'test'
        """
        total = len(dataset)
        train_size = int(train_ratio * total)
        val_size = int(val_ratio * total)
        test_size = total - train_size - val_size

        if shuffle:
            indices = list(range(total))
            random.seed(seed)
            random.shuffle(indices)
        else:
            indices = list(range(total))

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        return {
            'train': Subset(dataset, train_indices),
            'val': Subset(dataset, val_indices),
            'test': Subset(dataset, test_indices)
        }

    def split_and_normalize(self, dataset: torch.utils.data.Dataset,
                            train_ratio: float = 0.8,
                            val_ratio: float = 0.1,
                            test_ratio: float = 0.1,
                            shuffle: bool = True,
                            seed: int = 42) -> Dict[str, torch.utils.data.Dataset]:
        """
        Divise et normalise le dataset en une seule étape.
        Returns:
            Dict avec 'train', 'val', 'test' normalisés
        """
        splits = self.split_dataset(dataset, train_ratio, val_ratio, test_ratio, shuffle, seed)
        return {
            'train': self.normalize_dataset(splits['train']),
            'val': self.normalize_dataset(splits['val']),
            'test': self.normalize_dataset(splits['test'])
        }

    def create_dataloaders(self, dataset_splits: Dict[str, torch.utils.data.Dataset],
                           batch_size: int = 32,
                           num_workers: int = 2,
                           pin_memory: bool = True) -> Dict[str, DataLoader]:
        """
        Crée des DataLoaders à partir des splits.
        Args:
            dataset_splits: Dict avec 'train', 'val', 'test'
            batch_size: Taille du batch
            num_workers: Nombre de workers
            pin_memory: Pin memory pour GPU
        Returns:
            Dict avec 'train', 'val', 'test' DataLoaders
        """
        dataloaders = {}
        for name, dataset in dataset_splits.items():
            shuffle = name == 'train'
            dataloaders[name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
        return dataloaders

    def prepare_data(self, dataset: torch.utils.data.Dataset,
                     batch_size: int = 32,
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1,
                     num_workers: int = 2) -> Dict[str, DataLoader]:
        """
        Méthode unique pour préparer les données : split + normalisation + dataloaders.
        Args:
            dataset: Dataset à préparer
            batch_size: Taille du batch
            train_ratio: Ratio d'entraînement
            val_ratio: Ratio de validation
            test_ratio: Ratio de test
            num_workers: Nombre de workers
        Returns:
            Dict avec 'train', 'val', 'test' DataLoaders
        """
        splits = self.split_and_normalize(dataset, train_ratio, val_ratio, test_ratio)
        return self.create_dataloaders(splits, batch_size, num_workers)


# === FONCTIONS UTILITAIRES ===

def create_preprocessor(image_size: Tuple[int, int] = (224, 224),
                        use_clahe: bool = True,
                        normalize_illumination: bool = True,
                        denoise: bool = False,
                        sharpen: bool = False,
                        use_temporal_smoothing: bool = False,
                        **kwargs) -> VideoPreprocessorPipeline:
    """Factory pour créer un pipeline de prétraitement."""
    config = PreprocessingConfig(
        image_size=image_size,
        use_clahe=use_clahe,
        normalize_illumination=normalize_illumination,
        denoise=denoise,
        sharpen=sharpen,
        **kwargs
    )
    return VideoPreprocessorPipeline(config=config, use_temporal_smoothing=use_temporal_smoothing)


def preprocess_image_simple(image: Union[torch.Tensor, np.ndarray, Image.Image, str],
                            image_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Fonction simple pour prétraiter une image rapidement."""
    preprocessor = create_preprocessor(image_size=image_size)
    return preprocessor.preprocess_image(image)
