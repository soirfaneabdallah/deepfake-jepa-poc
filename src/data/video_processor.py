"""
Traitement des vidéos pour l'extraction de frames et le prétraitement.
Optimisé pour les vidéos de visages.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union, Generator
import decord
from decord import VideoReader, cpu, gpu
import subprocess
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VideoInfo:
    """Informations sur une vidéo."""
    path: str
    num_frames: int
    fps: float
    width: int
    height: int
    duration: float
    codec: str
    size_mb: float

class VideoProcessor:
    """
    Processeur vidéo pour l'extraction et le traitement des frames.
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.use_gpu = use_gpu
        self.target_size = target_size
        self.device = gpu(0) if use_gpu and decord.gpu_available() else cpu(0)
        
        # Configuration decord
        decord.bridge.set_bridge('torch')
    
    def get_video_info(self, video_path: str) -> VideoInfo:
        """
        Récupère les informations d'une vidéo.
        """
        vr = VideoReader(video_path, ctx=self.device)
        
        info = VideoInfo(
            path=video_path,
            num_frames=len(vr),
            fps=vr.get_avg_fps(),
            width=vr[0].shape[1],
            height=vr[0].shape[0],
            duration=len(vr) / vr.get_avg_fps(),
            codec=self._get_codec(video_path),
            size_mb=Path(video_path).stat().st_size / (1024 * 1024)
        )
        
        return info
    
    def extract_frames(
        self,
        video_path: str,
        num_frames: int = 16,
        stride: int = 2,
        start_frame: Optional[int] = None
    ) -> torch.Tensor:
        """
        Extrait des frames d'une vidéo.
        
        Args:
            video_path: Chemin de la vidéo
            num_frames: Nombre de frames à extraire
            stride: Pas entre les frames
            start_frame: Frame de départ (aléatoire si None)
            
        Returns:
            frames: (C, T, H, W) - Frames extraites
        """
        vr = VideoReader(video_path, ctx=self.device)
        total_frames = len(vr)
        
        # Calcul des indices
        if start_frame is None:
            max_start = max(0, total_frames - num_frames * stride)
            start_frame = np.random.randint(0, max_start + 1)
        
        indices = [
            min(start_frame + i * stride, total_frames - 1)
            for i in range(num_frames)
        ]
        
        # Extraction des frames
        frames = vr.get_batch(indices)  # (T, H, W, C)
        
        # Conversion en (C, T, H, W)
        frames = frames.permute(3, 0, 1, 2).float() / 255.0
        
        # Redimensionnement
        if frames.shape[-2:] != self.target_size:
            frames = torch.nn.functional.interpolate(
                frames,
                size=self.target_size,
                mode='bilinear',
                align_corners=False
            )
        
        return frames
    
    def extract_all_frames(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        fps: Optional[float] = None
    ) -> Generator[torch.Tensor, None, None]:
        """
        Extrait toutes les frames d'une vidéo avec échantillonnage.
        """
        vr = VideoReader(video_path, ctx=self.device)
        
        # Calcul du pas d'échantillonnage
        if fps is not None:
            original_fps = vr.get_avg_fps()
            sample_every = max(1, int(original_fps / fps))
        else:
            sample_every = 1
        
        # Extraction
        count = 0
        for i in range(0, len(vr), sample_every):
            if max_frames is not None and count >= max_frames:
                break
            
            frame = vr[i]  # (H, W, C)
            frame = frame.permute(2, 0, 1).float() / 255.0  # (C, H, W)
            
            # Redimensionnement
            if frame.shape[-2:] != self.target_size:
                frame = torch.nn.functional.interpolate(
                    frame.unsqueeze(0),
                    size=self.target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            yield frame
            count += 1
    
    def process_video(
        self,
        video_path: str,
        face_detector: Optional[object] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Traite complètement une vidéo : extraction + détection de visages.
        """
        # Extraction des frames
        frames = self.extract_frames(video_path, **kwargs)
        
        # Détection des visages
        if face_detector is not None:
            frames = face_detector.extract_faces(frames)
        
        return frames
    
    def _get_codec(self, video_path: str) -> str:
        """
        Récupère le codec d'une vidéo avec ffprobe.
        """
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream=codec_name', '-of',
                 'default=noprint_wrappers=1:nokey=1', video_path],
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except:
            return "unknown"

class FrameExtractor:
    """
    Extracteur de frames optimisé avec différentes stratégies.
    """
    
    def __init__(self, processor: VideoProcessor):
        self.processor = processor
    
    def uniform_sampling(
        self,
        video_path: str,
        num_frames: int
    ) -> torch.Tensor:
        """
        Échantillonnage uniforme des frames.
        """
        return self.processor.extract_frames(
            video_path,
            num_frames=num_frames,
            stride=1
        )
    
    def random_sampling(
        self,
        video_path: str,
        num_frames: int
    ) -> torch.Tensor:
        """
        Échantillonnage aléatoire des frames.
        """
        return self.processor.extract_frames(
            video_path,
            num_frames=num_frames,
            stride=1,
            start_frame=None
        )
    
    def dense_sampling(
        self,
        video_path: str,
        num_frames: int,
        fps: float = 10
    ) -> torch.Tensor:
        """
        Échantillonnage dense avec fps cible.
        """
        frames = list(self.processor.extract_all_frames(
            video_path,
            max_frames=num_frames,
            fps=fps
        ))
        
        if len(frames) < num_frames:
            # Padding si pas assez de frames
            while len(frames) < num_frames:
                frames.append(frames[-1])
        
        return torch.stack(frames)

class VideoAugmentor:
    """
    Augmenteur de vidéos pour l'entraînement.
    """
    
    def __init__(
        self,
        spatial_augmentation: bool = True,
        temporal_augmentation: bool = True,
        intensity: float = 1.0
    ):
        self.spatial_augmentation = spatial_augmentation
        self.temporal_augmentation = temporal_augmentation
        self.intensity = intensity
    
    def augment(
        self,
        frames: torch.Tensor
    ) -> torch.Tensor:
        """
        Applique les augmentations à un clip vidéo.
        """
        if self.spatial_augmentation:
            frames = self._spatial_augment(frames)
        
        if self.temporal_augmentation:
            frames = self._temporal_augment(frames)
        
        return frames
    
    def _spatial_augment(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Augmentations spatiales (par frame).
        """
        C, T, H, W = frames.shape
        
        # Reshape pour traitement par frame
        frames = frames.permute(1, 0, 2, 3)  # (T, C, H, W)
        
        # Flip horizontal
        if np.random.random() < 0.5 * self.intensity:
            frames = torch.flip(frames, dims=[-1])
        
        # Rotation légère
        if np.random.random() < 0.3 * self.intensity:
            angle = np.random.uniform(-10, 10)
            frames = self._rotate_frames(frames, angle)
        
        # Reshape retour
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        
        return frames
    
    def _temporal_augment(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Augmentations temporelles.
        """
        # Frame dropout
        if np.random.random() < 0.2 * self.intensity:
            mask = torch.rand(frames.size(1)) > 0.1
            frames = frames[:, mask]
        
        # Reverse playback
        if np.random.random() < 0.1 * self.intensity:
            frames = torch.flip(frames, dims=[1])
        
        return frames
    
    def _rotate_frames(
        self,
        frames: torch.Tensor,
        angle: float
    ) -> torch.Tensor:
        """
        Rotation des frames d'un angle donné.
        """
        # Conversion en numpy pour cv2
        frames_np = frames.numpy()
        
        # Rotation de chaque frame
        rotated = []
        for frame in frames_np:
            h, w = frame.shape[-2:]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            rotated_frame = cv2.warpAffine(
                frame.transpose(1, 2, 0),
                M,
                (w, h)
            )
            rotated.append(rotated_frame.transpose(2, 0, 1))
        
        return torch.from_numpy(np.stack(rotated))