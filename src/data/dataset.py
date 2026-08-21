"""
Dataset principal pour l'entraînement v-JEPA.
Gère le chargement des vidéos de visages avec échantillonnage efficace.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Callable
import logging
from dataclasses import dataclass, field
import decord
from decord import VideoReader, cpu, gpu
import json
import random

logger = logging.getLogger(__name__)

@dataclass
class VideoMetadata:
    """Métadonnées d'une vidéo."""
    path: str
    label: int  # 0 = réel, 1 = fake
    generator: str  # original, DeepFakes, Face2Face, etc.
    num_frames: int
    fps: float
    duration: float
    resolution: Tuple[int, int]
    compression: str = "c23"

class VideoFaceDataset(Dataset):
    """
    Dataset de vidéos de visages pour v-JEPA.
    
    Charge les vidéos FaceForensics++ et extrait des clips de visages.
    Optimisé pour l'entraînement auto-supervisé.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        num_frames: int = 16,
        frame_stride: int = 2,
        image_size: Tuple[int, int] = (224, 224),
        generators: Optional[List[str]] = None,
        include_real: bool = True,
        transform: Optional[Callable] = None,
        face_detector: Optional[object] = None,
        cache_metadata: bool = True,
        metadata_path: Optional[str] = None
    ):
        """
        Args:
            data_root: Chemin racine des données FaceForensics++
            split: 'train', 'val', ou 'test'
            num_frames: Nombre de frames par clip
            frame_stride: Pas d'échantillonnage des frames
            image_size: Taille des images (H, W)
            generators: Liste des générateurs à inclure
            include_real: Inclure les vidéos réelles
            transform: Transformations à appliquer
            face_detector: Détecteur de visages
            cache_metadata: Mettre en cache les métadonnées
            metadata_path: Chemin du fichier de métadonnées
        """
        self.data_root = Path(data_root)
        self.split = split
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.transform = transform
        self.face_detector = face_detector
        
        # Configuration decord
        decord.bridge.set_bridge('torch')
        
        # Chargement des vidéos
        self.videos = []
        self.metadata = {}
        
        # Chargement ou création des métadonnées
        if metadata_path and Path(metadata_path).exists():
            self._load_metadata(metadata_path)
        else:
            self._scan_videos(generators, include_real)
            if cache_metadata and metadata_path:
                self._save_metadata(metadata_path)
        
        logger.info(f"Dataset {split}: {len(self.videos)} vidéos chargées")
    
    def _scan_videos(
        self,
        generators: Optional[List[str]],
        include_real: bool
    ) -> None:
        """
        Scanne les vidéos dans le répertoire des données.
        """
        # Vidéos réelles
        if include_real:
            real_dir = self.data_root / "original_sequences" / "youtube" / "c23" / "videos"
            if real_dir.exists():
                for video_path in real_dir.glob("*.mp4"):
                    self._add_video(video_path, label=0, generator="original")
        
        # Vidéos manipulées
        if generators is None:
            generators = ["DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures"]
        
        for generator in generators:
            fake_dir = self.data_root / "manipulated_sequences" / generator / "c23" / "videos"
            if fake_dir.exists():
                for video_path in fake_dir.glob("*.mp4"):
                    self._add_video(video_path, label=1, generator=generator)
    
    def _add_video(
        self,
        video_path: Path,
        label: int,
        generator: str
    ) -> None:
        """
        Ajoute une vidéo au dataset avec ses métadonnées.
        """
        try:
            # Lecture des métadonnées vidéo
            vr = VideoReader(str(video_path), ctx=cpu(0))
            
            metadata = VideoMetadata(
                path=str(video_path),
                label=label,
                generator=generator,
                num_frames=len(vr),
                fps=vr.get_avg_fps(),
                duration=len(vr) / vr.get_avg_fps(),
                resolution=(vr[0].shape[1], vr[0].shape[0])
            )
            
            video_id = f"{generator}_{video_path.stem}"
            self.videos.append(video_id)
            self.metadata[video_id] = metadata
            
        except Exception as e:
            logger.warning(f"Erreur lors du chargement de {video_path}: {e}")
    
    def _load_metadata(self, metadata_path: str) -> None:
        """
        Charge les métadonnées depuis un fichier JSON.
        """
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        self.videos = data['videos']
        for video_id, meta in data['metadata'].items():
            self.metadata[video_id] = VideoMetadata(**meta)
    
    def _save_metadata(self, metadata_path: str) -> None:
        """
        Sauvegarde les métadonnées dans un fichier JSON.
        """
        data = {
            'videos': self.videos,
            'metadata': {
                vid: vars(meta) for vid, meta in self.metadata.items()
            }
        }
        
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def __len__(self) -> int:
        return len(self.videos)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Charge un clip vidéo et retourne les frames du visage.
        
        Returns:
            Dict contenant:
                - 'frames': (C, T, H, W) - Clip vidéo
                - 'label': int - Label (0=réel, 1=fake)
                - 'generator': str - Générateur
                - 'video_id': str - Identifiant vidéo
        """
        video_id = self.videos[idx]
        metadata = self.metadata[video_id]
        
        # Chargement de la vidéo
        frames = self._load_video_clip(metadata)
        
        # Détection et extraction des visages
        if self.face_detector is not None:
            frames = self.face_detector.extract_faces(frames)
        
        # Application des transformations
        if self.transform is not None:
            frames = self.transform(frames)
        
        return {
            'frames': frames,
            'label': metadata.label,
            'generator': metadata.generator,
            'video_id': video_id
        }
    
    def _load_video_clip(self, metadata: VideoMetadata) -> torch.Tensor:
        """
        Charge un clip vidéo à partir des métadonnées.
        """
        vr = VideoReader(metadata.path, ctx=cpu(0))
        
        # Échantillonnage des indices de frames
        indices = self._sample_frame_indices(metadata.num_frames)
        
        # Extraction des frames
        frames = vr.get_batch(indices)  # (T, H, W, C)
        
        # Conversion en (C, T, H, W)
        frames = frames.permute(3, 0, 1, 2).float()
        
        # Normalisation [0, 1]
        frames = frames / 255.0
        
        # Redimensionnement
        if frames.shape[-2:] != self.image_size:
            frames = torch.nn.functional.interpolate(
                frames,
                size=self.image_size,
                mode='bilinear',
                align_corners=False
            )
        
        return frames
    
    def _sample_frame_indices(self, total_frames: int) -> List[int]:
        """
        Échantillonne les indices de frames de manière uniforme.
        """
        if total_frames <= self.num_frames * self.frame_stride:
            # Pas assez de frames, répéter les dernières
            indices = list(range(0, total_frames, self.frame_stride))
            while len(indices) < self.num_frames:
                indices.append(indices[-1] if indices else 0)
            return indices[:self.num_frames]
        
        # Échantillonnage uniforme
        max_start = total_frames - (self.num_frames * self.frame_stride)
        start = random.randint(0, max_start)
        
        indices = [
            start + i * self.frame_stride 
            for i in range(self.num_frames)
        ]
        
        return indices

class VJEPADataset(VideoFaceDataset):
    """
    Dataset spécifique pour l'entraînement v-JEPA.
    Ne charge que les vidéos réelles (auto-supervisé).
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        **kwargs
    ):
        super().__init__(
            data_root=data_root,
            split=split,
            include_real=True,
            generators=[],  # Pas de vidéos manipulées
            **kwargs
        )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retourne deux vues augmentées du même clip pour JEPA.
        """
        sample = super().__getitem__(idx)
        
        # Création de deux vues pour l'apprentissage contrastif
        frames = sample['frames']
        
        return {
            'context': frames,  # Vue contexte (partiellement masquée)
            'target': frames.clone(),  # Vue cible (complète)
            'video_id': sample['video_id']
        }

class ContinualVideoDataset(VideoFaceDataset):
    """
    Dataset pour l'apprentissage continu.
    Chaque tâche correspond à un générateur de deepfake.
    """
    
    def __init__(
        self,
        data_root: str,
        task_id: int,
        generator: str,
        split: str = 'train',
        include_real: bool = True,
        **kwargs
    ):
        self.task_id = task_id
        self.generator = generator
        
        super().__init__(
            data_root=data_root,
            split=split,
            generators=[generator],
            include_real=include_real,
            **kwargs
        )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retourne un échantillon avec l'identifiant de tâche.
        """
        sample = super().__getitem__(idx)
        sample['task_id'] = self.task_id
        return sample

def create_dataloaders(
    data_root: str,
    config: Dict,
    split: str = 'train'
) -> DataLoader:
    """
    Crée les dataloaders pour l'entraînement.
    """
    dataset = VJEPADataset(
        data_root=data_root,
        split=split,
        num_frames=config['data']['video']['num_frames'],
        image_size=tuple(config['data']['preprocessing']['image_size']),
        transform=create_vjepa_augmentations(config)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config['system']['num_workers'],
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader