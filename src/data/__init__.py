"""
Module de gestion des données pour v-JEPA.
Fournit les outils de chargement, prétraitement et augmentation des vidéos.
"""

from .dataset import (
    VideoFaceDataset,
    ContinualVideoDataset,
    VJEPADataset
)

from .preprocessing import (
    VideoPreprocessor,
    FaceAligner,
    FrameNormalizer
)

from .face_detector import (
    FaceDetector,
    MTCNNDetector,
    RetinaFaceDetector,
    MediaPipeDetector
)

from .video_processor import (
    VideoProcessor,
    FrameExtractor,
    VideoAugmentor
)

from .augmentation import (
    SpatialAugmentation,
    TemporalAugmentation,
    VJEPAAugmentation,
    create_vjepa_augmentations
)

__all__ = [
    # Datasets
    'VideoFaceDataset',
    'ContinualVideoDataset',
    'VJEPADataset',
    
    # Prétraitement
    'VideoPreprocessor',
    'FaceAligner',
    'FrameNormalizer',
    
    # Détection de visages
    'FaceDetector',
    'MTCNNDetector',
    'RetinaFaceDetector',
    'MediaPipeDetector',
    
    # Traitement vidéo
    'VideoProcessor',
    'FrameExtractor',
    'VideoAugmentor',
    
    # Augmentations
    'SpatialAugmentation',
    'TemporalAugmentation',
    'VJEPAAugmentation',
    'create_vjepa_augmentations'
]

# Version du module
__version__ = '1.0.0'