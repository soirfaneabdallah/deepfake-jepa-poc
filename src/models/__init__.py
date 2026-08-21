"""
Module des modèles pour la détection de deepfakes avec v-JEPA.
Contient les architectures d'encodeurs, détecteurs et analyseurs.
"""

from .jepa import (
    VJEPAModel,
    VJEPAConfig,
    LatentPredictor,
    VideoEncoder
)

from .encoders import (
    VideoViTEncoder,
    VideoSwinEncoder,
    SlowFastEncoder,
    create_encoder
)

from .anomaly_detector import (
    AnomalyDetector,
    MahalanobisDetector,
    OneClassSVMDetector,
    IsolationForestDetector,
    DeepSVDDDetector
)

from .forensic_analyzer import (
    ForensicAnalyzer,
    SpectralAnalyzer,
    CompressionArtifactDetector,
    TextureAnalyzer,
    GeometricAnalyzer
)

from .hybrid_detector import (
    HybridDeepfakeDetector,
    MultiModalFusion,
    AdaptiveFusion
)

from .temporal_models import (
    TemporalTransformer,
    TemporalLSTM,
    TemporalGRU,
    create_temporal_model
)

__all__ = [
    # JEPA
    'VJEPAModel',
    'VJEPAConfig',
    'LatentPredictor',
    'VideoEncoder',
    
    # Encodeurs
    'VideoViTEncoder',
    'VideoSwinEncoder',
    'SlowFastEncoder',
    'create_encoder',
    
    # Détection d'anomalies
    'AnomalyDetector',
    'MahalanobisDetector',
    'OneClassSVMDetector',
    'IsolationForestDetector',
    'DeepSVDDDetector',
    
    # Analyse médico-légale
    'ForensicAnalyzer',
    'SpectralAnalyzer',
    'CompressionArtifactDetector',
    'TextureAnalyzer',
    'GeometricAnalyzer',
    
    # Détecteur hybride
    'HybridDeepfakeDetector',
    'MultiModalFusion',
    'AdaptiveFusion',
    
    # Modèles temporels
    'TemporalTransformer',
    'TemporalLSTM',
    'TemporalGRU',
    'create_temporal_model'
]

__version__ = '1.0.0'