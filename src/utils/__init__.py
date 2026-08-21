"""
Module utilitaires pour la détection de deepfakes avec v-JEPA.
Fournit des outils de logging, configuration, visualisation et métriques.
"""

from .logger import (
    setup_logger,
    get_logger,
    LoggerConfig,
    TensorBoardLogger,
    WandbLogger,
    MultiLogger
)

from .config import (
    ConfigManager,
    load_config,
    save_config,
    merge_configs,
    ConfigValidator,
    ExperimentConfig
)

from .visualize import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_metrics_comparison,
    visualize_video_frames,
    plot_latent_space,
    create_training_report,
    VideoVisualizer
)

from .metrics import (
    compute_accuracy,
    compute_precision_recall_f1,
    compute_auc,
    compute_confusion_matrix,
    compute_classification_report,
    MetricTracker,
    RunningAverage
)

__all__ = [
    # Logger
    'setup_logger',
    'get_logger',
    'LoggerConfig',
    'TensorBoardLogger',
    'WandbLogger',
    'MultiLogger',
    
    # Configuration
    'ConfigManager',
    'load_config',
    'save_config',
    'merge_configs',
    'ConfigValidator',
    'ExperimentConfig',
    
    # Visualisation
    'plot_training_curves',
    'plot_confusion_matrix',
    'plot_metrics_comparison',
    'visualize_video_frames',
    'plot_latent_space',
    'create_training_report',
    'VideoVisualizer',
    
    # Métriques
    'compute_accuracy',
    'compute_precision_recall_f1',
    'compute_auc',
    'compute_confusion_matrix',
    'compute_classification_report',
    'MetricTracker',
    'RunningAverage'
]

__version__ = '1.0.0'