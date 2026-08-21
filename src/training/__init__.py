"""
Module d'entraînement pour la détection de deepfakes avec v-JEPA.
Fournit les entraîneurs spécialisés et les outils d'évaluation.
"""

from .trainer import (
    BaseTrainer,
    TrainingConfig,
    EarlyStopping,
    CheckpointManager,
    MetricsTracker
)

from .jepa_trainer import (
    JEPATrainer,
    JEPATrainingConfig,
    JEPALossCalculator
)

from .continual_trainer import (
    ContinualTrainer,
    ContinualConfig,
    TaskManager,
    ForgettingTracker
)

from .evaluator import (
    ModelEvaluator,
    EvaluationMetrics,
    CrossDatasetEvaluator,
    RobustnessEvaluator,
    AnomalyEvaluator
)

__all__ = [
    # Base
    'BaseTrainer',
    'TrainingConfig',
    'EarlyStopping',
    'CheckpointManager',
    'MetricsTracker',
    
    # JEPA
    'JEPATrainer',
    'JEPATrainingConfig',
    'JEPALossCalculator',
    
    # Apprentissage continu
    'ContinualTrainer',
    'ContinualConfig',
    'TaskManager',
    'ForgettingTracker',
    
    # Évaluation
    'ModelEvaluator',
    'EvaluationMetrics',
    'CrossDatasetEvaluator',
    'RobustnessEvaluator',
    'AnomalyEvaluator'
]

__version__ = '1.0.0'