"""
Module d'évaluation pour la détection de deepfakes.
Fournit des métriques complètes, analyses ROC et tests de robustesse.
"""

from .metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    compute_detection_metrics,
    compute_eer,
    compute_detection_error_tradeoff,
    MetricsAggregator
)

from .roc_analysis import (
    ROCAnalyzer,
    ROCCurveAnalyzer,
    DETCurveAnalyzer,
    PrecisionRecallAnalyzer,
    ThresholdOptimizer
)

from .cross_dataset import (
    CrossDatasetEvaluator,
    GeneralizationMetrics,
    DatasetBiasAnalyzer,
    CrossDatasetReport
)

from .robustness import (
    RobustnessEvaluator,
    AdversarialEvaluator,
    PerturbationEvaluator,
    CompressionRobustness,
    NoiseRobustness,
    RobustnessReport
)

__all__ = [
    # Métriques
    'ClassificationMetrics',
    'compute_classification_metrics',
    'compute_detection_metrics',
    'compute_eer',
    'compute_detection_error_tradeoff',
    'MetricsAggregator',
    
    # Analyse ROC
    'ROCAnalyzer',
    'ROCCurveAnalyzer',
    'DETCurveAnalyzer',
    'PrecisionRecallAnalyzer',
    'ThresholdOptimizer',
    
    # Cross-dataset
    'CrossDatasetEvaluator',
    'GeneralizationMetrics',
    'DatasetBiasAnalyzer',
    'CrossDatasetReport',
    
    # Robustesse
    'RobustnessEvaluator',
    'AdversarialEvaluator',
    'PerturbationEvaluator',
    'CompressionRobustness',
    'NoiseRobustness',
    'RobustnessReport'
]

__version__ = '1.0.0'