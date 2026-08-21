"""
Module d'apprentissage continu pour la détection de deepfakes.
Implémente EWC, mémoire duale et stratégies avancées pour éviter l'oubli catastrophique.
"""

from .ewc import (
    ElasticWeightConsolidation,
    OnlineEWC,
    SIEWC,
    MASEWC
)

from .memory import (
    EpisodicMemory,
    SemanticMemory,
    DualMemory,
    MemorySampler,
    HerdingSampler,
    UncertaintySampler
)

from .strategies import (
    ContinualStrategy,
    EWCStrategy,
    ReplayStrategy,
    EWCWithReplayStrategy,
    LwFStrategy,
    ProgressiveNetworkStrategy
)

from .metrics import (
    ForgettingMetric,
    BackwardTransfer,
    ForwardTransfer,
    ContinualMetrics,
    compute_forgetting,
    compute_backward_transfer,
    compute_forward_transfer
)

__all__ = [
    # EWC
    'ElasticWeightConsolidation',
    'OnlineEWC',
    'SIEWC',
    'MASEWC',
    
    # Mémoire
    'EpisodicMemory',
    'SemanticMemory',
    'DualMemory',
    'MemorySampler',
    'HerdingSampler',
    'UncertaintySampler',
    
    # Stratégies
    'ContinualStrategy',
    'EWCStrategy',
    'ReplayStrategy',
    'EWCWithReplayStrategy',
    'LwFStrategy',
    'ProgressiveNetworkStrategy',
    
    # Métriques
    'ForgettingMetric',
    'BackwardTransfer',
    'ForwardTransfer',
    'ContinualMetrics',
    'compute_forgetting',
    'compute_backward_transfer',
    'compute_forward_transfer'
]

__version__ = '1.0.0'