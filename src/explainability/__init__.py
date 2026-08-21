"""
Module d'explicabilité pour la détection de deepfakes.
Fournit des outils de visualisation pour comprendre les décisions du modèle.

Méthodes implémentées :
- Grad-CAM : Gradient-weighted Class Activation Mapping
- Grad-CAM++ : Version améliorée avec pondération des gradients
- Saliency Maps : Cartes de saillance basées sur les gradients
- Visualisations combinées : Superposition et comparaison
"""

from .grad_cam import (
    GradCAM,
    GradCAMExplainer,
    TemporalGradCAM
)

from .grad_cam_pp import (
    GradCAMPlusPlus,
    GradCAMPlusPlusExplainer
)

from .saliency_maps import (
    SaliencyMap,
    IntegratedGradients,
    SmoothGrad,
    GuidedBackpropagation
)

from .visualization import (
    ExplanationVisualizer,
    OverlayVisualizer,
    ComparisonVisualizer,
    TemporalVisualizer,
    create_visualization_report
)

__all__ = [
    # Grad-CAM
    'GradCAM',
    'GradCAMExplainer',
    'TemporalGradCAM',
    
    # Grad-CAM++
    'GradCAMPlusPlus',
    'GradCAMPlusPlusExplainer',
    
    # Cartes de saillance
    'SaliencyMap',
    'IntegratedGradients',
    'SmoothGrad',
    'GuidedBackpropagation',
    
    # Visualisation
    'ExplanationVisualizer',
    'OverlayVisualizer',
    'ComparisonVisualizer',
    'TemporalVisualizer',
    'create_visualization_report'
]

__version__ = '1.0.0'