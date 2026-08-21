"""
Visualisations combinées pour l'explicabilité des modèles.
Fournit des outils pour superposer et comparer les explications.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ExplanationVisualizer:
    """
    Visualiseur d'explications pour les modèles de deepfake.
    """
    
    def __init__(
        self,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.5
    ):
        self.colormap = colormap
        self.alpha = alpha
    
    def overlay(
        self,
        image: torch.Tensor,
        explanation: torch.Tensor,
        alpha: Optional[float] = None
    ) -> np.ndarray:
        """
        Superpose une explication sur une image.
        
        Args:
            image: (C, H, W) - Image originale
            explanation: (H, W) - Carte d'explication
            alpha: Transparence
            
        Returns:
            overlay: (H, W, C) - Image superposée
        """
        alpha = alpha or self.alpha
        
        # Conversion de l'image
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().transpose(1, 2, 0)
        
        # Normalisation de l'image
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Conversion de l'explication
        if isinstance(explanation, torch.Tensor):
            explanation = explanation.cpu().numpy()
        
        # Application du colormap
        explanation_colored = cv2.applyColorMap(
            (explanation * 255).astype(np.uint8),
            self.colormap
        )
        explanation_colored = explanation_colored / 255.0
        
        # Superposition
        overlay = alpha * explanation_colored + (1 - alpha) * image
        
        return overlay
    
    def create_side_by_side(
        self,
        image: torch.Tensor,
        explanations: Dict[str, torch.Tensor],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Crée une visualisation côte à côte des différentes explications.
        
        Args:
            image: Image originale
            explanations: Dict des explications
            save_path: Chemin de sauvegarde
            
        Returns:
            combined: Image combinée
        """
        num_explanations = len(explanations)
        
        # Création de la figure
        fig, axes = plt.subplots(1, num_explanations + 1, figsize=(5 * (num_explanations + 1), 5))
        
        # Image originale
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        else:
            image_np = image
        
        axes[0].imshow(image_np)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Explications
        for i, (name, explanation) in enumerate(explanations.items(), 1):
            overlay = self.overlay(image, explanation)
            axes[i].imshow(overlay)
            axes[i].set_title(name)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        
        return fig

class OverlayVisualizer(ExplanationVisualizer):
    """
    Visualiseur d'overlay avancé avec options de style.
    """
    
    def __init__(
        self,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.5,
        use_contours: bool = False,
        contour_threshold: float = 0.5
    ):
        super().__init__(colormap, alpha)
        self.use_contours = use_contours
        self.contour_threshold = contour_threshold
    
    def overlay_with_contours(
        self,
        image: torch.Tensor,
        explanation: torch.Tensor,
        alpha: Optional[float] = None
    ) -> np.ndarray:
        """
        Superpose avec contours pour une meilleure visualisation.
        """
        overlay = super().overlay(image, explanation, alpha)
        
        if self.use_contours:
            # Conversion en binaire
            binary = (explanation > self.contour_threshold).astype(np.uint8)
            
            # Détection des contours
            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Dessin des contours
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
        return overlay

class ComparisonVisualizer:
    """
    Visualiseur de comparaison entre différentes méthodes.
    """
    
    def __init__(self):
        self.visualizer = ExplanationVisualizer()
    
    def compare_methods(
        self,
        image: torch.Tensor,
        method_results: Dict[str, torch.Tensor],
        save_path: Optional[str] = None
    ) -> None:
        """
        Compare différentes méthodes d'explication.
        """
        num_methods = len(method_results)
        
        # Création de la grille
        fig, axes = plt.subplots(
            2,
            (num_methods + 1) // 2,
            figsize=(15, 10)
        )
        
        axes = axes.flatten()
        
        # Image originale
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        
        axes[0].imshow(image_np)
        axes[0].set_title('Image Originale')
        axes[0].axis('off')
        
        # Méthodes
        for i, (name, explanation) in enumerate(method_results.items(), 1):
            overlay = self.visualizer.overlay(image, explanation)
            axes[i].imshow(overlay)
            axes[i].set_title(name)
            axes[i].axis('off')
        
        # Masquer les axes inutilisés
        for i in range(num_methods + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()

class TemporalVisualizer:
    """
    Visualiseur temporel pour les explications vidéo.
    """
    
    def __init__(self):
        self.visualizer = ExplanationVisualizer()
    
    def create_temporal_animation(
        self,
        frames: torch.Tensor,
        explanations: torch.Tensor,
        save_path: Optional[str] = None,
        fps: int = 5
    ) -> None:
        """
        Crée une animation temporelle des explications.
        
        Args:
            frames: (T, C, H, W) - Frames vidéo
            explanations: (T, H, W) - Explications par frame
            save_path: Chemin de sauvegarde (GIF)
            fps: Frames par seconde
        """
        import imageio
        
        frames_overlay = []
        
        for t in range(frames.size(0)):
            frame = frames[t]
            explanation = explanations[t]
            
            overlay = self.visualizer.overlay(frame, explanation)
            overlay = (overlay * 255).astype(np.uint8)
            
            frames_overlay.append(overlay)
        
        if save_path:
            imageio.mimsave(save_path, frames_overlay, fps=fps)
    
    def plot_frame_importance(
        self,
        importance_scores: torch.Tensor,
        save_path: Optional[str] = None
    ) -> None:
        """
        Affiche l'importance de chaque frame.
        """
        plt.figure(figsize=(10, 4))
        
        frames = range(len(importance_scores))
        plt.bar(frames, importance_scores.cpu().numpy())
        
        plt.xlabel('Frame')
        plt.ylabel('Importance')
        plt.title('Importance Temporelle des Frames')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()

def create_visualization_report(
    image: torch.Tensor,
    explanations: Dict[str, torch.Tensor],
    save_dir: str,
    include_original: bool = True
) -> Dict[str, str]:
    """
    Crée un rapport de visualisation complet.
    
    Args:
        image: Image originale
        explanations: Dict des explications
        save_dir: Répertoire de sauvegarde
        include_original: Inclure l'image originale
        
    Returns:
        paths: Chemins des fichiers sauvegardés
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = ExplanationVisualizer()
    paths = {}
    
    # Image originale
    if include_original:
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        
        plt.figure(figsize=(6, 6))
        plt.imshow(image_np)
        plt.title('Image Originale')
        plt.axis('off')
        
        original_path = save_dir / 'original.png'
        plt.savefig(original_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        paths['original'] = str(original_path)
    
    # Explications individuelles
    for name, explanation in explanations.items():
        overlay = visualizer.overlay(image, explanation)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.title(f'Explication: {name}')
        plt.axis('off')
        
        explanation_path = save_dir / f'{name}.png'
        plt.savefig(explanation_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        paths[name] = str(explanation_path)
    
    # Comparaison côte à côte
    comparison_path = save_dir / 'comparison.png'
    visualizer.create_side_by_side(
        image,
        explanations,
        save_path=str(comparison_path)
    )
    paths['comparison'] = str(comparison_path)
    
    logger.info(f"Rapport de visualisation créé dans {save_dir}")
    
    return paths