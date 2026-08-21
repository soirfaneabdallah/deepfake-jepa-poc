"""
Utilitaires de visualisation pour le projet.
Fournit des fonctions pour visualiser les résultats et les données.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Affiche les courbes d'entraînement.
    """
    num_plots = 1 + (1 if train_metrics else 0)
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    
    if num_plots == 1:
        axes = [axes]
    
    # Courbe de perte
    axes[0].plot(train_losses, label='Train', linewidth=2)
    axes[0].plot(val_losses, label='Validation', linewidth=2)
    axes[0].set_xlabel('Époque')
    axes[0].set_ylabel('Perte')
    axes[0].set_title('Courbe de Perte')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Courbe de métrique
    if train_metrics and val_metrics:
        metric_name = list(train_metrics.keys())[0]
        
        axes[1].plot(train_metrics[metric_name], label='Train', linewidth=2)
        axes[1].plot(val_metrics[metric_name], label='Validation', linewidth=2)
        axes[1].set_xlabel('Époque')
        axes[1].set_ylabel(metric_name.replace('_', ' ').title())
        axes[1].set_title(f'Courbe de {metric_name.replace("_", " ").title()}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(
    confusion_mat: np.ndarray,
    class_names: List[str] = ['Réel', 'Fake'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Affiche la matrice de confusion.
    """
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Nombre d\'échantillons'}
    )
    
    plt.xlabel('Prédictions')
    plt.ylabel('Vérité Terrain')
    plt.title('Matrice de Confusion')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metrics_to_plot: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Compare les métriques de différents modèles.
    """
    models = list(metrics_dict.keys())
    num_metrics = len(metrics_to_plot)
    
    x = np.arange(len(models))
    width = 0.8 / num_metrics
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, metric in enumerate(metrics_to_plot):
        values = [metrics_dict[model].get(metric, 0) for model in models]
        offset = (i - num_metrics / 2 + 0.5) * width
        
        bars = ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title())
        
        # Ajout des valeurs
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{value:.3f}',
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    ax.set_xlabel('Modèles')
    ax.set_ylabel('Score')
    ax.set_title('Comparaison des Métriques')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_video_frames(
    video: torch.Tensor,
    num_frames: int = 8,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 4)
) -> None:
    """
    Visualise les frames d'une vidéo.
    
    Args:
        video: (C, T, H, W) ou (T, C, H, W)
        num_frames: Nombre de frames à afficher
    """
    # Conversion en format (T, C, H, W)
    if video.dim() == 4 and video.size(0) <= 10:
        video = video.permute(1, 0, 2, 3)
    
    C, T, H, W = video.shape
    
    # Sélection des frames
    if T > num_frames:
        indices = np.linspace(0, T-1, num_frames, dtype=int)
        frames = video[:, indices]
    else:
        frames = video
        num_frames = T
    
    fig, axes = plt.subplots(1, num_frames, figsize=figsize)
    
    for i in range(num_frames):
        frame = frames[:, i].cpu().numpy().transpose(1, 2, 0)
        
        # Normalisation
        frame = (frame - frame.min()) / (frame.max() - frame.min())
        
        axes[i].imshow(frame)
        axes[i].axis('off')
        axes[i].set_title(f'Frame {i+1}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_latent_space(
    features: torch.Tensor,
    labels: torch.Tensor,
    method: str = 'tsne',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Visualise l'espace latent en 2D.
    
    Args:
        features: (N, D) - Features latentes
        labels: (N,) - Labels
        method: 'tsne' ou 'pca'
    """
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Réduction de dimensionnalité
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    features_2d = reducer.fit_transform(features_np)
    
    # Affichage
    plt.figure(figsize=figsize)
    
    unique_labels = np.unique(labels_np)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels_np == label
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[i]],
            label=f'Classe {label}',
            alpha=0.6,
            s=50
        )
    
    plt.xlabel(f'{method.upper()} Composante 1')
    plt.ylabel(f'{method.upper()} Composante 2')
    plt.title(f'Visualisation de l\'Espace Latent ({method.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_training_report(
    metrics_history: Dict[str, List[float]],
    save_path: str
) -> None:
    """
    Crée un rapport d'entraînement complet.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Courbes d'entraînement
    train_losses = metrics_history.get('train_loss', [])
    val_losses = metrics_history.get('val_loss', [])
    
    if train_losses and val_losses:
        plot_training_curves(
            train_losses,
            val_losses,
            save_path=str(save_path / 'training_curves.png')
        )
    
    # Métriques finales
    final_metrics = {}
    for key, values in metrics_history.items():
        if values:
            final_metrics[key] = values[-1]
    
    # Sauvegarde des métriques
    import json
    with open(save_path / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    logger.info(f"Rapport d'entraînement créé dans {save_path}")

class VideoVisualizer:
    """
    Visualiseur vidéo avancé.
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def create_video_grid(
        self,
        videos: List[torch.Tensor],
        titles: List[str],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 10)
    ) -> None:
        """
        Crée une grille de vidéos.
        """
        num_videos = len(videos)
        num_frames = videos[0].size(1) if videos[0].dim() == 5 else videos[0].size(0)
        
        fig, axes = plt.subplots(
            num_videos,
            num_frames,
            figsize=figsize
        )
        
        for i, (video, title) in enumerate(zip(videos, titles)):
            if video.dim() == 5:
                video = video[0]  # Prendre le premier batch
            
            for j in range(num_frames):
                frame = video[j].cpu().numpy().transpose(1, 2, 0)
                frame = (frame - frame.min()) / (frame.max() - frame.min())
                
                axes[i, j].imshow(frame)
                axes[i, j].axis('off')
                
                if j == 0:
                    axes[i, j].set_ylabel(title, fontsize=12)
        
        plt.tight_layout()
        
        if save_path or self.save_dir:
            path = save_path or self.save_dir / 'video_grid.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_frame_comparison(
        self,
        real_frame: torch.Tensor,
        fake_frame: torch.Tensor,
        save_path: Optional[str] = None
    ) -> None:
        """
        Compare une frame réelle et une frame fake.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Frame réelle
        real_np = real_frame.cpu().numpy().transpose(1, 2, 0)
        real_np = (real_np - real_np.min()) / (real_np.max() - real_np.min())
        axes[0].imshow(real_np)
        axes[0].set_title('Frame Réelle')
        axes[0].axis('off')
        
        # Frame fake
        fake_np = fake_frame.cpu().numpy().transpose(1, 2, 0)
        fake_np = (fake_np - fake_np.min()) / (fake_np.max() - fake_np.min())
        axes[1].imshow(fake_np)
        axes[1].set_title('Frame Fake')
        axes[1].axis('off')
        
        # Différence
        diff = np.abs(real_np - fake_np)
        axes[2].imshow(diff, cmap='hot')
        axes[2].set_title('Différence')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path or self.save_dir:
            path = save_path or self.save_dir / 'frame_comparison.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
        
        plt.show()