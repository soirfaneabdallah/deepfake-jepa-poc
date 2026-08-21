"""
Analyse ROC détaillée pour la détection de deepfakes.
Fournit des courbes ROC, DET, précision-rappel et optimisation de seuil.
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, auc
)
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ROCAnalyzer:
    """
    Analyseur ROC complet.
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_roc(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calcule la courbe ROC.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc_score = roc_auc_score(y_true, y_scores)
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc_score
        }
    
    def plot_roc(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "Courbe ROC",
        save_path: Optional[str] = None,
        show_diagonal: bool = True
    ) -> None:
        """
        Affiche la courbe ROC.
        """
        roc_data = self.compute_roc(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        
        plt.plot(
            roc_data['fpr'],
            roc_data['tpr'],
            label=f"ROC (AUC = {roc_data['auc']:.3f})",
            linewidth=2
        )
        
        if show_diagonal:
            plt.plot(
                [0, 1],
                [0, 1],
                'k--',
                label='Aléatoire',
                alpha=0.5
            )
        
        plt.xlabel('Taux de Faux Positifs (FPR)')
        plt.ylabel('Taux de Vrais Positifs (TPR)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path or self.save_dir:
            path = save_path or self.save_dir / 'roc_curve.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_multi_roc(
        self,
        results: Dict[str, Tuple[np.ndarray, np.ndarray]],
        title: str = "Comparaison des Courbes ROC",
        save_path: Optional[str] = None
    ) -> None:
        """
        Affiche plusieurs courbes ROC pour comparaison.
        
        Args:
            results: Dict {nom: (y_true, y_scores)}
        """
        plt.figure(figsize=(10, 8))
        
        for name, (y_true, y_scores) in results.items():
            roc_data = self.compute_roc(y_true, y_scores)
            
            plt.plot(
                roc_data['fpr'],
                roc_data['tpr'],
                label=f"{name} (AUC = {roc_data['auc']:.3f})",
                linewidth=2
            )
        
        plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire', alpha=0.5)
        
        plt.xlabel('Taux de Faux Positifs (FPR)')
        plt.ylabel('Taux de Vrais Positifs (TPR)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path or self.save_dir:
            path = save_path or self.save_dir / 'multi_roc.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
        
        plt.close()

class ROCCurveAnalyzer(ROCAnalyzer):
    """
    Analyseur ROC avancé avec métriques supplémentaires.
    """
    
    def compute_youden_index(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calcule l'indice de Youden (J = sensibilité + spécificité - 1).
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        youden_indices = tpr - fpr
        optimal_idx = np.argmax(youden_indices)
        
        return youden_indices[optimal_idx], thresholds[optimal_idx]
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        criterion: str = 'youden'  # youden, f1, accuracy
    ) -> Tuple[float, float]:
        """
        Trouve le seuil optimal selon différents critères.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        if criterion == 'youden':
            # Indice de Youden
            scores = tpr - fpr
        elif criterion == 'f1':
            # F1-score
            scores = []
            for threshold in thresholds:
                y_pred = (y_scores > threshold).astype(int)
                from sklearn.metrics import f1_score
                scores.append(f1_score(y_true, y_pred))
            scores = np.array(scores)
        else:  # accuracy
            scores = []
            for threshold in thresholds:
                y_pred = (y_scores > threshold).astype(int)
                from sklearn.metrics import accuracy_score
                scores.append(accuracy_score(y_true, y_pred))
            scores = np.array(scores)
        
        optimal_idx = np.argmax(scores)
        
        return thresholds[optimal_idx], scores[optimal_idx]

class DETCurveAnalyzer(ROCAnalyzer):
    """
    Analyseur de courbe DET (Detection Error Tradeoff).
    """
    
    def plot_det_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "Courbe DET",
        save_path: Optional[str] = None
    ) -> None:
        """
        Affiche la courbe DET.
        """
        from scipy.stats import norm
        
        # Calcul des taux
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        
        # Transformation en échelle normale
        fpr_norm = norm.ppf(fpr + 1e-10)
        fnr_norm = norm.ppf(fnr + 1e-10)
        
        plt.figure(figsize=(8, 6))
        
        plt.plot(fpr_norm, fnr_norm, linewidth=2, label='DET')
        
        # Points de référence
        for far in [0.01, 0.05, 0.1, 0.2]:
            far_norm = norm.ppf(far)
            plt.axvline(x=far_norm, linestyle='--', alpha=0.3, color='gray')
            plt.text(far_norm, plt.ylim()[1], f'{far*100:.0f}%', 
                    rotation=90, verticalalignment='top')
        
        plt.xlabel('Taux de Fausses Acceptations (FAR)')
        plt.ylabel('Taux de Faux Rejets (FRR)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path or self.save_dir:
            path = save_path or self.save_dir / 'det_curve.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
        
        plt.close()

class PrecisionRecallAnalyzer(ROCAnalyzer):
    """
    Analyseur de courbe précision-rappel.
    """
    
    def compute_pr_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calcule la courbe précision-rappel.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'average_precision': ap
        }
    
    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "Courbe Précision-Rappel",
        save_path: Optional[str] = None
    ) -> None:
        """
        Affiche la courbe précision-rappel.
        """
        pr_data = self.compute_pr_curve(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        
        plt.plot(
            pr_data['recall'],
            pr_data['precision'],
            label=f"PR (AP = {pr_data['average_precision']:.3f})",
            linewidth=2
        )
        
        plt.xlabel('Rappel')
        plt.ylabel('Précision')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path or self.save_dir:
            path = save_path or self.save_dir / 'pr_curve.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
        
        plt.close()

class ThresholdOptimizer:
    """
    Optimiseur de seuil pour la détection.
    """
    
    def __init__(self):
        self.analyzer = ROCCurveAnalyzer()
    
    def optimize(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        criteria: List[str] = ['youden', 'f1', 'accuracy']
    ) -> Dict[str, Dict[str, float]]:
        """
        Optimise le seuil selon plusieurs critères.
        """
        results = {}
        
        for criterion in criteria:
            threshold, score = self.analyzer.find_optimal_threshold(
                y_true,
                y_scores,
                criterion
            )
            
            # Calcul des métriques au seuil optimal
            y_pred = (y_scores > threshold).astype(int)
            
            from .metrics import compute_classification_metrics
            metrics = compute_classification_metrics(
                y_true,
                y_pred,
                y_scores
            )
            
            results[criterion] = {
                'threshold': threshold,
                'score': score,
                'metrics': metrics.to_dict()
            }
        
        return results
    
    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Affiche l'analyse des seuils.
        """
        from sklearn.metrics import f1_score, accuracy_score
        
        thresholds = np.linspace(0, 1, 100)
        
        f1_scores = []
        accuracies = []
        
        for threshold in thresholds:
            y_pred = (y_scores > threshold).astype(int)
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
            accuracies.append(accuracy_score(y_true, y_pred))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # F1-score
        axes[0].plot(thresholds, f1_scores, linewidth=2)
        axes[0].set_xlabel('Seuil')
        axes[0].set_ylabel('F1-score')
        axes[0].set_title('F1-score vs Seuil')
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(thresholds, accuracies, linewidth=2, color='orange')
        axes[1].set_xlabel('Seuil')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy vs Seuil')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()