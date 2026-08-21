"""
Métriques d'évaluation complètes pour la détection de deepfakes.
Inclut les métriques standard et spécifiques à la détection d'anomalies.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score, precision_recall_curve, roc_curve
)
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class ClassificationMetrics:
    """Métriques de classification complètes."""
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    specificity: float = 0.0
    auc_roc: float = 0.0
    average_precision: float = 0.0
    matthews_corrcoef: float = 0.0
    cohen_kappa: float = 0.0
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0
    inference_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convertit en dictionnaire."""
        return {
            'accuracy': self.accuracy,
            'balanced_accuracy': self.balanced_accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'specificity': self.specificity,
            'auc_roc': self.auc_roc,
            'average_precision': self.average_precision,
            'matthews_corrcoef': self.matthews_corrcoef,
            'cohen_kappa': self.cohen_kappa,
            'true_positive': self.true_positive,
            'true_negative': self.true_negative,
            'false_positive': self.false_positive,
            'false_negative': self.false_negative,
            'inference_time_ms': self.inference_time_ms
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convertit en DataFrame pandas."""
        return pd.DataFrame([self.to_dict()])

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    inference_time_ms: float = 0.0
) -> ClassificationMetrics:
    """
    Calcule toutes les métriques de classification.
    
    Args:
        y_true: Labels réels (0 ou 1)
        y_pred: Prédictions (0 ou 1)
        y_scores: Scores de probabilité (pour AUC)
        inference_time_ms: Temps d'inférence moyen en ms
        
    Returns:
        ClassificationMetrics
    """
    metrics = ClassificationMetrics()
    
    # Métriques de base
    metrics.accuracy = accuracy_score(y_true, y_pred)
    metrics.balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    metrics.precision = precision_score(y_true, y_pred, zero_division=0)
    metrics.recall = recall_score(y_true, y_pred, zero_division=0)
    metrics.f1_score = f1_score(y_true, y_pred, zero_division=0)
    metrics.matthews_corrcoef = matthews_corrcoef(y_true, y_pred)
    metrics.cohen_kappa = cohen_kappa_score(y_true, y_pred)
    
    # Matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.true_positive = tp
    metrics.true_negative = tn
    metrics.false_positive = fp
    metrics.false_negative = fn
    
    # Spécificité
    metrics.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Métriques basées sur les scores
    if y_scores is not None:
        try:
            metrics.auc_roc = roc_auc_score(y_true, y_scores)
            metrics.average_precision = average_precision_score(y_true, y_scores)
        except ValueError as e:
            logger.warning(f"Impossible de calculer AUC: {e}")
    
    # Temps d'inférence
    metrics.inference_time_ms = inference_time_ms
    
    return metrics

def compute_detection_metrics(
    anomaly_scores: np.ndarray,
    y_true: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Calcule les métriques spécifiques à la détection d'anomalies.
    
    Args:
        anomaly_scores: Scores d'anomalie
        y_true: Labels réels (0 = normal, 1 = anomalie)
        threshold: Seuil de détection
        
    Returns:
        Dict des métriques
    """
    # Détermination du seuil
    if threshold is None:
        # Seuil optimal basé sur la courbe ROC
        fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]
    
    # Prédictions basées sur le seuil
    y_pred = (anomaly_scores > threshold).astype(int)
    
    # Calcul des métriques
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        anomaly_scores
    ).to_dict()
    
    # Métriques supplémentaires
    metrics['threshold'] = threshold
    
    # Taux de détection à différents taux de fausses alarmes
    for far_target in [0.01, 0.05, 0.1]:
        detection_rate = compute_detection_rate_at_far(
            anomaly_scores,
            y_true,
            far_target
        )
        metrics[f'detection_rate_at_far_{far_target}'] = detection_rate
    
    return metrics

def compute_eer(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Tuple[float, float]:
    """
    Calcule l'Equal Error Rate (EER).
    
    Args:
        y_true: Labels réels
        y_scores: Scores de probabilité
        
    Returns:
        eer: Equal Error Rate
        threshold: Seuil au point EER
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Calcul du point d'égalité
    fnr = 1 - tpr
    eer_values = np.abs(fpr - fnr)
    eer_idx = np.argmin(eer_values)
    
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    threshold = thresholds[eer_idx]
    
    return eer, threshold

def compute_detection_error_tradeoff(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Calcule la courbe DET (Detection Error Tradeoff).
    
    Returns:
        Dict contenant FAR, FRR et les seuils
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    return {
        'far': fpr,  # False Acceptance Rate
        'frr': fnr,  # False Rejection Rate
        'thresholds': thresholds
    }

def compute_detection_rate_at_far(
    scores: np.ndarray,
    y_true: np.ndarray,
    target_far: float = 0.01
) -> float:
    """
    Calcule le taux de détection à un taux de fausses alarmes donné.
    
    Args:
        scores: Scores de détection
        y_true: Labels réels (1 = positif)
        target_far: Taux de fausses alarmes cible
        
    Returns:
        detection_rate: Taux de détection au FAR cible
    """
    # Séparation des scores
    positive_scores = scores[y_true == 1]
    negative_scores = scores[y_true == 0]
    
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return 0.0
    
    # Calcul du seuil pour le FAR cible
    threshold = np.percentile(negative_scores, 100 * (1 - target_far))
    
    # Taux de détection
    detection_rate = np.mean(positive_scores > threshold)
    
    return detection_rate

class MetricsAggregator:
    """
    Agrégateur de métriques pour l'analyse statistique.
    """
    
    def __init__(self):
        self.metrics_list = []
    
    def add_metrics(
        self,
        metrics: Dict[str, float],
        run_id: Optional[str] = None
    ) -> None:
        """
        Ajoute des métriques.
        """
        metrics_with_id = metrics.copy()
        if run_id is not None:
            metrics_with_id['run_id'] = run_id
        self.metrics_list.append(metrics_with_id)
    
    def compute_statistics(self) -> pd.DataFrame:
        """
        Calcule les statistiques sur toutes les métriques.
        """
        df = pd.DataFrame(self.metrics_list)
        
        # Colonnes numériques uniquement
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Calcul des statistiques
        stats = pd.DataFrame({
            'mean': df[numeric_cols].mean(),
            'std': df[numeric_cols].std(),
            'min': df[numeric_cols].min(),
            'max': df[numeric_cols].max(),
            'median': df[numeric_cols].median(),
            'ci_lower': df[numeric_cols].apply(lambda x: stats.t.interval(0.95, len(x)-1)[0] if len(x) > 1 else x.iloc[0]),
            'ci_upper': df[numeric_cols].apply(lambda x: stats.t.interval(0.95, len(x)-1)[1] if len(x) > 1 else x.iloc[0])
        })
        
        return stats
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convertit en DataFrame."""
        return pd.DataFrame(self.metrics_list)
    
    def save(self, path: str) -> None:
        """Sauvegarde en CSV."""
        self.to_dataframe().to_csv(path, index=False)
        logger.info(f"Métriques sauvegardées: {path}")