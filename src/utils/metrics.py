"""
Utilitaires de calcul de métriques.
Fournit des fonctions simples et des trackers de métriques.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)

def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Calcule l'accuracy.
    """
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0

def compute_precision_recall_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    average: str = 'binary'
) -> Tuple[float, float, float]:
    """
    Calcule la précision, le rappel et le F1-score.
    """
    predictions_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    precision = precision_score(targets_np, predictions_np, average=average, zero_division=0)
    recall = recall_score(targets_np, predictions_np, average=average, zero_division=0)
    f1 = f1_score(targets_np, predictions_np, average=average, zero_division=0)
    
    return precision, recall, f1

def compute_auc(
    probabilities: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Calcule l'AUC-ROC.
    """
    try:
        probabilities_np = probabilities.cpu().numpy()
        targets_np = targets.cpu().numpy()
        return roc_auc_score(targets_np, probabilities_np)
    except ValueError:
        logger.warning("Impossible de calculer l'AUC (une seule classe présente)")
        return 0.0

def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 2
) -> np.ndarray:
    """
    Calcule la matrice de confusion.
    """
    predictions_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    return confusion_matrix(targets_np, predictions_np, labels=range(num_classes))

def compute_classification_report(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_names: List[str] = ['real', 'fake']
) -> str:
    """
    Calcule le rapport de classification.
    """
    predictions_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    return classification_report(
        targets_np,
        predictions_np,
        target_names=target_names
    )

class RunningAverage:
    """
    Moyenne mobile pour le suivi des métriques.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.sum = 0.0
        self.count = 0
    
    def update(self, value: float) -> None:
        """Met à jour la moyenne."""
        self.values.append(value)
        self.sum += value
        self.count += 1
        
        if len(self.values) == self.window_size:
            self.sum -= self.values[0]
    
    @property
    def average(self) -> float:
        """Retourne la moyenne actuelle."""
        if len(self.values) == 0:
            return 0.0
        return self.sum / len(self.values)
    
    @property
    def value(self) -> float:
        """Retourne la dernière valeur."""
        return self.values[-1] if self.values else 0.0
    
    def reset(self) -> None:
        """Réinitialise la moyenne."""
        self.values.clear()
        self.sum = 0.0
        self.count = 0

class MetricTracker:
    """
    Tracker de métriques pour l'entraînement.
    """
    
    def __init__(self):
        self.metrics = {}
        self.running_averages = {}
    
    def update(
        self,
        metrics: Dict[str, float],
        phase: str = 'train'
    ) -> None:
        """
        Met à jour les métriques.
        """
        for key, value in metrics.items():
            full_key = f"{phase}_{key}"
            
            if full_key not in self.metrics:
                self.metrics[full_key] = []
                self.running_averages[full_key] = RunningAverage()
            
            self.metrics[full_key].append(value)
            self.running_averages[full_key].update(value)
    
    def get_metric(self, key: str, phase: Optional[str] = None) -> List[float]:
        """
        Récupère l'historique d'une métrique.
        """
        if phase:
            key = f"{phase}_{key}"
        
        return self.metrics.get(key, [])
    
    def get_running_average(self, key: str, phase: Optional[str] = None) -> float:
        """
        Récupère la moyenne mobile d'une métrique.
        """
        if phase:
            key = f"{phase}_{key}"
        
        return self.running_averages.get(key, RunningAverage()).average
    
    def get_all_metrics(self) -> Dict[str, List[float]]:
        """
        Récupère toutes les métriques.
        """
        return self.metrics
    
    def get_summary(self) -> Dict[str, float]:
        """
        Récupère un résumé des métriques.
        """
        summary = {}
        
        for key, values in self.metrics.items():
            if values:
                summary[f"{key}_mean"] = np.mean(values)
                summary[f"{key}_std"] = np.std(values)
                summary[f"{key}_min"] = np.min(values)
                summary[f"{key}_max"] = np.max(values)
                summary[f"{key}_last"] = values[-1]
        
        return summary
    
    def reset(self) -> None:
        """
        Réinitialise toutes les métriques.
        """
        self.metrics.clear()
        self.running_averages.clear()
    
    def save(self, path: str) -> None:
        """
        Sauvegarde les métriques en JSON.
        """
        import json
        
        # Conversion des valeurs en types Python natifs
        metrics_serializable = {}
        for key, values in self.metrics.items():
            metrics_serializable[key] = [
                float(v) if isinstance(v, (int, float, np.number)) else v
                for v in values
            ]
        
        with open(path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        logger.info(f"Métriques sauvegardées: {path}")
    
    def load(self, path: str) -> None:
        """
        Charge les métriques depuis JSON.
        """
        import json
        
        with open(path, 'r') as f:
            self.metrics = json.load(f)
        
        logger.info(f"Métriques chargées: {path}")