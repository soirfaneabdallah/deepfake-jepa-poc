"""
Métriques pour évaluer l'oubli catastrophique et le transfert.
Fournit des mesures quantitatives de la performance en apprentissage continu.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ForgettingMetric:
    """Métrique d'oubli catastrophique."""
    average_forgetting: float = 0.0
    max_forgetting: float = 0.0
    forgetting_per_task: List[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'average_forgetting': self.average_forgetting,
            'max_forgetting': self.max_forgetting,
            'forgetting_per_task': self.forgetting_per_task
        }

@dataclass
class BackwardTransfer:
    """Transfert négatif (impact sur les tâches précédentes)."""
    average_transfer: float = 0.0
    transfer_per_task: List[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'average_transfer': self.average_transfer,
            'transfer_per_task': self.transfer_per_task
        }

@dataclass
class ForwardTransfer:
    """Transfert positif (impact sur les tâches futures)."""
    average_transfer: float = 0.0
    transfer_per_task: List[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'average_transfer': self.average_transfer,
            'transfer_per_task': self.transfer_per_task
        }

@dataclass
class ContinualMetrics:
    """Métriques complètes d'apprentissage continu."""
    forgetting: ForgettingMetric = None
    backward_transfer: BackwardTransfer = None
    forward_transfer: ForwardTransfer = None
    final_accuracy: float = 0.0
    average_accuracy: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'forgetting': self.forgetting.to_dict() if self.forgetting else None,
            'backward_transfer': self.backward_transfer.to_dict() if self.backward_transfer else None,
            'forward_transfer': self.forward_transfer.to_dict() if self.forward_transfer else None,
            'final_accuracy': self.final_accuracy,
            'average_accuracy': self.average_accuracy
        }

def compute_forgetting(
    performance_matrix: np.ndarray
) -> ForgettingMetric:
    """
    Calcule l'oubli catastrophique.
    
    Args:
        performance_matrix: (T, T) - Matrice des performances
            performance_matrix[i, j] = performance sur la tâche j après
            l'entraînement sur la tâche i
        
    Returns:
        ForgettingMetric
    """
    num_tasks = performance_matrix.shape[0]
    forgetting_per_task = []
    
    for task_id in range(num_tasks - 1):
        # Performance initiale sur la tâche
        initial_performance = performance_matrix[task_id, task_id]
        
        # Performance finale sur la tâche
        final_performance = performance_matrix[-1, task_id]
        
        # Oubli = performance initiale - performance finale
        forgetting = max(0.0, initial_performance - final_performance)
        forgetting_per_task.append(forgetting)
    
    # Métriques agrégées
    average_forgetting = np.mean(forgetting_per_task) if forgetting_per_task else 0.0
    max_forgetting = np.max(forgetting_per_task) if forgetting_per_task else 0.0
    
    return ForgettingMetric(
        average_forgetting=average_forgetting,
        max_forgetting=max_forgetting,
        forgetting_per_task=forgetting_per_task
    )

def compute_backward_transfer(
    performance_matrix: np.ndarray
) -> BackwardTransfer:
    """
    Calcule le transfert négatif (Backward Transfer).
    
    BWT = moyenne sur i<j de (R_{i,j} - R_{j,j})
    où R_{i,j} est la performance sur la tâche j après l'entraînement sur i
    """
    num_tasks = performance_matrix.shape[0]
    transfer_per_task = []
    
    for task_id in range(1, num_tasks):
        # Performance après l'entraînement sur la tâche
        performance_after = performance_matrix[task_id, :task_id]
        
        # Performance initiale
        performance_initial = performance_matrix[:task_id, :task_id].diagonal()
        
        # Transfert négatif
        transfer = np.mean(performance_after - performance_initial)
        transfer_per_task.append(transfer)
    
    average_transfer = np.mean(transfer_per_task) if transfer_per_task else 0.0
    
    return BackwardTransfer(
        average_transfer=average_transfer,
        transfer_per_task=transfer_per_task
    )

def compute_forward_transfer(
    performance_matrix: np.ndarray,
    baseline_performance: Optional[np.ndarray] = None
) -> ForwardTransfer:
    """
    Calcule le transfert positif (Forward Transfer).
    
    FWT = moyenne sur i<j de (R_{i,j} - R_{baseline,j})
    où R_{baseline,j} est la performance sur la tâche j sans entraînement préalable
    """
    num_tasks = performance_matrix.shape[0]
    
    if baseline_performance is None:
        # Utiliser la performance initiale comme baseline
        baseline_performance = performance_matrix.diagonal()
    
    transfer_per_task = []
    
    for task_id in range(num_tasks):
        # Performance sur la tâche après entraînement sur les tâches précédentes
        performance_with_transfer = performance_matrix[task_id - 1, task_id] if task_id > 0 else baseline_performance[task_id]
        
        # Transfert positif
        transfer = performance_with_transfer - baseline_performance[task_id]
        transfer_per_task.append(transfer)
    
    average_transfer = np.mean(transfer_per_task) if transfer_per_task else 0.0
    
    return ForwardTransfer(
        average_transfer=average_transfer,
        transfer_per_task=transfer_per_task
    )

def compute_all_metrics(
    performance_matrix: np.ndarray,
    baseline_performance: Optional[np.ndarray] = None
) -> ContinualMetrics:
    """
    Calcule toutes les métriques d'apprentissage continu.
    """
    # Oubli
    forgetting = compute_forgetting(performance_matrix)
    
    # Transfert négatif
    backward_transfer = compute_backward_transfer(performance_matrix)
    
    # Transfert positif
    forward_transfer = compute_forward_transfer(
        performance_matrix,
        baseline_performance
    )
    
    # Performances finales
    final_accuracy = performance_matrix[-1, -1]
    average_accuracy = performance_matrix.diagonal().mean()
    
    return ContinualMetrics(
        forgetting=forgetting,
        backward_transfer=backward_transfer,
        forward_transfer=forward_transfer,
        final_accuracy=final_accuracy,
        average_accuracy=average_accuracy
    )

class ContinualMetricsTracker:
    """
    Suivi des métriques d'apprentissage continu pendant l'entraînement.
    """
    
    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        self.performance_matrix = np.zeros((num_tasks, num_tasks))
        self.current_task = -1
        
    def update(
        self,
        task_id: int,
        performance: Dict[int, float]
    ) -> None:
        """
        Met à jour la matrice de performance.
        
        Args:
            task_id: Tâche courante
            performance: Dict mapping task_id -> performance
        """
        self.current_task = task_id
        
        for eval_task_id, perf in performance.items():
            self.performance_matrix[task_id, eval_task_id] = perf
    
    def get_metrics(self) -> ContinualMetrics:
        """
        Calcule les métriques complètes.
        """
        # Remplir les performances non évaluées
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                if i < j and self.performance_matrix[i, j] == 0:
                    # Utiliser la performance diagonale comme estimation
                    self.performance_matrix[i, j] = self.performance_matrix[j, j]
        
        return compute_all_metrics(self.performance_matrix)
    
    def plot_performance_matrix(self, save_path: Optional[str] = None) -> None:
        """
        Visualise la matrice de performance.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.performance_matrix,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=[f'Tâche {i}' for i in range(self.num_tasks)],
            yticklabels=[f'Tâche {i}' for i in range(self.num_tasks)]
        )
        plt.title('Matrice de Performance en Apprentissage Continu')
        plt.xlabel('Tâche évaluée')
        plt.ylabel('Tâche entraînée')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()