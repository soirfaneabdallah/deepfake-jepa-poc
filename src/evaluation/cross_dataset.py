"""
Évaluation cross-dataset pour tester la généralisation.
Mesure la capacité du modèle à détecter des deepfakes de sources inconnues.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .metrics import compute_classification_metrics

logger = logging.getLogger(__name__)

@dataclass
class GeneralizationMetrics:
    """Métriques de généralisation."""
    source_dataset: str = ""
    target_dataset: str = ""
    accuracy: float = 0.0
    auc_roc: float = 0.0
    f1_score: float = 0.0
    performance_drop: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'source_dataset': self.source_dataset,
            'target_dataset': self.target_dataset,
            'accuracy': self.accuracy,
            'auc_roc': self.auc_roc,
            'f1_score': self.f1_score,
            'performance_drop': self.performance_drop
        }

class CrossDatasetEvaluator:
    """
    Évaluateur cross-dataset pour mesurer la généralisation.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda',
        save_dir: Optional[str] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(
        self,
        datasets: Dict[str, torch.utils.data.DataLoader],
        source_dataset: Optional[str] = None
    ) -> List[GeneralizationMetrics]:
        """
        Évalue le modèle sur plusieurs datasets.
        
        Args:
            datasets: Dict {nom: dataloader}
            source_dataset: Dataset d'entraînement
            
        Returns:
            Liste des métriques de généralisation
        """
        results = []
        
        # Évaluation sur chaque dataset
        for dataset_name, dataloader in datasets.items():
            logger.info(f"Évaluation sur {dataset_name}")
            
            # Prédictions
            y_true, y_pred, y_scores = self._predict(dataloader)
            
            # Métriques
            metrics = compute_classification_metrics(
                y_true,
                y_pred,
                y_scores
            )
            
            # Création des métriques de généralisation
            gen_metrics = GeneralizationMetrics(
                source_dataset=source_dataset or "unknown",
                target_dataset=dataset_name,
                accuracy=metrics.accuracy,
                auc_roc=metrics.auc_roc,
                f1_score=metrics.f1_score
            )
            
            results.append(gen_metrics)
        
        # Calcul de la perte de performance
        if source_dataset and source_dataset in datasets:
            source_metrics = next(
                r for r in results if r.target_dataset == source_dataset
            )
            
            for result in results:
                result.performance_drop = source_metrics.accuracy - result.accuracy
        
        return results
    
    def _predict(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prédit sur un dataloader.
        """
        self.model.eval()
        all_preds = []
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x = batch['frames']
                    y = batch['label']
                else:
                    x, y = batch
                
                x = x.to(self.device)
                
                # Forward pass
                logits = self.model(x)
                probabilities = torch.softmax(logits, dim=1)
                
                predictions = logits.argmax(dim=1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_scores.extend(probabilities[:, 1].cpu().numpy())
                all_labels.extend(y.numpy())
        
        return (
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_scores)
        )
    
    def create_report(
        self,
        results: List[GeneralizationMetrics],
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Crée un rapport cross-dataset.
        """
        df = pd.DataFrame([r.to_dict() for r in results])
        
        if save_path or self.save_dir:
            path = save_path or self.save_dir / 'cross_dataset_results.csv'
            df.to_csv(path, index=False)
            logger.info(f"Rapport sauvegardé: {path}")
        
        return df
    
    def plot_generalization_matrix(
        self,
        results: List[GeneralizationMetrics],
        save_path: Optional[str] = None
    ) -> None:
        """
        Affiche la matrice de généralisation.
        """
        # Création de la matrice
        datasets = list(set(r.target_dataset for r in results))
        matrix = np.zeros((len(datasets), len(datasets)))
        
        for result in results:
            if result.source_dataset in datasets:
                i = datasets.index(result.source_dataset)
                j = datasets.index(result.target_dataset)
                matrix[i, j] = result.accuracy
        
        # Affichage
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.3f',
            cmap='YlGnBu',
            xticklabels=datasets,
            yticklabels=datasets
        )
        
        plt.title('Matrice de Généralisation Cross-Dataset')
        plt.xlabel('Dataset Cible')
        plt.ylabel('Dataset Source')
        
        if save_path or self.save_dir:
            path = save_path or self.save_dir / 'generalization_matrix.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
        
        plt.close()

class DatasetBiasAnalyzer:
    """
    Analyseur de biais entre datasets.
    """
    
    def __init__(self, feature_extractor: Optional[callable] = None):
        self.feature_extractor = feature_extractor
    
    def compute_dataset_shift(
        self,
        features_source: np.ndarray,
        features_target: np.ndarray
    ) -> Dict[str, float]:
        """
        Mesure le décalage entre deux datasets.
        """
        # Statistiques
        mean_source = features_source.mean(axis=0)
        mean_target = features_target.mean(axis=0)
        
        # Distance de Bhattacharyya
        from scipy.spatial.distance import bhattacharyya
        
        std_source = features_source.std(axis=0) + 1e-8
        std_target = features_target.std(axis=0) + 1e-8
        
        # Distance de Wasserstein simplifiée
        wasserstein = np.linalg.norm(mean_source - mean_target)
        
        # Divergence KL
        from scipy.stats import entropy
        kl_divergence = entropy(
            mean_source + 1e-8,
            mean_target + 1e-8
        )
        
        return {
            'mean_shift': wasserstein,
            'kl_divergence': kl_divergence,
            'variance_ratio': np.mean(std_target / std_source)
        }

class CrossDatasetReport:
    """
    Rapport complet d'évaluation cross-dataset.
    """
    
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_report(
        self,
        results: List[GeneralizationMetrics],
        dataset_names: List[str]
    ) -> str:
        """
        Génère un rapport complet.
        """
        report_path = self.save_dir / 'cross_dataset_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Rapport d'Évaluation Cross-Dataset\n\n")
            f.write("## Résumé\n\n")
            f.write(f"- Datasets évalués : {', '.join(dataset_names)}\n")
            f.write(f"- Nombre de combinaisons : {len(results)}\n\n")
            
            f.write("## Tableau des Performances\n\n")
            f.write("| Source | Cible | Accuracy | AUC | F1 | Drop |\n")
            f.write("|--------|-------|----------|-----|----|----|\n")
            
            for result in results:
                f.write(
                    f"| {result.source_dataset} | {result.target_dataset} | "
                    f"{result.accuracy:.3f} | {result.auc_roc:.3f} | "
                    f"{result.f1_score:.3f} | {result.performance_drop:.3f} |\n"
                )
            
            f.write("\n## Analyse\n\n")
            
            # Meilleure combinaison
            best = max(results, key=lambda r: r.accuracy)
            f.write(f"- Meilleure performance : {best.source_dataset} → {best.target_dataset} "
                    f"(Accuracy: {best.accuracy:.3f})\n")
            
            # Pire combinaison
            worst = min(results, key=lambda r: r.accuracy)
            f.write(f"- Pire performance : {worst.source_dataset} → {worst.target_dataset} "
                    f"(Accuracy: {worst.accuracy:.3f})\n")
            
            # Drop moyen
            avg_drop = np.mean([r.performance_drop for r in results])
            f.write(f"- Perte de performance moyenne : {avg_drop:.3f}\n")
        
        return str(report_path)