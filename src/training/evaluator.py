"""
Évaluation complète des modèles de détection de deepfakes.
Fournit des métriques détaillées et des analyses de robustesse.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import logging
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Métriques d'évaluation."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    average_precision: float = 0.0
    specificity: float = 0.0
    matthews_corrcoef: float = 0.0
    inference_time: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convertit en dictionnaire."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc,
            'average_precision': self.average_precision,
            'specificity': self.specificity,
            'matthews_corrcoef': self.matthews_corrcoef,
            'inference_time': self.inference_time
        }

class ModelEvaluator:
    """
    Évaluateur de modèles pour la détection de deepfakes.
    """
    
    def __init__(
        self,
        model: nn.Module,
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
        test_loader: DataLoader,
        return_predictions: bool = False
    ) -> EvaluationMetrics:
        """
        Évalue le modèle sur un ensemble de test.
        """
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        inference_times = []
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    x = batch['frames']
                    y = batch['label']
                else:
                    x, y = batch
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Mesure du temps d'inférence
                start_time = time.time()
                logits = self.model(x)
                inference_time = time.time() - start_time
                
                # Probabilités
                probabilities = torch.softmax(logits, dim=1)
                
                # Prédictions
                predictions = logits.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                inference_times.append(inference_time)
        
        # Conversion en numpy
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        targets = np.array(all_targets)
        
        # Calcul des métriques
        metrics = self._compute_metrics(
            predictions,
            probabilities,
            targets,
            np.mean(inference_times)
        )
        
        # Visualisations
        if self.save_dir:
            self._plot_confusion_matrix(targets, predictions)
            self._plot_roc_curve(targets, probabilities)
            self._plot_precision_recall_curve(targets, probabilities)
        
        if return_predictions:
            return metrics, (predictions, probabilities, targets)
        
        return metrics
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        targets: np.ndarray,
        inference_time: float
    ) -> EvaluationMetrics:
        """
        Calcule toutes les métriques.
        """
        metrics = EvaluationMetrics()
        
        # Métriques de base
        metrics.accuracy = accuracy_score(targets, predictions)
        metrics.precision = precision_score(targets, predictions, zero_division=0)
        metrics.recall = recall_score(targets, predictions, zero_division=0)
        metrics.f1_score = f1_score(targets, predictions, zero_division=0)
        
        # Métriques avancées
        try:
            metrics.auc_roc = roc_auc_score(targets, probabilities)
            metrics.average_precision = average_precision_score(targets, probabilities)
        except ValueError:
            logger.warning("Impossible de calculer AUC (une seule classe présente)")
        
        # Spécificité
        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
        metrics.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Matthews correlation coefficient
        from sklearn.metrics import matthews_corrcoef
        metrics.matthews_corrcoef = matthews_corrcoef(targets, predictions)
        
        # Temps d'inférence
        metrics.inference_time = inference_time
        
        return metrics
    
    def _plot_confusion_matrix(
        self,
        targets: np.ndarray,
        predictions: np.ndarray
    ) -> None:
        """
        Affiche la matrice de confusion.
        """
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Réel', 'Fake'],
            yticklabels=['Réel', 'Fake']
        )
        plt.title('Matrice de Confusion')
        plt.xlabel('Prédictions')
        plt.ylabel('Vérité Terrain')
        
        plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray
    ) -> None:
        """
        Affiche la courbe ROC.
        """
        fpr, tpr, _ = roc_curve(targets, probabilities)
        auc = roc_auc_score(targets, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
        plt.xlabel('Taux de Faux Positifs')
        plt.ylabel('Taux de Vrais Positifs')
        plt.title('Courbe ROC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.save_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray
    ) -> None:
        """
        Affiche la courbe précision-rappel.
        """
        precision, recall, _ = precision_recall_curve(targets, probabilities)
        ap = average_precision_score(targets, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR (AP = {ap:.3f})')
        plt.xlabel('Rappel')
        plt.ylabel('Précision')
        plt.title('Courbe Précision-Rappel')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.save_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

class CrossDatasetEvaluator:
    """
    Évaluation cross-dataset pour tester la généralisation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        
    def evaluate_cross_dataset(
        self,
        datasets: Dict[str, DataLoader]
    ) -> Dict[str, Dict[str, float]]:
        """
        Évalue le modèle sur plusieurs datasets.
        """
        results = {}
        
        for dataset_name, dataloader in datasets.items():
            logger.info(f"Évaluation sur {dataset_name}")
            
            evaluator = ModelEvaluator(
                self.model,
                device=self.device
            )
            
            metrics = evaluator.evaluate(dataloader)
            results[dataset_name] = metrics.to_dict()
            
            logger.info(f"{dataset_name}: accuracy={metrics.accuracy:.4f}, f1={metrics.f1_score:.4f}")
        
        return results

class RobustnessEvaluator:
    """
    Évaluation de la robustesse aux perturbations.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        
    def evaluate_robustness(
        self,
        test_loader: DataLoader,
        perturbations: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Évalue la robustesse aux différentes perturbations.
        """
        results = {}
        
        # Évaluation sans perturbation
        clean_metrics = self._evaluate_clean(test_loader)
        results['clean'] = clean_metrics
        
        # Évaluation avec perturbations
        for perturbation in perturbations:
            perturbed_loader = self._apply_perturbation(
                test_loader,
                perturbation
            )
            
            perturbed_metrics = self._evaluate_perturbed(perturbed_loader)
            results[perturbation['name']] = perturbed_metrics
        
        return results
    
    def _evaluate_clean(
        self,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """Évalue sans perturbation."""
        evaluator = ModelEvaluator(self.model, device=self.device)
        metrics = evaluator.evaluate(test_loader)
        return metrics.to_dict()
    
    def _evaluate_perturbed(
        self,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """Évalue avec perturbation."""
        evaluator = ModelEvaluator(self.model, device=self.device)
        metrics = evaluator.evaluate(test_loader)
        return metrics.to_dict()
    
    def _apply_perturbation(
        self,
        test_loader: DataLoader,
        perturbation: Dict[str, Any]
    ) -> DataLoader:
        """
        Applique une perturbation au dataset.
        """
        # TODO: Implémenter les perturbations
        return test_loader

class AnomalyEvaluator:
    """
    Évaluation spécifique pour la détection d'anomalies.
    """
    
    def __init__(
        self,
        anomaly_detector: nn.Module,
        device: str = 'cuda'
    ):
        self.anomaly_detector = anomaly_detector
        self.device = device
        
    def evaluate_anomaly_detection(
        self,
        real_features: torch.Tensor,
        fake_features: torch.Tensor,
        contamination: float = 0.1
    ) -> Dict[str, float]:
        """
        Évalue la détection d'anomalies.
        """
        # Fit sur les vrais visages
        self.anomaly_detector.fit(real_features)
        
        # Scores sur les vrais et faux
        real_scores = self.anomaly_detector.score(real_features)
        fake_scores = self.anomaly_detector.score(fake_features)
        
        # Labels
        real_labels = torch.zeros_like(real_scores)
        fake_labels = torch.ones_like(fake_scores)
        
        # Prédictions
        real_pred, _ = self.anomaly_detector.predict(real_features)
        fake_pred, _ = self.anomaly_detector.predict(fake_features)
        
        # Métriques
        all_scores = torch.cat([real_scores, fake_scores]).numpy()
        all_labels = torch.cat([real_labels, fake_labels]).numpy()
        all_pred = torch.cat([real_pred, fake_pred]).numpy()
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_pred),
            'precision': precision_score(all_labels, all_pred),
            'recall': recall_score(all_labels, all_pred),
            'f1_score': f1_score(all_labels, all_pred),
            'auc_roc': roc_auc_score(all_labels, all_scores),
            'average_precision': average_precision_score(all_labels, all_scores)
        }
        
        return metrics