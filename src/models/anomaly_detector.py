"""
Détection d'anomalies dans l'espace latent v-JEPA.
Implémente plusieurs méthodes pour identifier les deepfakes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Union
from abc import ABC, abstractmethod
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import mahalanobis
import logging

logger = logging.getLogger(__name__)

class AnomalyDetector(ABC):
    """
    Classe abstraite pour les détecteurs d'anomalies.
    """
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.is_fitted = False
        self.threshold = None
        
    @abstractmethod
    def fit(self, features: torch.Tensor) -> None:
        """
        Ajuste le détecteur sur les features des vrais visages.
        
        Args:
            features: (N, D) - Features latentes des vrais visages
        """
        pass
    
    @abstractmethod
    def score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Calcule le score d'anomalie.
        
        Args:
            features: (N, D) - Features à évaluer
            
        Returns:
            scores: (N,) - Scores d'anomalie
        """
        pass
    
    def predict(
        self,
        features: torch.Tensor,
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prédit si les features sont anormales.
        
        Returns:
            predictions: (N,) - 0 = normal, 1 = anomalie
            scores: (N,) - Scores d'anomalie
        """
        scores = self.score(features)
        
        if threshold is None:
            threshold = self.threshold if self.threshold is not None else self._compute_threshold(scores)
        
        predictions = (scores > threshold).long()
        
        return predictions, scores
    
    def _compute_threshold(self, scores: torch.Tensor) -> float:
        """
        Calcule le seuil basé sur le percentile.
        """
        if self.contamination is not None:
            threshold = np.percentile(
                scores.cpu().numpy(),
                100 * (1 - self.contamination)
            )
        else:
            # Seuil basé sur la médiane + 3*MAD
            median = torch.median(scores)
            mad = torch.median(torch.abs(scores - median))
            threshold = (median + 3 * 1.4826 * mad).item()
        
        return threshold
    
    def save(self, path: str) -> None:
        """Sauvegarde le détecteur."""
        torch.save({
            'threshold': self.threshold,
            'is_fitted': self.is_fitted,
            'contamination': self.contamination
        }, path)
    
    def load(self, path: str) -> None:
        """Charge le détecteur."""
        checkpoint = torch.load(path)
        self.threshold = checkpoint['threshold']
        self.is_fitted = checkpoint['is_fitted']
        self.contamination = checkpoint['contamination']

class MahalanobisDetector(AnomalyDetector):
    """
    Détecteur basé sur la distance de Mahalanobis.
    Suppose une distribution gaussienne des vrais visages.
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        covariance_type: str = 'empirical',  # empirical, robust, shrunk
        regularization: float = 1e-6
    ):
        super().__init__(contamination)
        self.covariance_type = covariance_type
        self.regularization = regularization
        
        self.mean = None
        self.cov_inv = None
        self.cov = None
        
    def fit(self, features: torch.Tensor) -> None:
        """
        Ajuste la distribution gaussienne sur les features.
        """
        features_np = features.cpu().numpy()
        
        # Calcul de la moyenne
        self.mean = torch.mean(features, dim=0)
        
        # Calcul de la covariance
        if self.covariance_type == 'empirical':
            cov = np.cov(features_np.T)
        elif self.covariance_type == 'robust':
            # Utilisation de Minimum Covariance Determinant
            mcd = MinCovDet(random_state=42)
            mcd.fit(features_np)
            cov = mcd.covariance_
        elif self.covariance_type == 'shrunk':
            # Covariance avec shrinkage
            ec = EmpiricalCovariance()
            ec.fit(features_np)
            cov = ec.covariance_
            # Shrinkage manuel
            alpha = 0.1
            cov = (1 - alpha) * cov + alpha * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
        else:
            raise ValueError(f"Type de covariance non supporté: {self.covariance_type}")
        
        # Régularisation
        cov += self.regularization * np.eye(cov.shape[0])
        
        # Inversion de la covariance
        self.cov = torch.from_numpy(cov).float()
        self.cov_inv = torch.linalg.inv(self.cov)
        
        self.is_fitted = True
        
        # Calcul du seuil
        scores = self.score(features)
        self.threshold = self._compute_threshold(scores)
        
        logger.info(f"Détecteur Mahalanobis ajusté. Seuil: {self.threshold:.4f}")
    
    def score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Calcule la distance de Mahalanobis.
        """
        if not self.is_fitted:
            raise RuntimeError("Le détecteur n'est pas ajusté")
        
        # Calcul de la distance
        diff = features - self.mean.to(features.device)
        cov_inv = self.cov_inv.to(features.device)
        
        scores = torch.sqrt(
            torch.einsum('bi,ij,bj->b', diff, cov_inv, diff)
        )
        
        return scores

class OneClassSVMDetector(AnomalyDetector):
    """
    Détecteur One-Class SVM.
    Apprend une frontière autour des vrais visages.
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        kernel: str = 'rbf',
        gamma: Union[str, float] = 'scale',
        nu: float = 0.1
    ):
        super().__init__(contamination)
        
        self.svm = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            nu=nu
        )
    
    def fit(self, features: torch.Tensor) -> None:
        """
        Ajuste le One-Class SVM.
        """
        features_np = features.cpu().numpy()
        self.svm.fit(features_np)
        
        self.is_fitted = True
        
        # Calcul du seuil
        scores = self.score(features)
        self.threshold = self._compute_threshold(scores)
        
        logger.info(f"One-Class SVM ajusté. Seuil: {self.threshold:.4f}")
    
    def score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Calcule le score de décision.
        """
        if not self.is_fitted:
            raise RuntimeError("Le détecteur n'est pas ajusté")
        
        features_np = features.cpu().numpy()
        
        # Score négatif = anomalie
        scores = -self.svm.decision_function(features_np)
        
        return torch.from_numpy(scores).float()

class IsolationForestDetector(AnomalyDetector):
    """
    Détecteur Isolation Forest.
    Isole les anomalies dans des partitions aléatoires.
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: Union[int, str] = 'auto'
    ):
        super().__init__(contamination)
        
        self.forest = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
    
    def fit(self, features: torch.Tensor) -> None:
        """
        Ajuste l'Isolation Forest.
        """
        features_np = features.cpu().numpy()
        self.forest.fit(features_np)
        
        self.is_fitted = True
        
        # Calcul du seuil
        scores = self.score(features)
        self.threshold = self._compute_threshold(scores)
        
        logger.info(f"Isolation Forest ajusté. Seuil: {self.threshold:.4f}")
    
    def score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Calcule le score d'anomalie.
        """
        if not self.is_fitted:
            raise RuntimeError("Le détecteur n'est pas ajusté")
        
        features_np = features.cpu().numpy()
        
        # Score négatif = anomalie
        scores = -self.forest.score_samples(features_np)
        
        return torch.from_numpy(scores).float()

class DeepSVDDDetector(AnomalyDetector):
    """
    Deep Support Vector Data Description.
    Apprend une hypersphère minimale autour des vrais visages.
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 32,
        nu: float = 0.1
    ):
        super().__init__(contamination)
        
        self.nu = nu
        self.latent_dim = latent_dim
        
        # Réseau pour la projection
        layers = []
        input_dim = 512  # Dimension des features v-JEPA
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, latent_dim))
        
        self.network = nn.Sequential(*layers)
        self.center = None
        
    def fit(
        self,
        features: torch.Tensor,
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        device: str = 'cuda'
    ) -> None:
        """
        Ajuste le Deep SVDD sur les features.
        """
        self.network = self.network.to(device)
        features = features.to(device)
        
        # Optimiseur
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Entraînement
        self.network.train()
        for epoch in range(epochs):
            # Shuffle
            indices = torch.randperm(features.size(0))
            
            for i in range(0, features.size(0), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch = features[batch_indices]
                
                # Forward pass
                projected = self.network(batch)
                
                # Calcul de la perte
                if self.center is None:
                    # Initialisation du centre
                    self.center = projected.mean(dim=0).detach()
                
                # Distance au centre
                dist = torch.sum((projected - self.center) ** 2, dim=1)
                
                # Perte SVDD
                loss = torch.mean(dist)
                
                # Régularisation
                if self.nu > 0:
                    loss += self.nu * torch.sum(
                        torch.norm(self.network[0].weight, dim=1) ** 2
                    )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0:
                logger.debug(f"Deep SVDD - Époque {epoch}: loss={loss.item():.4f}")
        
        # Mise à jour du centre
        self.network.eval()
        with torch.no_grad():
            projected = self.network(features)
            self.center = projected.mean(dim=0)
        
        self.is_fitted = True
        
        # Calcul du seuil
        scores = self.score(features.cpu())
        self.threshold = self._compute_threshold(scores)
        
        logger.info(f"Deep SVDD ajusté. Seuil: {self.threshold:.4f}")
    
    def score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Calcule la distance au centre de l'hypersphère.
        """
        if not self.is_fitted:
            raise RuntimeError("Le détecteur n'est pas ajusté")
        
        self.network.eval()
        with torch.no_grad():
            projected = self.network(features)
            scores = torch.sum(
                (projected - self.center.to(features.device)) ** 2,
                dim=1
            )
        
        return scores

class EnsembleAnomalyDetector(AnomalyDetector):
    """
    Ensemble de détecteurs d'anomalies.
    Combine plusieurs méthodes pour plus de robustesse.
    """
    
    def __init__(
        self,
        detectors: List[AnomalyDetector],
        weights: Optional[List[float]] = None,
        combination: str = 'mean'  # mean, max, min, weighted
    ):
        super().__init__(contamination=0.1)
        
        self.detectors = detectors
        self.weights = weights if weights is not None else [1.0] * len(detectors)
        self.combination = combination
        
    def fit(self, features: torch.Tensor) -> None:
        """
        Ajuste tous les détecteurs.
        """
        for detector in self.detectors:
            detector.fit(features)
        
        self.is_fitted = True
        
        # Calcul du seuil
        scores = self.score(features)
        self.threshold = self._compute_threshold(scores)
    
    def score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Combine les scores de tous les détecteurs.
        """
        all_scores = []
        for detector in self.detectors:
            scores = detector.score(features)
            all_scores.append(scores)
        
        # Stack des scores
        stacked_scores = torch.stack(all_scores)  # (D, N)
        
        # Normalisation des scores
        normalized_scores = []
        for scores in stacked_scores:
            if scores.std() > 0:
                normalized = (scores - scores.mean()) / scores.std()
            else:
                normalized = scores
            normalized_scores.append(normalized)
        
        normalized_scores = torch.stack(normalized_scores)
        
        # Combinaison
        if self.combination == 'mean':
            combined = normalized_scores.mean(dim=0)
        elif self.combination == 'max':
            combined = normalized_scores.max(dim=0)[0]
        elif self.combination == 'min':
            combined = normalized_scores.min(dim=0)[0]
        elif self.combination == 'weighted':
            weights = torch.tensor(self.weights).unsqueeze(1)
            combined = (normalized_scores * weights).sum(dim=0) / weights.sum()
        else:
            raise ValueError(f"Combinaison non supportée: {self.combination}")
        
        return combined

def create_anomaly_detector(
    method: str = 'mahalanobis',
    **kwargs
) -> AnomalyDetector:
    """
    Factory pour créer un détecteur d'anomalies.
    """
    detectors = {
        'mahalanobis': MahalanobisDetector,
        'one_class_svm': OneClassSVMDetector,
        'isolation_forest': IsolationForestDetector,
        'deep_svdd': DeepSVDDDetector
    }
    
    if method not in detectors:
        raise ValueError(f"Détecteur non supporté: {method}")
    
    return detectors[method](**kwargs)