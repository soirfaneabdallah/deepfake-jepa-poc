"""
Tests de robustesse pour la détection de deepfakes.
Évalue la performance face aux perturbations et attaques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from .metrics import compute_classification_metrics

logger = logging.getLogger(__name__)

@dataclass
class RobustnessReport:
    """Rapport de robustesse."""
    perturbation_name: str = ""
    severity: float = 0.0
    accuracy: float = 0.0
    auc_roc: float = 0.0
    f1_score: float = 0.0
    performance_drop: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'perturbation': self.perturbation_name,
            'severity': self.severity,
            'accuracy': self.accuracy,
            'auc_roc': self.auc_roc,
            'f1_score': self.f1_score,
            'performance_drop': self.performance_drop
        }

class RobustnessEvaluator:
    """
    Évaluateur de robustesse général.
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
        clean_loader: torch.utils.data.DataLoader,
        perturbed_loaders: Dict[str, torch.utils.data.DataLoader]
    ) -> List[RobustnessReport]:
        """
        Évalue la robustesse aux perturbations.
        """
        # Performance sur données propres
        clean_metrics = self._evaluate_loader(clean_loader)
        
        results = []
        
        for name, loader in perturbed_loaders.items():
            perturbed_metrics = self._evaluate_loader(loader)
            
            report = RobustnessReport(
                perturbation_name=name,
                severity=0.0,
                accuracy=perturbed_metrics.accuracy,
                auc_roc=perturbed_metrics.auc_roc,
                f1_score=perturbed_metrics.f1_score,
                performance_drop=clean_metrics.accuracy - perturbed_metrics.accuracy
            )
            
            results.append(report)
        
        return results
    
    def _evaluate_loader(
        self,
        loader: torch.utils.data.DataLoader
    ) -> Any:
        """
        Évalue un dataloader.
        """
        self.model.eval()
        all_preds = []
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict):
                    x = batch['frames']
                    y = batch['label']
                else:
                    x, y = batch
                
                x = x.to(self.device)
                logits = self.model(x)
                probabilities = torch.softmax(logits, dim=1)
                predictions = logits.argmax(dim=1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_scores.extend(probabilities[:, 1].cpu().numpy())
                all_labels.extend(y.numpy())
        
        return compute_classification_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_scores)
        )

class AdversarialEvaluator:
    """
    Évaluateur d'attaques adversariales.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
    
    def fgsm_attack(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        epsilon: float = 0.03
    ) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM).
        """
        inputs = inputs.clone().detach().requires_grad_(True)
        
        # Forward pass
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Perturbation
        perturbation = epsilon * inputs.grad.sign()
        adversarial_inputs = inputs + perturbation
        adversarial_inputs = torch.clamp(adversarial_inputs, 0, 1)
        
        return adversarial_inputs
    
    def pgd_attack(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        num_steps: int = 10
    ) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD).
        """
        adversarial_inputs = inputs.clone().detach()
        
        for _ in range(num_steps):
            adversarial_inputs = adversarial_inputs.clone().detach().requires_grad_(True)
            
            # Forward pass
            logits = self.model(adversarial_inputs)
            loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Mise à jour
            with torch.no_grad():
                adversarial_inputs = adversarial_inputs + alpha * adversarial_inputs.grad.sign()
                
                # Projection
                perturbation = torch.clamp(
                    adversarial_inputs - inputs,
                    -epsilon,
                    epsilon
                )
                adversarial_inputs = torch.clamp(
                    inputs + perturbation,
                    0,
                    1
                )
        
        return adversarial_inputs

class PerturbationEvaluator:
    """
    Évaluateur de perturbations classiques.
    """
    
    def __init__(self):
        pass
    
    def add_gaussian_noise(
        self,
        inputs: torch.Tensor,
        std: float = 0.1
    ) -> torch.Tensor:
        """
        Ajoute du bruit gaussien.
        """
        noise = torch.randn_like(inputs) * std
        perturbed = inputs + noise
        return torch.clamp(perturbed, 0, 1)
    
    def add_salt_pepper_noise(
        self,
        inputs: torch.Tensor,
        amount: float = 0.05
    ) -> torch.Tensor:
        """
        Ajoute du bruit sel et poivre.
        """
        mask = torch.rand_like(inputs) < amount
        salt = torch.rand_like(inputs) > 0.5
        
        perturbed = inputs.clone()
        perturbed[mask & salt] = 1.0
        perturbed[mask & ~salt] = 0.0
        
        return perturbed
    
    def blur(
        self,
        inputs: torch.Tensor,
        kernel_size: int = 5
    ) -> torch.Tensor:
        """
        Applique un flou gaussien.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Création du noyau gaussien
        sigma = kernel_size / 6
        kernel = self._gaussian_kernel(kernel_size, sigma)
        kernel = kernel.to(inputs.device)
        
        # Application du flou
        if inputs.dim() == 5:  # Vidéo
            B, C, T, H, W = inputs.shape
            inputs_reshaped = inputs.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
            blurred = F.conv2d(
                inputs_reshaped,
                kernel.expand(C, 1, kernel_size, kernel_size),
                padding=kernel_size//2,
                groups=C
            )
            blurred = blurred.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        else:  # Image
            blurred = F.conv2d(
                inputs,
                kernel.expand(inputs.size(1), 1, kernel_size, kernel_size),
                padding=kernel_size//2,
                groups=inputs.size(1)
            )
        
        return blurred
    
    def _gaussian_kernel(
        self,
        size: int,
        sigma: float
    ) -> torch.Tensor:
        """Crée un noyau gaussien."""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        x, y = torch.meshgrid(coords, coords)
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

class CompressionRobustness:
    """
    Évaluateur de robustesse à la compression.
    """
    
    def __init__(self):
        pass
    
    def jpeg_compress(
        self,
        inputs: torch.Tensor,
        quality: int = 50
    ) -> torch.Tensor:
        """
        Applique une compression JPEG.
        """
        # Conversion en numpy pour cv2
        if isinstance(inputs, torch.Tensor):
            inputs_np = inputs.cpu().numpy()
        
        # Compression JPEG
        compressed = []
        for img in inputs_np:
            # Conversion en uint8
            img = (img * 255).astype(np.uint8)
            
            # Compression
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', img.transpose(1, 2, 0), encode_param)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            
            # Conversion retour
            decoded = decoded.transpose(2, 0, 1) / 255.0
            compressed.append(decoded)
        
        return torch.from_numpy(np.stack(compressed))
    
    def video_compression(
        self,
        inputs: torch.Tensor,
        bitrate: str = '500k'
    ) -> torch.Tensor:
        """
        Simule une compression vidéo.
        """
        # TODO: Implémenter avec ffmpeg
        return inputs

class NoiseRobustness:
    """
    Évaluateur de robustesse au bruit.
    """
    
    def __init__(self):
        self.perturbation_evaluator = PerturbationEvaluator()
    
    def evaluate_noise_levels(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2]
    ) -> Dict[float, float]:
        """
        Évalue la robustesse à différents niveaux de bruit.
        """
        model.eval()
        results = {}
        
        for noise_level in noise_levels:
            # Ajout de bruit
            noisy_inputs = self.perturbation_evaluator.add_gaussian_noise(
                inputs,
                std=noise_level
            )
            
            # Prédiction
            with torch.no_grad():
                logits = model(noisy_inputs)
                predictions = logits.argmax(dim=1)
            
            # Prédiction sur données propres
            with torch.no_grad():
                clean_logits = model(inputs)
                clean_predictions = clean_logits.argmax(dim=1)
            
            # Taux de changement
            change_rate = (predictions != clean_predictions).float().mean().item()
            
            results[noise_level] = 1 - change_rate  # Robustesse
        
        return results