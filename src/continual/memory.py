"""
Mémoire duale pour l'apprentissage continu.
Combine mémoire épisodique (échantillons) et sémantique (statistiques).

La mémoire épisodique stocke des exemples représentatifs,
tandis que la mémoire sémantique stocke des prototypes de classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
from collections import deque
import logging
import random

logger = logging.getLogger(__name__)

class MemorySampler(ABC):
    """
    Classe abstraite pour les stratégies d'échantillonnage de la mémoire.
    """
    
    @abstractmethod
    def sample(
        self,
        memory: List[Tuple[torch.Tensor, int]],
        batch_size: int
    ) -> List[Tuple[torch.Tensor, int]]:
        """
        Échantillonne depuis la mémoire.
        """
        pass

class RandomSampler(MemorySampler):
    """
    Échantillonnage aléatoire uniforme.
    """
    
    def sample(
        self,
        memory: List[Tuple[torch.Tensor, int]],
        batch_size: int
    ) -> List[Tuple[torch.Tensor, int]]:
        if len(memory) <= batch_size:
            return memory
        
        indices = np.random.choice(len(memory), batch_size, replace=False)
        return [memory[i] for i in indices]

class HerdingSampler(MemorySampler):
    """
    Échantillonnage par herding.
    Sélectionne les échantillons les plus proches du prototype de classe.
    """
    
    def __init__(self, feature_extractor: Optional[callable] = None):
        self.feature_extractor = feature_extractor
        
    def sample(
        self,
        memory: List[Tuple[torch.Tensor, int]],
        batch_size: int
    ) -> List[Tuple[torch.Tensor, int]]:
        if len(memory) <= batch_size:
            return memory
        
        # Calcul des prototypes de classes
        class_prototypes = {}
        for data, label in memory:
            if label not in class_prototypes:
                class_prototypes[label] = []
            class_prototypes[label].append(data)
        
        # Sélection par herding
        selected = []
        for label, samples in class_prototypes.items():
            # Calcul du prototype
            prototype = torch.stack(samples).mean(dim=0)
            
            # Sélection des échantillons les plus proches
            distances = [
                torch.norm(sample - prototype).item()
                for sample in samples
            ]
            
            # Tri par distance
            sorted_indices = np.argsort(distances)
            
            # Sélection des meilleurs
            num_select = max(1, batch_size // len(class_prototypes))
            for idx in sorted_indices[:num_select]:
                selected.append((samples[idx], label))
        
        return selected[:batch_size]

class UncertaintySampler(MemorySampler):
    """
    Échantillonnage par incertitude.
    Sélectionne les échantillons les plus incertains.
    """
    
    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model
        
    def sample(
        self,
        memory: List[Tuple[torch.Tensor, int]],
        batch_size: int
    ) -> List[Tuple[torch.Tensor, int]]:
        if len(memory) <= batch_size:
            return memory
        
        if self.model is None:
            return RandomSampler().sample(memory, batch_size)
        
        # Calcul de l'incertitude
        uncertainties = []
        self.model.eval()
        
        with torch.no_grad():
            for data, label in memory:
                data = data.unsqueeze(0)
                outputs = self.model(data)
                probabilities = F.softmax(outputs, dim=1)
                
                # Incertitude = 1 - probabilité maximale
                uncertainty = 1 - probabilities.max().item()
                uncertainties.append((uncertainty, data, label))
        
        # Tri par incertitude décroissante
        uncertainties.sort(key=lambda x: x[0], reverse=True)
        
        # Sélection des plus incertains
        selected = [(data, label) for _, data, label in uncertainties[:batch_size]]
        
        return selected

class EpisodicMemory:
    """
    Mémoire épisodique pour stocker des exemples représentatifs.
    """
    
    def __init__(
        self,
        capacity: int = 200,
        sampler: Optional[MemorySampler] = None
    ):
        self.capacity = capacity
        self.sampler = sampler or RandomSampler()
        self.buffer = deque(maxlen=capacity)
        
    def add(
        self,
        data: torch.Tensor,
        label: int,
        features: Optional[torch.Tensor] = None
    ) -> None:
        """
        Ajoute un échantillon à la mémoire.
        """
        self.buffer.append((data.cpu(), label, features))
    
    def add_batch(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> None:
        """
        Ajoute un batch d'échantillons.
        """
        for i in range(data.size(0)):
            feat = features[i] if features is not None else None
            self.add(data[i], labels[i].item(), feat)
    
    def sample(
        self,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Échantillonne un batch depuis la mémoire.
        """
        if len(self.buffer) == 0:
            return None, None
        
        # Extraction des données
        items = [(data, label) for data, label, _ in self.buffer]
        
        # Échantillonnage
        sampled = self.sampler.sample(items, min(batch_size, len(items)))
        
        # Conversion en batch
        data_batch = torch.stack([item[0] for item in sampled])
        label_batch = torch.tensor([item[1] for item in sampled])
        
        return data_batch, label_batch
    
    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne tous les échantillons.
        """
        if len(self.buffer) == 0:
            return None, None
        
        data = torch.stack([item[0] for item in self.buffer])
        labels = torch.tensor([item[1] for item in self.buffer])
        
        return data, labels
    
    def __len__(self) -> int:
        return len(self.buffer)

class SemanticMemory:
    """
    Mémoire sémantique pour stocker des prototypes de classes.
    """
    
    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim
        self.class_stats = {}
        
    def update(
        self,
        class_id: int,
        features: torch.Tensor
    ) -> None:
        """
        Met à jour les statistiques d'une classe.
        
        Args:
            class_id: Identifiant de la classe
            features: (N, D) - Features des échantillons
        """
        if class_id not in self.class_stats:
            self.class_stats[class_id] = {
                'mean': features.mean(dim=0),
                'var': features.var(dim=0),
                'count': features.size(0)
            }
        else:
            # Mise à jour incrémentale
            old_stats = self.class_stats[class_id]
            old_count = old_stats['count']
            old_mean = old_stats['mean']
            old_var = old_stats['var']
            
            new_count = old_count + features.size(0)
            new_mean = old_mean + (features.mean(dim=0) - old_mean) * features.size(0) / new_count
            
            # Mise à jour de la variance (formule de Welford)
            delta = features.mean(dim=0) - old_mean
            new_var = (
                old_var * old_count +
                features.var(dim=0) * features.size(0) +
                delta ** 2 * old_count * features.size(0) / new_count
            ) / new_count
            
            self.class_stats[class_id] = {
                'mean': new_mean,
                'var': new_var,
                'count': new_count
            }
    
    def get_prototype(self, class_id: int) -> Optional[torch.Tensor]:
        """
        Retourne le prototype d'une classe.
        """
        if class_id in self.class_stats:
            return self.class_stats[class_id]['mean']
        return None
    
    def get_all_prototypes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne tous les prototypes.
        """
        if not self.class_stats:
            return None, None
        
        prototypes = []
        labels = []
        
        for class_id, stats in self.class_stats.items():
            prototypes.append(stats['mean'])
            labels.append(class_id)
        
        return torch.stack(prototypes), torch.tensor(labels)
    
    def compute_distance(
        self,
        features: torch.Tensor,
        class_id: int
    ) -> torch.Tensor:
        """
        Calcule la distance aux prototypes.
        """
        prototype = self.get_prototype(class_id)
        if prototype is None:
            return torch.full((features.size(0),), float('inf'))
        
        # Distance de Mahalanobis simplifiée
        diff = features - prototype
        distance = torch.norm(diff, dim=1)
        
        return distance

class DualMemory:
    """
    Mémoire duale combinant épisodique et sémantique.
    """
    
    def __init__(
        self,
        episodic_capacity: int = 200,
        feature_dim: int = 512,
        sampler: Optional[MemorySampler] = None
    ):
        self.episodic_memory = EpisodicMemory(
            capacity=episodic_capacity,
            sampler=sampler
        )
        self.semantic_memory = SemanticMemory(feature_dim=feature_dim)
        
    def add(
        self,
        data: torch.Tensor,
        label: int,
        features: Optional[torch.Tensor] = None
    ) -> None:
        """
        Ajoute un échantillon aux deux mémoires.
        """
        self.episodic_memory.add(data, label, features)
        
        if features is not None:
            self.semantic_memory.update(label, features.unsqueeze(0))
    
    def add_batch(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> None:
        """
        Ajoute un batch aux deux mémoires.
        """
        self.episodic_memory.add_batch(data, labels, features)
        
        if features is not None:
            for class_id in labels.unique():
                mask = labels == class_id
                class_features = features[mask]
                self.semantic_memory.update(class_id.item(), class_features)
    
    def sample_episodic(
        self,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Échantillonne depuis la mémoire épisodique.
        """
        return self.episodic_memory.sample(batch_size)
    
    def get_prototypes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne les prototypes sémantiques.
        """
        return self.semantic_memory.get_all_prototypes()
    
    def compute_prototype_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la perte basée sur les prototypes.
        """
        prototypes, prototype_labels = self.get_prototypes()
        
        if prototypes is None:
            return torch.tensor(0.0, device=features.device)
        
        prototypes = prototypes.to(features.device)
        prototype_labels = prototype_labels.to(features.device)
        
        # Distance aux prototypes de la même classe
        loss = 0.0
        for i, label in enumerate(labels):
            mask = prototype_labels == label
            if mask.any():
                class_prototypes = prototypes[mask]
                distances = torch.norm(features[i] - class_prototypes, dim=1)
                loss += distances.min()
        
        return loss / len(labels)
    
    def clear(self) -> None:
        """
        Vide les deux mémoires.
        """
        self.episodic_memory.buffer.clear()
        self.semantic_memory.class_stats.clear()