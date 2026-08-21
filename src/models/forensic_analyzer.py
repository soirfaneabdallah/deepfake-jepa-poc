"""
Analyse médico-légale pour la détection de deepfakes.
Extrait des caractéristiques forensiques complémentaires aux features JEPA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

class SpectralAnalyzer(nn.Module):
    """
    Analyse spectrale pour détecter les artefacts de génération.
    Les GANs laissent des pics fréquentiels caractéristiques.
    """
    
    def __init__(self, num_frequency_bands: int = 16):
        super().__init__()
        self.num_frequency_bands = num_frequency_bands
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Analyse le spectre fréquentiel des frames.
        
        Args:
            x: (B, C, T, H, W) - Vidéo
            
        Returns:
            features: (B, num_frequency_bands * 4)
        """
        B, C, T, H, W = x.shape
        
        # Application FFT 2D sur chaque frame
        features = []
        
        for b in range(B):
            for t in range(T):
                frame = x[b, :, t]  # (C, H, W)
                
                # FFT 2D
                fft = torch.fft.fft2(frame)
                magnitude = torch.abs(fft)
                magnitude = torch.fft.fftshift(magnitude)
                
                # Extraction des bandes de fréquence
                bands = self._extract_frequency_bands(magnitude)
                features.append(bands)
        
        features = torch.stack(features)  # (B*T, num_bands * C)
        features = features.view(B, T, -1)
        
        # Agrégation temporelle
        features = features.mean(dim=1)  # (B, num_bands * C)
        
        return features
    
    def _extract_frequency_bands(self, magnitude: torch.Tensor) -> torch.Tensor:
        """
        Extrait l'énergie dans différentes bandes de fréquence.
        """
        C, H, W = magnitude.shape
        center_h, center_w = H // 2, W // 2
        
        bands = []
        for i in range(self.num_frequency_bands):
            # Rayon de la bande
            r_min = i * min(H, W) // (2 * self.num_frequency_bands)
            r_max = (i + 1) * min(H, W) // (2 * self.num_frequency_bands)
            
            # Masque circulaire
            y, x = torch.meshgrid(
                torch.arange(H, device=magnitude.device),
                torch.arange(W, device=magnitude.device)
            )
            
            dist = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
            mask = (dist >= r_min) & (dist < r_max)
            
            # Énergie dans la bande
            energy = magnitude[:, mask].mean(dim=1)
            bands.append(energy)
        
        return torch.cat(bands)  # (C * num_bands,)

class CompressionArtifactDetector(nn.Module):
    """
    Détection des artefacts de compression.
    Les deepfakes ont souvent des artefacts de compression incohérents.
    """
    
    def __init__(self):
        super().__init__()
        
        # Filtres pour la détection d'artefacts
        self.laplacian = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Détecte les artefacts de compression.
        
        Args:
            x: (B, C, T, H, W)
            
        Returns:
            features: (B, 6)
        """
        B, C, T, H, W = x.shape
        
        # Reshape pour traitement par frame
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)
        
        # Application du filtre laplacien
        laplacian = self.laplacian.to(x.device)
        edges = F.conv2d(x.mean(dim=1, keepdim=True), laplacian, padding=1)
        
        # Statistiques des artefacts
        edge_mean = edges.mean(dim=(2, 3))
        edge_std = edges.std(dim=(2, 3))
        edge_max = edges.max(dim=3)[0].max(dim=2)[0]
        
        # Détection de blocs (artefacts JPEG)
        blockiness = self._detect_blockiness(x)
        
        # Reshape retour
        features = torch.cat([
            edge_mean.view(B, T, 1),
            edge_std.view(B, T, 1),
            edge_max.view(B, T, 1),
            blockiness.view(B, T, 3)
        ], dim=-1)
        
        # Agrégation temporelle
        features = features.mean(dim=1)  # (B, 6)
        
        return features
    
    def _detect_blockiness(self, x: torch.Tensor) -> torch.Tensor:
        """
        Détecte la structure en blocs (artefacts JPEG).
        """
        B, C, H, W = x.shape
        
        # Différences horizontales et verticales
        diff_h = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        diff_v = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        
        # Détection des frontières de blocs (tous les 8 pixels)
        block_size = 8
        
        block_h = diff_h[:, :, :, ::block_size].mean(dim=(2, 3))
        block_v = diff_v[:, :, ::block_size, :].mean(dim=(2, 3))
        non_block_h = diff_h[:, :, :, 1::block_size].mean(dim=(2, 3))
        non_block_v = diff_v[:, :, 1::block_size, :].mean(dim=(2, 3))
        
        # Ratio block/non-block
        ratio_h = block_h / (non_block_h + 1e-8)
        ratio_v = block_v / (non_block_v + 1e-8)
        
        # Moyenne sur les canaux
        blockiness_h = ratio_h.mean(dim=1)
        blockiness_v = ratio_v.mean(dim=1)
        blockiness_total = (blockiness_h + blockiness_v) / 2
        
        return torch.stack([blockiness_h, blockiness_v, blockiness_total], dim=-1)

class TextureAnalyzer(nn.Module):
    """
    Analyse de texture pour détecter les incohérences.
    Les deepfakes ont souvent des textures de peau irrégulières.
    """
    
    def __init__(self):
        super().__init__()
        
        # Filtres de Gabor pour l'analyse de texture
        self.gabor_filters = self._create_gabor_filters()
        
    def _create_gabor_filters(self) -> nn.Parameter:
        """
        Crée des filtres de Gabor avec différentes orientations.
        """
        filters = []
        orientations = [0, 45, 90, 135]
        
        for theta in orientations:
            theta_rad = theta * np.pi / 180
            
            # Paramètres du filtre
            sigma = 4.0
            lambda_ = 10.0
            gamma = 0.5
            psi = 0
            
            # Création du filtre
            kernel_size = 21
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            
            for i in range(kernel_size):
                for j in range(kernel_size):
                    x = j - kernel_size // 2
                    y = i - kernel_size // 2
                    
                    x_theta = x * np.cos(theta_rad) + y * np.sin(theta_rad)
                    y_theta = -x * np.sin(theta_rad) + y * np.cos(theta_rad)
                    
                    kernel[i, j] = np.exp(
                        -(x_theta ** 2 + gamma ** 2 * y_theta ** 2) / (2 * sigma ** 2)
                    ) * np.cos(2 * np.pi * x_theta / lambda_ + psi)
            
            filters.append(torch.from_numpy(kernel))
        
        filters = torch.stack(filters).view(-1, 1, kernel_size, kernel_size)
        return nn.Parameter(filters, requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Analyse la texture des frames.
        
        Args:
            x: (B, C, T, H, W)
            
        Returns:
            features: (B, 12)
        """
        B, C, T, H, W = x.shape
        
        # Reshape pour traitement par frame
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)
        
        # Conversion en niveaux de gris
        gray = x.mean(dim=1, keepdim=True)  # (B*T, 1, H, W)
        
        # Application des filtres de Gabor
        gabor_responses = []
        for gabor_filter in self.gabor_filters:
            response = F.conv2d(
                gray,
                gabor_filter.unsqueeze(0),
                padding=10
            )
            
            # Statistiques
            mean_response = response.mean(dim=(2, 3))
            std_response = response.std(dim=(2, 3))
            energy = (response ** 2).mean(dim=(2, 3))
            
            gabor_responses.extend([mean_response, std_response, energy])
        
        features = torch.cat(gabor_responses, dim=-1)  # (B*T, 12)
        
        # Reshape retour
        features = features.view(B, T, -1)
        
        # Agrégation temporelle
        features = features.mean(dim=1)  # (B, 12)
        
        return features

class GeometricAnalyzer(nn.Module):
    """
    Analyse géométrique pour détecter les incohérences.
    Vérifie les ratios anthropométriques du visage.
    """
    
    def __init__(self):
        super().__init__()
        
        # Ratios anthropométriques attendus
        self.expected_ratios = torch.tensor([
            0.46,  # Distance yeux/nez / largeur visage
            0.36,  # Distance nez/bouche / longueur visage
            1.62,  # Ratio longueur/largeur (nombre d'or)
            0.85,  # Symétrie gauche/droite
            0.31,  # Distance entre les yeux / largeur visage
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        landmarks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Analyse la géométrie du visage.
        
        Args:
            x: (B, C, T, H, W)
            landmarks: (B, T, 68, 2) - Landmarks faciaux
            
        Returns:
            features: (B, 10)
        """
        B, C, T, H, W = x.shape
        
        if landmarks is None:
            # Extraction approximative des landmarks via les gradients
            landmarks = self._extract_approximate_landmarks(x)
        
        # Calcul des ratios géométriques
        ratios = self._compute_ratios(landmarks)  # (B, T, 5)
        
        # Symétrie
        symmetry = self._compute_symmetry(x)  # (B, T, 3)
        
        # Incohérences temporelles
        temporal_inconsistency = self._compute_temporal_inconsistency(landmarks)  # (B, 2)
        
        # Agrégation
        ratio_stats = torch.cat([
            ratios.mean(dim=1),  # (B, 5)
            ratios.std(dim=1),   # (B, 5)
        ], dim=1)  # (B, 10)
        
        return ratio_stats
    
    def _extract_approximate_landmarks(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Extraction approximative des landmarks via les gradients.
        """
        B, C, T, H, W = x.shape
        
        # Points d'intérêt via les gradients
        grad_x = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
        grad_y = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        
        # Positions des gradients maximum (approximation des landmarks)
        landmarks = torch.zeros(B, T, 68, 2, device=x.device)
        
        return landmarks
    
    def _compute_ratios(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calcule les ratios anthropométriques.
        """
        B, T, N, _ = landmarks.shape
        
        # Initialisation des ratios
        ratios = torch.zeros(B, T, 5, device=landmarks.device)
        
        # TODO: Implémenter le calcul des ratios avec les landmarks réels
        
        return ratios
    
    def _compute_symmetry(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcule la symétrie du visage.
        """
        B, C, T, H, W = x.shape
        
        # Flip horizontal
        x_flipped = torch.flip(x, dims=[-1])
        
        # Différence entre l'original et le flip
        diff = torch.abs(x - x_flipped)
        
        # Symétrie moyenne
        symmetry = diff.mean(dim=(1, 3, 4))  # (B, T)
        
        # Symétrie par région (gauche, centre, droite)
        left_sym = diff[:, :, :, :, :W//3].mean(dim=(1, 3, 4))
        center_sym = diff[:, :, :, :, W//3:2*W//3].mean(dim=(1, 3, 4))
        right_sym = diff[:, :, :, :, 2*W//3:].mean(dim=(1, 3, 4))
        
        return torch.stack([symmetry, left_sym, center_sym, right_sym], dim=-1)
    
    def _compute_temporal_inconsistency(
        self,
        landmarks: torch.Tensor
    ) -> torch.Tensor:
        """
        Détecte les incohérences temporelles dans les landmarks.
        """
        B, T, N, _ = landmarks.shape
        
        # Différences entre frames consécutives
        if T > 1:
            diff = torch.abs(landmarks[:, 1:] - landmarks[:, :-1])
            inconsistency = diff.mean(dim=(1, 2, 3))
            
            # Détection de sauts brusques
            jumps = torch.abs(diff.mean(dim=2)).max(dim=1)[0]
            
            return torch.stack([inconsistency, jumps], dim=1)
        else:
            return torch.zeros(B, 2, device=landmarks.device)

class ForensicAnalyzer(nn.Module):
    """
    Analyseur médico-légal complet.
    Combine toutes les analyses forensiques.
    """
    
    def __init__(
        self,
        use_spectral: bool = True,
        use_compression: bool = True,
        use_texture: bool = True,
        use_geometric: bool = True
    ):
        super().__init__()
        
        self.use_spectral = use_spectral
        self.use_compression = use_compression
        self.use_texture = use_texture
        self.use_geometric = use_geometric
        
        # Initialisation des analyseurs
        if use_spectral:
            self.spectral_analyzer = SpectralAnalyzer()
        
        if use_compression:
            self.compression_detector = CompressionArtifactDetector()
        
        if use_texture:
            self.texture_analyzer = TextureAnalyzer()
        
        if use_geometric:
            self.geometric_analyzer = GeometricAnalyzer()
    
    def forward(
        self,
        x: torch.Tensor,
        landmarks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extrait toutes les caractéristiques forensiques.
        
        Args:
            x: (B, C, T, H, W)
            landmarks: Landmarks faciaux optionnels
            
        Returns:
            features: (B, total_features)
        """
        features = []
        
        if self.use_spectral:
            spectral_features = self.spectral_analyzer(x)
            features.append(spectral_features)
        
        if self.use_compression:
            compression_features = self.compression_detector(x)
            features.append(compression_features)
        
        if self.use_texture:
            texture_features = self.texture_analyzer(x)
            features.append(texture_features)
        
        if self.use_geometric:
            geometric_features = self.geometric_analyzer(x, landmarks)
            features.append(geometric_features)
        
        # Concatenation
        combined = torch.cat(features, dim=1)
        
        return combined