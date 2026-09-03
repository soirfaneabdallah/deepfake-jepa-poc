"""
Video Joint Embedding Predictive Architecture (v-JEPA)
Implémentation complète pour la détection de deepfakes.

Basé sur l'architecture de Meta AI (LeCun et al., 2023)
Adapté pour l'apprentissage de représentations robustes des visages authentiques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import torch
from pathlib import Path
from typing import Optional, Union
import urllib.request
import json

@dataclass
class VJEPAConfig:
    """Configuration de l'architecture v-JEPA."""
    input_size: Tuple[int, int] = (224, 224)
    patch_size: Tuple[int, int] = (16, 16)
    tubelet_size: int = 2
    num_frames: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    spatial_mask_ratio: float = 0.75
    temporal_mask_ratio: float = 0.9
    predictor_depth: int = 4
    predictor_hidden_dim: int = 384
    output_dim: int = 512

class PatchEmbed3D(nn.Module):
    """
    Embedding des patches 3D (spatio-temporels).
    Convertit une vidéo en séquence de tokens.
    """
    
    def __init__(self, config: VJEPAConfig):
        super().__init__()
        self.config = config
        
        # Calcul des dimensions
        self.tubelet_size = config.tubelet_size
        self.patch_size = config.patch_size
        
        # Nombre de patches
        self.num_patches_temporal = config.num_frames // config.tubelet_size
        self.num_patches_height = config.input_size[0] // config.patch_size[0]
        self.num_patches_width = config.input_size[1] // config.patch_size[1]
        self.num_patches = (self.num_patches_temporal * 
                           self.num_patches_height * 
                           self.num_patches_width)
        
        # Projection des patches
        patch_dim = 3 * config.tubelet_size * config.patch_size[0] * config.patch_size[1]
        self.proj = nn.Conv3d(
            in_channels=3,
            out_channels=config.embed_dim,
            kernel_size=(config.tubelet_size, config.patch_size[0], config.patch_size[1]),
            stride=(config.tubelet_size, config.patch_size[0], config.patch_size[1])
        )
        
        # Encodage positionnel
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, config.embed_dim)
        )
        self.pos_drop = nn.Dropout(config.dropout)
        
        # Initialisation
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv3d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) - Batch, Channels, Frames, Height, Width
        Returns:
            tokens: (B, N, D) - Batch, Num patches, Embed dim
        """
        B, C, T, H, W = x.shape
        
        # Projection des patches
        x = self.proj(x)  # (B, D, T', H', W')
        
        # Reshape en séquence de tokens
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        
        # Ajout de l'encodage positionnel
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        return x

class TransformerBlock(nn.Module):
    """Bloc Transformer avec attention multi-têtes."""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention résiduelle
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP résiduel
        x = x + self.mlp(self.norm2(x))
        return x

class VideoEncoder(nn.Module):
    """Encodeur vidéo spatio-temporel pour v-JEPA."""
    
    def __init__(self, config: VJEPAConfig):
        super().__init__()
        self.config = config
        
        # Embedding des patches
        self.patch_embed = PatchEmbed3D(config)
        
        # Blocs Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout
            )
            for _ in range(config.depth)
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)
            mask: (B, N) - Masque binaire (1 = garder, 0 = masquer)
        Returns:
            features: (B, N, D)
        """
        # Embedding des patches
        x = self.patch_embed(x)
        
        # Application du masque si fourni
        if mask is not None:
            # Remplacer les tokens masqués par un token [MASK]
            mask_token = nn.Parameter(torch.zeros(1, 1, self.config.embed_dim))
            mask_token = mask_token.to(x.device)
            x = x * mask.unsqueeze(-1) + mask_token * (1 - mask.unsqueeze(-1))
        
        # Passage dans les blocs Transformer
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x

class LatentPredictor(nn.Module):
    """Prédicteur dans l'espace latent."""
    
    def __init__(self, config: VJEPAConfig):
        super().__init__()
        self.config = config
        
        # Projection d'entrée
        self.input_proj = nn.Linear(config.embed_dim, config.predictor_hidden_dim)
        
        # Blocs Transformer du prédicteur
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.predictor_hidden_dim,
                num_heads=6,
                mlp_ratio=4.0,
                dropout=config.dropout
            )
            for _ in range(config.predictor_depth)
        ])
        
        # Projection de sortie
        self.norm = nn.LayerNorm(config.predictor_hidden_dim)
        self.output_proj = nn.Linear(config.predictor_hidden_dim, config.embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = self.output_proj(x)
        return x

class VJEPAModel(nn.Module):
    """
    Video JEPA complet avec encodeurs context et target.
    """
    
    def __init__(self, config: VJEPAConfig):
        super().__init__()
        self.config = config
        
        # Encodeur contexte (apprentissage)
        self.context_encoder = VideoEncoder(config)
        
        # Encodeur cible (EMA)
        self.target_encoder = VideoEncoder(config)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        
        # Désactivation des gradients pour l'encodeur cible
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # Prédicteur latent
        self.predictor = LatentPredictor(config)
        
        # Projecteur pour la détection d'anomalies
        self.projector = nn.Sequential(
            nn.Linear(config.embed_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, config.output_dim)
        )
        
        # Paramètres EMA
        self.ema_decay = 0.998
        self.ema_end_decay = 0.9998
        self.ema_anneal_steps = 100000
        self.current_step = 0
    
    def update_target_encoder(self):
        """Mise à jour EMA de l'encodeur cible."""
        # Calcul du decay avec annealing
        if self.current_step < self.ema_anneal_steps:
            progress = self.current_step / self.ema_anneal_steps
            decay = self.ema_decay + (self.ema_end_decay - self.ema_decay) * progress
        else:
            decay = self.ema_end_decay
        
        with torch.no_grad():
            for ctx_param, tgt_param in zip(
                self.context_encoder.parameters(),
                self.target_encoder.parameters()
            ):
                tgt_param.data.mul_(decay).add_(ctx_param.data, alpha=1 - decay)
        
        self.current_step += 1
    
    def generate_mask(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Génère le masque spatio-temporel.
        
        Returns:
            context_mask: Masque pour l'encodeur contexte
            target_mask: Masque pour l'encodeur cible
        """
        num_patches = self.context_encoder.patch_embed.num_patches
        
        # Masque temporel
        num_temporal_patches = num_patches // (14 * 14)  # Approximation
        temporal_mask = torch.rand(batch_size, num_temporal_patches, device=device)
        temporal_mask = temporal_mask > self.config.temporal_mask_ratio
        
        # Masque spatial (par frame)
        spatial_mask = torch.rand(batch_size, 14 * 14, device=device)
        spatial_mask = spatial_mask > self.config.spatial_mask_ratio
        
        # Combinaison des masques
        mask = temporal_mask.unsqueeze(-1) & spatial_mask.unsqueeze(1)
        mask = mask.flatten(1)
        
        return mask.float(), mask.float()
    
    def forward(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass complet du v-JEPA.
        
        Args:
            x_context: (B, C, T, H, W) - Vidéo contexte
            x_target: (B, C, T, H, W) - Vidéo cible
            context_mask: Masque pour l'encodeur contexte
            target_mask: Masque pour l'encodeur cible
        """
        # Encodage du contexte
        context_features = self.context_encoder(x_context, context_mask)
        
        # Prédiction dans l'espace latent
        context_pred = self.predictor(context_features)
        
        # Encodage de la cible (sans gradient)
        with torch.no_grad():
            target_features = self.target_encoder(x_target, target_mask)
        
        return context_pred, target_features
    
    def encode(
        self,
        x: torch.Tensor,
        use_target: bool = False
    ) -> torch.Tensor:
        """
        Extrait les features latentes pour la détection d'anomalies.
        
        Args:
            x: (B, C, T, H, W) - Vidéo
            use_target: Utiliser l'encodeur cible ou contexte
        """
        encoder = self.target_encoder if use_target else self.context_encoder
        features = encoder(x)  # (B, N, D)
        
        # Pooling global (moyenne sur tous les patches)
        features = features.mean(dim=1)  # (B, D)
        
        # Projection
        features = self.projector(features)
        
        return features
    
    def compute_anomaly_score(
        self,
        x: torch.Tensor,
        reference_features: torch.Tensor,
        reference_cov_inv: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule le score d'anomalie basé sur la distance de Mahalanobis.
        """
        features = self.encode(x)
        
        # Distance de Mahalanobis
        diff = features - reference_features.mean(dim=0)
        score = torch.sqrt(diff @ reference_cov_inv @ diff.T)
        
        return score
    


class VJEPAPretrainedLoader:
    """
    Chargeur de modèles v-JEPA pré-entraînés.
    Supporte : fichiers locaux, Hugging Face, URLs.
    """
    
    # Modèles disponibles
    MODELS = {
        'vjepa_base': {
            'url': 'https://dl.fbaipublicfiles.com/jepa/vjepa_base.pth',
            'config': {'embed_dim': 384, 'depth': 6, 'num_heads': 6}
        },
        'vjepa_large': {
            'url': 'https://dl.fbaipublicfiles.com/jepa/vjepa_large.pth',
            'config': {'embed_dim': 768, 'depth': 12, 'num_heads': 12}
        }
    }
    
    @classmethod
    def load(cls, 
             source: Union[str, Path], 
             device: str = 'cuda',
             strict: bool = False) -> torch.nn.Module:
        """
        Charge un modèle v-JEPA.
        
        Args:
            source: Chemin local, nom du modèle ('vjepa_base'), ou URL
            device: 'cuda' ou 'cpu'
            strict: Chargement strict des poids
            
        Returns:
            VJEPAModel chargé
        """
        
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        source = str(source)
        
        # Si c'est un nom de modèle prédéfini
        if source in cls.MODELS:
            config_dict = cls.MODELS[source]['config']
            url = cls.MODELS[source]['url']
            local_path = Path.home() / '.cache' / 'vjepa' / f'{source}.pth'
            
            if not local_path.exists():
                local_path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, local_path)
            
            source = local_path
        
        # Config par défaut
        config = VJEPAConfig()
        model = VJEPAModel(config).to(device)
        
        # Chargement
        if Path(source).exists():
            checkpoint = torch.load(source, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Filtrer les clés incompatibles
            model_state = model.state_dict()
            filtered = {k: v for k, v in state_dict.items() 
                       if k in model_state and v.shape == model_state[k].shape}
            
            model.load_state_dict(filtered, strict=strict)
            print(f"[V-JEPA] Modèle chargé: {source} ({len(filtered)}/{len(model_state)} paramètres)")
        
        return model
