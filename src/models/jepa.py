"""
Video Joint Embedding Predictive Architecture (v-JEPA)
Implémentation complète et optimisée pour la détection de deepfakes.

Basé sur les travaux de Meta AI (LeCun et al., 2023)
Adapté pour l'apprentissage de représentations robustes des visages authentiques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List, Dict, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class VJEPAConfig:
    """Configuration de l'architecture v-JEPA."""
    # Dimensions d'entrée
    input_size: Tuple[int, int] = (224, 224)
    num_frames: int = 16
    patch_size: Tuple[int, int] = (16, 16)
    tubelet_size: int = 2
    
    # Architecture de l'encodeur
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    drop_path_rate: float = 0.1
    
    # Prédicteur
    predictor_depth: int = 4
    predictor_hidden_dim: int = 384
    predictor_num_heads: int = 6
    
    # Masquage
    spatial_mask_ratio: float = 0.75
    temporal_mask_ratio: float = 0.90
    mask_strategy: str = 'block'  # block, random, tube
    
    # EMA
    ema_decay: float = 0.998
    ema_end_decay: float = 0.9998
    ema_anneal_steps: int = 100000
    
    # Projection
    output_dim: int = 512
    projection_hidden_dim: int = 1024
    
    # Options
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True

class PatchEmbed3D(nn.Module):
    """
    Embedding des patches 3D (spatio-temporels) pour vidéo.
    Convertit une vidéo en séquence de tokens.
    """
    
    def __init__(self, config: VJEPAConfig):
        super().__init__()
        self.config = config
        
        # Dimensions des patches
        self.tubelet_size = config.tubelet_size
        self.patch_size = config.patch_size
        
        # Calcul du nombre de patches
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
        
        # Encodage positionnel (appris)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, config.embed_dim)
        )
        
        # Token temporel pour l'agrégation
        self.temporal_token = nn.Parameter(
            torch.zeros(1, 1, config.embed_dim)
        )
        
        # Dropout
        self.pos_drop = nn.Dropout(config.dropout)
        
        # Initialisation
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation des poids."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_token, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) - Vidéo d'entrée
            mask: (B, N) - Masque binaire (1 = garder, 0 = masquer)
            
        Returns:
            tokens: (B, N+1, D) - Tokens avec token temporel
        """
        B, C, T, H, W = x.shape
        
        # Projection des patches
        x = self.proj(x)  # (B, D, T', H', W')
        
        # Reshape en séquence de tokens
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        
        # Ajout de l'encodage positionnel
        x = x + self.pos_embed
        
        # Ajout du token temporel
        temporal_tokens = self.temporal_token.expand(B, -1, -1)
        x = torch.cat([temporal_tokens, x], dim=1)  # (B, N+1, D)
        
        # Application du masque si fourni
        if mask is not None:
            # Étendre le masque pour inclure le token temporel
            mask = torch.cat([torch.ones(B, 1, device=mask.device), mask], dim=1)
            
            # Token de masquage
            mask_token = nn.Parameter(torch.zeros(1, 1, self.config.embed_dim))
            mask_token = mask_token.to(x.device)
            
            x = x * mask.unsqueeze(-1) + mask_token * (1 - mask.unsqueeze(-1))
        
        x = self.pos_drop(x)
        
        return x

class FlashAttention(nn.Module):
    """
    Attention flash optimisée pour les longues séquences.
    Utilise l'implémentation de PyTorch si disponible.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Projection QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention avec flash attention si disponible
        if hasattr(F, 'scaled_dot_product_attention'):
            # Flash attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # Attention standard
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            if self.dropout > 0:
                attn = F.dropout(attn, p=self.dropout, training=self.training)
            attn_output = attn @ v
        
        # Fusion des têtes
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        output = self.proj(attn_output)
        
        return output

class TransformerBlock(nn.Module):
    """
    Bloc Transformer avec attention flash et MLP.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        use_flash_attention: bool = True
    ):
        super().__init__()
        
        # Attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlashAttention(dim, num_heads, dropout) if use_flash_attention \
                   else nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention résiduelle
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # MLP résiduel
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class DropPath(nn.Module):
    """
    Drop path pour la régularisation stochastique.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        
        return x.div(keep_prob) * random_tensor

class VideoEncoder(nn.Module):
    """
    Encodeur vidéo spatio-temporel pour v-JEPA.
    """
    
    def __init__(self, config: VJEPAConfig):
        super().__init__()
        self.config = config
        
        # Embedding des patches
        self.patch_embed = PatchEmbed3D(config)
        
        # Drop path rates
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)]
        
        # Blocs Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                drop_path=dpr[i],
                use_flash_attention=config.use_flash_attention
            )
            for i in range(config.depth)
        ])
        
        # Normalisation finale
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Gradient checkpointing
        self.use_gradient_checkpointing = config.use_gradient_checkpointing
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) - Vidéo
            mask: (B, N) - Masque binaire
            
        Returns:
            features: (B, N+1, D) - Features des patches
        """
        # Embedding des patches
        x = self.patch_embed(x, mask)
        
        # Passage dans les blocs Transformer
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        
        # Normalisation
        x = self.norm(x)
        
        return x

class LatentPredictor(nn.Module):
    """
    Prédicteur dans l'espace latent pour v-JEPA.
    Prédit les features des patches masqués.
    """
    
    def __init__(self, config: VJEPAConfig):
        super().__init__()
        self.config = config
        
        # Projection d'entrée
        self.input_proj = nn.Linear(config.embed_dim, config.predictor_hidden_dim)
        
        # Blocs Transformer du prédicteur
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.predictor_hidden_dim,
                num_heads=config.predictor_num_heads,
                mlp_ratio=4.0,
                dropout=config.dropout,
                use_flash_attention=config.use_flash_attention
            )
            for _ in range(config.predictor_depth)
        ])
        
        # Projection de sortie
        self.norm = nn.LayerNorm(config.predictor_hidden_dim)
        self.output_proj = nn.Linear(config.predictor_hidden_dim, config.embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N+1, D) - Features du contexte
            
        Returns:
            predictions: (B, N+1, D) - Prédictions
        """
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = self.output_proj(x)
        
        return x

class VJEPAModel(nn.Module):
    """
    Video JEPA complet avec encodeurs context et target.
    Architecture principale pour l'apprentissage auto-supervisé.
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
            nn.Linear(config.embed_dim, config.projection_hidden_dim),
            nn.BatchNorm1d(config.projection_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.projection_hidden_dim, config.output_dim)
        )
        
        # État EMA
        self.register_buffer('ema_decay', torch.tensor(config.ema_decay))
        self.register_buffer('ema_end_decay', torch.tensor(config.ema_end_decay))
        self.register_buffer('ema_anneal_steps', torch.tensor(config.ema_anneal_steps))
        self.register_buffer('current_step', torch.tensor(0))
    
    def update_target_encoder(self):
        """Mise à jour EMA de l'encodeur cible avec annealing."""
        # Calcul du decay avec annealing
        if self.current_step < self.ema_anneal_steps:
            progress = self.current_step / self.ema_anneal_steps
            decay = self.ema_decay + (self.ema_end_decay - self.ema_decay) * progress
        else:
            decay = self.ema_end_decay
        
        # Mise à jour des paramètres
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
        Génère les masques spatio-temporels pour le contexte et la cible.
        
        Returns:
            context_mask: (B, N) - Masque pour le contexte
            target_mask: (B, N) - Masque pour la cible
        """
        num_patches = self.context_encoder.patch_embed.num_patches
        
        if self.config.mask_strategy == 'block':
            # Masquage par blocs
            context_mask = self._generate_block_mask(batch_size, num_patches, device)
        elif self.config.mask_strategy == 'tube':
            # Masquage par tubes temporels
            context_mask = self._generate_tube_mask(batch_size, num_patches, device)
        else:
            # Masquage aléatoire
            context_mask = torch.rand(batch_size, num_patches, device=device) > self.config.spatial_mask_ratio
        
        # La cible voit toutes les patches
        target_mask = torch.ones(batch_size, num_patches, device=device)
        
        return context_mask.float(), target_mask.float()
    
    def _generate_block_mask(
        self,
        batch_size: int,
        num_patches: int,
        device: torch.device
    ) -> torch.Tensor:
        """Génère un masque par blocs spatio-temporels."""
        # Dimensions des patches
        num_temporal = self.config.num_frames // self.config.tubelet_size
        num_spatial = num_patches // num_temporal
        
        # Taille des blocs
        block_temporal = max(1, int(num_temporal * 0.3))
        block_spatial = max(1, int(num_spatial * 0.3))
        
        mask = torch.ones(batch_size, num_patches, device=device)
        
        for b in range(batch_size):
            # Nombre de blocs à masquer
            num_blocks = int(num_patches * self.config.spatial_mask_ratio / (block_temporal * block_spatial))
            
            for _ in range(num_blocks):
                # Position du bloc
                t_start = torch.randint(0, num_temporal - block_temporal + 1, (1,)).item()
                s_start = torch.randint(0, num_spatial - block_spatial + 1, (1,)).item()
                
                # Masquage du bloc
                for t in range(t_start, t_start + block_temporal):
                    for s in range(s_start, s_start + block_spatial):
                        idx = t * num_spatial + s
                        mask[b, idx] = 0
        
        return mask
    
    def _generate_tube_mask(
        self,
        batch_size: int,
        num_patches: int,
        device: torch.device
    ) -> torch.Tensor:
        """Génère un masque par tubes temporels."""
        num_temporal = self.config.num_frames // self.config.tubelet_size
        num_spatial = num_patches // num_temporal
        
        mask = torch.ones(batch_size, num_patches, device=device)
        
        for b in range(batch_size):
            # Masquage de tubes temporels
            num_tubes = int(num_spatial * self.config.spatial_mask_ratio)
            spatial_indices = torch.randperm(num_spatial)[:num_tubes]
            
            for s in spatial_indices:
                for t in range(num_temporal):
                    idx = t * num_spatial + s
                    mask[b, idx] = 0
        
        return mask
    
    def forward(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass du v-JEPA.
        
        Args:
            x_context: (B, C, T, H, W) - Vidéo contexte
            x_target: (B, C, T, H, W) - Vidéo cible
            context_mask: (B, N) - Masque contexte
            target_mask: (B, N) - Masque cible
            
        Returns:
            context_pred: (B, N+1, D) - Prédictions du contexte
            target_features: (B, N+1, D) - Features cibles
        """
        # Génération des masques si non fournis
        if context_mask is None or target_mask is None:
            context_mask, target_mask = self.generate_mask(
                x_context.size(0),
                x_context.device
            )
        
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
        use_target: bool = False,
        return_all_tokens: bool = False
    ) -> torch.Tensor:
        """
        Extrait les features latentes pour la détection.
        
        Args:
            x: (B, C, T, H, W) - Vidéo
            use_target: Utiliser l'encodeur cible
            return_all_tokens: Retourner tous les tokens ou juste le token temporel
            
        Returns:
            features: (B, D) ou (B, N+1, D)
        """
        encoder = self.target_encoder if use_target else self.context_encoder
        features = encoder(x)
        
        if return_all_tokens:
            return features
        
        # Extraction du token temporel (premier token)
        temporal_features = features[:, 0]  # (B, D)
        
        # Projection
        projected = self.projector(temporal_features)
        
        return projected
    
    def compute_anomaly_score(
        self,
        x: torch.Tensor,
        reference_mean: torch.Tensor,
        reference_cov_inv: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule le score d'anomalie basé sur la distance de Mahalanobis.
        """
        features = self.encode(x)
        
        # Distance de Mahalanobis
        diff = features - reference_mean
        score = torch.sqrt(
            torch.einsum('bi,ij,bj->b', diff, reference_cov_inv, diff)
        )
        
        return score