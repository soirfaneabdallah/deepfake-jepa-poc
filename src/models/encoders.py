"""
Encodeurs vidéo alternatifs pour la comparaison.
Inclut ViT, Swin et SlowFast.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math

class VideoViTEncoder(nn.Module):
    """
    Vision Transformer pour vidéo.
    Alternative à l'encodeur v-JEPA.
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (224, 224),
        num_frames: int = 16,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.patch_size = patch_size
        
        # Calcul du nombre de patches
        self.num_patches_h = input_size[0] // patch_size
        self.num_patches_w = input_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Embedding des patches
        self.patch_embed = nn.Conv3d(
            3, embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        
        # Encodage positionnel
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_frames * self.num_patches, embed_dim)
        )
        
        # Token de classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Blocs Transformer
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialisation
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)
            
        Returns:
            features: (B, D)
        """
        B, C, T, H, W = x.shape
        
        # Embedding des patches
        x = self.patch_embed(x)  # (B, D, T, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        
        # Ajout de l'encodage positionnel
        x = x + self.pos_embed
        
        # Ajout du token de classification
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Passage dans les blocs Transformer
        for block in self.blocks:
            x = block(x)
        
        # Normalisation
        x = self.norm(x)
        
        # Extraction du token de classification
        cls_features = x[:, 0]
        
        return cls_features

class VideoSwinEncoder(nn.Module):
    """
    Swin Transformer pour vidéo.
    Utilise des fenêtres d'attention pour l'efficacité.
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (224, 224),
        num_frames: int = 16,
        patch_size: Tuple[int, int] = (4, 4),
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: Tuple[int, int, int] = (8, 7, 7),
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.patch_size = patch_size
        
        # Embedding des patches
        self.patch_embed = nn.Conv3d(
            3, embed_dim,
            kernel_size=(1, patch_size[0], patch_size[1]),
            stride=(1, patch_size[0], patch_size[1])
        )
        
        # Encodage positionnel
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim))
        
        # Construction des étages Swin
        self.layers = nn.ModuleList()
        current_dim = embed_dim
        
        for i, (depth, num_head) in enumerate(zip(depths, num_heads)):
            layer = SwinTransformerStage(
                dim=current_dim,
                depth=depth,
                num_heads=num_head,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            self.layers.append(layer)
            
            # Doubler la dimension pour le prochain étage
            if i < len(depths) - 1:
                current_dim *= 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)
            
        Returns:
            features: (B, D)
        """
        B, C, T, H, W = x.shape
        
        # Embedding des patches
        x = self.patch_embed(x)
        
        # Ajout de l'encodage positionnel
        x = x + self.pos_embed.view(1, -1, 1, 1, 1)
        
        # Passage dans les étages Swin
        for layer in self.layers:
            x = layer(x)
        
        # Global pooling
        features = x.mean(dim=(2, 3, 4))
        
        return features

class SwinTransformerStage(nn.Module):
    """Étage Swin Transformer."""
    
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        mlp_ratio: float,
        dropout: float
    ):
        super().__init__()
        
        self.window_size = window_size
        
        # Blocs Swin
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                shift_size=0 if (i % 2 == 0) else window_size[0] // 2
            )
            for i in range(depth)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Bloc Swin Transformer avec attention par fenêtres."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        mlp_ratio: float,
        dropout: float,
        shift_size: int
    ):
        super().__init__()
        
        self.window_size = window_size
        self.shift_size = shift_size
        
        # Normalisation
        self.norm1 = nn.LayerNorm(dim)
        
        # Attention par fenêtres
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, T, H, W = x.shape
        
        # Reshape pour l'attention
        x_reshaped = x.permute(0, 2, 3, 4, 1)  # (B, T, H, W, D)
        
        # Application du shift si nécessaire
        if self.shift_size > 0:
            x_shifted = torch.roll(
                x_reshaped,
                shifts=(-self.shift_size, -self.shift_size, -self.shift_size),
                dims=(1, 2, 3)
            )
        else:
            x_shifted = x_reshaped
        
        # Attention
        attn_output = self.attn(self.norm1(x_shifted))
        
        # Retour du shift
        if self.shift_size > 0:
            attn_output = torch.roll(
                attn_output,
                shifts=(self.shift_size, self.shift_size, self.shift_size),
                dims=(1, 2, 3)
            )
        
        # Reshape retour
        attn_output = attn_output.permute(0, 4, 1, 2, 3)
        
        # Résiduel
        x = x + attn_output
        
        # MLP
        x = x + self.mlp(self.norm2(x.permute(0, 2, 3, 4, 1))).permute(0, 4, 1, 2, 3)
        
        return x

class WindowAttention(nn.Module):
    """Attention par fenêtres pour Swin Transformer."""
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        dropout: float
    ):
        super().__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Nombre total de tokens par fenêtre
        self.num_window_tokens = window_size[0] * window_size[1] * window_size[2]
        
        # Projections
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout
        
        # Masque d'attention relatif
        self.relative_position_bias = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) *
                (2 * window_size[1] - 1) *
                (2 * window_size[2] - 1),
                num_heads
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, D = x.shape
        
        # Reshape en fenêtres
        x_windows = self._window_partition(x)  # (B*nW, T', H', W', D)
        
        # Projection QKV
        qkv = self.qkv(x_windows).reshape(
            -1, self.num_window_tokens, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*nW, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        if self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # Sortie
        output = (attn @ v).transpose(1, 2).reshape(
            -1, self.num_window_tokens, self.dim
        )
        output = self.proj(output)
        
        # Reconstruction
        output = self._window_reverse(output, B, T, H, W)
        
        return output
    
    def _window_partition(self, x: torch.Tensor) -> torch.Tensor:
        """Divise en fenêtres."""
        B, T, H, W, D = x.shape
        wt, wh, ww = self.window_size
        
        # Padding
        pad_t = (wt - T % wt) % wt
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww
        
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
        
        T_p, H_p, W_p = T + pad_t, H + pad_h, W + pad_w
        
        # Reshape en fenêtres
        x = x.view(B, T_p // wt, wt, H_p // wh, wh, W_p // ww, ww, D)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        x = x.reshape(-1, wt, wh, ww, D)
        
        return x
    
    def _window_reverse(
        self,
        windows: torch.Tensor,
        B: int,
        T: int,
        H: int,
        W: int
    ) -> torch.Tensor:
        """Reconstruit à partir des fenêtres."""
        wt, wh, ww = self.window_size
        T_p, H_p, W_p = T + (wt - T % wt) % wt, H + (wh - H % wh) % wh, W + (ww - W % ww) % ww
        
        # Reshape inverse
        x = windows.view(B, T_p // wt, H_p // wh, W_p // ww, wt, wh, ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7)
        x = x.reshape(B, T_p, H_p, W_p, -1)
        
        # Crop pour enlever le padding
        x = x[:, :T, :H, :W, :]
        
        return x

class SlowFastEncoder(nn.Module):
    """
    Encodeur SlowFast pour la détection de deepfakes.
    Deux branches : lente (spatiale) et rapide (temporelle).
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (224, 224),
        num_frames: int = 16,
        alpha: int = 8,  # Ratio temporel
        beta: float = 0.125,  # Ratio de canaux
        embed_dim: int = 256
    ):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.num_frames = num_frames
        
        # Branche lente
        self.slow_path = SlowPath(
            input_size=input_size,
            num_frames=num_frames // alpha,
            base_dim=embed_dim
        )
        
        # Branche rapide
        self.fast_path = FastPath(
            input_size=input_size,
            num_frames=num_frames,
            base_dim=int(embed_dim * beta)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + int(embed_dim * beta), embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)
            
        Returns:
            features: (B, D)
        """
        # Branche lente (échantillonnage temporel)
        x_slow = x[:, :, ::self.alpha]
        slow_features = self.slow_path(x_slow)
        
        # Branche rapide (toutes les frames)
        fast_features = self.fast_path(x)
        
        # Fusion
        combined = torch.cat([slow_features, fast_features], dim=1)
        features = self.fusion(combined)
        
        return features

class SlowPath(nn.Module):
    """Branche lente pour les détails spatiaux."""
    
    def __init__(
        self,
        input_size: Tuple[int, int],
        num_frames: int,
        base_dim: int
    ):
        super().__init__()
        
        self.conv1 = nn.Conv3d(3, base_dim, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
        self.bn1 = nn.BatchNorm3d(base_dim)
        
        self.conv2 = nn.Conv3d(base_dim, base_dim * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(base_dim * 2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        return x.flatten(1)

class FastPath(nn.Module):
    """Branche rapide pour les détails temporels."""
    
    def __init__(
        self,
        input_size: Tuple[int, int],
        num_frames: int,
        base_dim: int
    ):
        super().__init__()
        
        self.conv1 = nn.Conv3d(3, base_dim, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3))
        self.bn1 = nn.BatchNorm3d(base_dim)
        
        self.conv2 = nn.Conv3d(base_dim, base_dim * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(base_dim * 2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        return x.flatten(1)

def create_encoder(
    encoder_type: str,
    **kwargs
) -> nn.Module:
    """
    Factory pour créer un encodeur vidéo.
    """
    encoders = {
        'video_vit': VideoViTEncoder,
        'video_swin': VideoSwinEncoder,
        'slowfast': SlowFastEncoder
    }
    
    if encoder_type not in encoders:
        raise ValueError(f"Encodeur non supporté: {encoder_type}")
    
    return encoders[encoder_type](**kwargs)