#!/usr/bin/env python3
"""
Script d'entraînement du modèle v-JEPA.
Pré-entraînement auto-supervisé sur les vidéos de vrais visages.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, Optional

# Ajout du chemin src au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from src.models.jepa import VJEPAModel, VJEPAConfig
from src.data.video_dataset import VJEPADataset
from src.data.augmentation import create_vjepa_augmentations
from src.training.jepa_trainer import JEPATrainer, JEPATrainingConfig
from src.utils.logger import setup_logger, MultiLogger
from src.utils.config import ConfigManager

logger = logging.getLogger(__name__)

def parse_args():
    """Parse les arguments."""
    parser = argparse.ArgumentParser(description="Entraînement v-JEPA")
    
    parser.add_argument("--config", type=str, default="config/vjepa.yaml",
                       help="Fichier de configuration")
    parser.add_argument("--data_dir", type=str, default="./data/processed",
                       help="Répertoire des données")
    parser.add_argument("--checkpoint_dir", type=str, default="./results/checkpoints/jepa",
                       help="Répertoire des checkpoints")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Nombre d'époques")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Taille du batch")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                       help="Taux d'apprentissage")
    parser.add_argument("--resume", type=str, default=None,
                       help="Reprendre depuis un checkpoint")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Utiliser Weights & Biases")
    
    return parser.parse_args()

def create_model(config: Dict) -> VJEPAModel:
    """
    Crée le modèle v-JEPA.
    """
    vjepa_config = VJEPAConfig(
        input_size=tuple(config.get('input_size', [224, 224])),
        num_frames=config.get('num_frames', 16),
        embed_dim=config.get('embed_dim', 768),
        depth=config.get('depth', 12),
        num_heads=config.get('num_heads', 12),
        spatial_mask_ratio=config.get('spatial_mask_ratio', 0.75),
        temporal_mask_ratio=config.get('temporal_mask_ratio', 0.90),
        output_dim=config.get('output_dim', 512)
    )
    
    model = VJEPAModel(vjepa_config)
    logger.info(f"Modèle v-JEPA créé: {sum(p.numel() for p in model.parameters()):,} paramètres")
    
    return model

def create_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Crée les dataloaders.
    """
    # Dataset d'entraînement
    train_dataset = VJEPADataset(
        data_root=str(data_dir),
        split='train',
        num_frames=16,
        image_size=(224, 224),
        transform=create_vjepa_augmentations({})
    )
    
    # Dataset de validation
    val_dataset = VJEPADataset(
        data_root=str(data_dir),
        split='val',
        num_frames=16,
        image_size=(224, 224)
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {'train': train_loader, 'val': val_loader}

def main():
    """Fonction principale."""
    args = parse_args()
    
    # Logger
    logger = setup_logger(
        name="train_jepa",
        log_file="./results/logs/train_jepa.log"
    )
    
    # Multi-logger
    multi_logger = MultiLogger(
        use_tensorboard=True,
        use_wandb=args.use_wandb,
        tensorboard_dir="./results/tensorboard/jepa",
        wandb_project="deepfake-vjepa"
    )
    
    # Configuration
    config_manager = ConfigManager()
    config = config_manager.load_config([args.config])
    
    logger.info("Démarrage de l'entraînement v-JEPA")
    logger.info(f"Configuration: {args.config}")
    
    # Création du modèle
    model = create_model(config.vjepa.architecture)
    
    # Création des dataloaders
    dataloaders = create_dataloaders(
        Path(args.data_dir),
        args.batch_size
    )
    
    # Configuration de l'entraînement
    training_config = JEPATrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        spatial_mask_ratio=config.vjepa.masking.spatial_mask_ratio,
        temporal_mask_ratio=config.vjepa.masking.temporal_mask_ratio,
        loss_type=config.vjepa.loss.type,
        checkpoint_frequency=10
    )
    
    # Création de l'entraîneur
    trainer = JEPATrainer(
        model=model,
        config=training_config,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Entraînement
    logger.info("Démarrage de l'entraînement...")
    
    if args.resume:
        logger.info(f"Reprise depuis: {args.resume}")
        metrics = trainer.train(
            dataloaders['train'],
            dataloaders['val'],
            resume_from=args.resume
        )
    else:
        metrics = trainer.train(
            dataloaders['train'],
            dataloaders['val']
        )
    
    logger.info("Entraînement terminé")
    logger.info(f"Métriques finales: {metrics}")
    
    # Fermeture du logger
    multi_logger.close()

if __name__ == "__main__":
    main()