#!/usr/bin/env python3
"""
Script d'entraînement du détecteur de deepfakes.
Utilise les features v-JEPA pré-entraînées pour la classification.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.models.jepa import VJEPAModel, VJEPAConfig
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.models.anomaly_detector import create_anomaly_detector
from src.data.video_dataset import VideoFaceDataset
from src.training.trainer import BaseTrainer, TrainingConfig
from src.utils.logger import setup_logger
from src.utils.config import ConfigManager

logger = logging.getLogger(__name__)

def parse_args():
    """Parse les arguments."""
    parser = argparse.ArgumentParser(description="Entraînement du détecteur")
    
    parser.add_argument("--config", type=str, default="config/training.yaml")
    parser.add_argument("--data_dir", type=str, default="./data/processed")
    parser.add_argument("--jepa_checkpoint", type=str, required=True,
                       help="Checkpoint du modèle v-JEPA pré-entraîné")
    parser.add_argument("--checkpoint_dir", type=str, default="./results/checkpoints/detector")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--freeze_jepa", action="store_true",
                       help="Geler les poids v-JEPA")
    parser.add_argument("--use_anomaly", action="store_true",
                       help="Utiliser la détection d'anomalies")
    
    return parser.parse_args()

def load_pretrained_jepa(checkpoint_path: str) -> VJEPAModel:
    """
    Charge le modèle v-JEPA pré-entraîné.
    """
    config = VJEPAConfig()
    model = VJEPAModel(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Modèle v-JEPA chargé depuis {checkpoint_path}")
    
    return model

def create_detector(
    jepa_model: VJEPAModel,
    freeze_jepa: bool = False
) -> nn.Module:
    """
    Crée le détecteur hybride.
    """
    if freeze_jepa:
        for param in jepa_model.parameters():
            param.requires_grad = False
        logger.info("Poids v-JEPA gelés")
    
    detector = HybridDeepfakeDetector(
        vjepa_model=jepa_model,
        forensic_analyzer=None,  # Optionnel
        jepa_dim=512,
        forensic_dim=0,
        fusion_type='attention',
        num_classes=2
    )
    
    return detector

def create_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Crée les dataloaders pour l'entraînement du détecteur.
    """
    # Datasets
    train_dataset = VideoFaceDataset(
        data_root=str(data_dir),
        split='train',
        num_frames=16,
        image_size=(224, 224)
    )
    
    val_dataset = VideoFaceDataset(
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
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {'train': train_loader, 'val': val_loader}

def train_anomaly_detector(
    jepa_model: VJEPAModel,
    train_loader: DataLoader,
    method: str = 'mahalanobis'
) -> nn.Module:
    """
    Entraîne le détecteur d'anomalies sur les features v-JEPA.
    """
    logger.info(f"Entraînement du détecteur d'anomalies ({method})")
    
    # Extraction des features des vrais visages
    jepa_model.eval()
    real_features = []
    
    with torch.no_grad():
        for batch in train_loader:
            if isinstance(batch, dict):
                x = batch['frames']
                y = batch['label']
            else:
                x, y = batch
            
            # Filtrer les vrais visages
            x_real = x[y == 0]
            
            if len(x_real) > 0:
                features = jepa_model.encode(x_real.cuda())
                real_features.append(features.cpu())
    
    real_features = torch.cat(real_features, dim=0)
    logger.info(f"Features réelles extraites: {real_features.shape}")
    
    # Création et entraînement du détecteur
    detector = create_anomaly_detector(method=method)
    detector.fit(real_features)
    
    return detector

def main():
    """Fonction principale."""
    args = parse_args()
    
    # Logger
    logger = setup_logger(
        name="train_detector",
        log_file="./results/logs/train_detector.log"
    )
    
    logger.info("Démarrage de l'entraînement du détecteur")
    
    # Chargement du modèle v-JEPA
    jepa_model = load_pretrained_jepa(args.jepa_checkpoint)
    
    # Création du détecteur
    detector = create_detector(jepa_model, args.freeze_jepa)
    
    # Création des dataloaders
    dataloaders = create_dataloaders(
        Path(args.data_dir),
        args.batch_size
    )
    
    # Configuration de l'entraînement
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_checkpoints=True,
        checkpoint_frequency=5
    )
    
    # Entraîneur
    trainer = BaseTrainer(
        model=detector,
        config=training_config,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Entraînement
    logger.info("Entraînement du détecteur...")
    metrics = trainer.train(
        dataloaders['train'],
        dataloaders['val']
    )
    
    # Entraînement du détecteur d'anomalies
    if args.use_anomaly:
        anomaly_detector = train_anomaly_detector(
            jepa_model,
            dataloaders['train']
        )
        
        # Sauvegarde du détecteur d'anomalies
        torch.save(
            anomaly_detector,
            Path(args.checkpoint_dir) / "anomaly_detector.pth"
        )
        logger.info("Détecteur d'anomalies sauvegardé")
    
    logger.info("Entraînement terminé")
    logger.info(f"Métriques finales: {metrics}")

if __name__ == "__main__":
    main()