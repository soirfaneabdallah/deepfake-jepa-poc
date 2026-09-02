#!/usr/bin/env python3
"""
Script de prétraitement des données vidéo.
Extrait les frames, détecte les visages et crée les datasets.
"""

import argparse
import logging
from pathlib import Path
import sys
import json
from typing import Dict, List, Optional, Tuple
import numpy as np

# Ajout du chemin src au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.video_dataset import VideoFaceDataset
from src.data.face_detector import create_face_detector
from src.data.preprocessing import create_preprocessor
from src.data.video_processor import VideoProcessor
from src.utils.logger import setup_logger
from src.utils.config import ConfigManager

logger = logging.getLogger(__name__)

def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Prétraitement des données vidéo")
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/processed",
        help="Répertoire des données traitées"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/splits",
        help="Répertoire de sortie"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Chemin du fichier de configuration"
    )
    
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Nombre de frames par vidéo"
    )
    
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Taille des images"
    )
    
    parser.add_argument(
        "--face_detector",
        type=str,
        default="mtcnn",
        choices=["mtcnn", "retinaface", "mediapipe"],
        help="Détecteur de visages"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Nombre de workers pour le chargement"
    )
    
    return parser.parse_args()

def extract_frames_from_videos(
    video_dir: Path,
    output_dir: Path,
    video_processor: VideoProcessor,
    num_frames: int = 16,
    face_detector=None
) -> None:
    """
    Extrait les frames de toutes les vidéos.
    """
    video_files = list(video_dir.glob("*.mp4"))
    logger.info(f"Extraction des frames de {len(video_files)} vidéos")
    
    for video_file in tqdm(video_files, desc="Extraction des frames"):
        try:
            # Extraction des frames
            frames = video_processor.extract_frames(
                str(video_file),
                num_frames=num_frames
            )
            
            # Détection des visages
            if face_detector:
                frames = face_detector.extract_faces(frames)
            
            # Sauvegarde des frames
            output_path = output_dir / f"{video_file.stem}.pt"
            torch.save(frames, output_path)
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {video_file}: {e}")

def create_dataset_splits(
    data_dir: Path,
    output_dir: Path,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42
) -> Dict[str, Dict[str, List[str]]]:
    """
    Crée les splits train/val/test.
    """
    from sklearn.model_selection import train_test_split
    
    # Collecte des fichiers
    real_files = list((data_dir / "real").glob("*.pt"))
    fake_files = list((data_dir / "fake").glob("*.pt"))
    
    logger.info(f"Fichiers réels: {len(real_files)}")
    logger.info(f"Fichiers fake: {len(fake_files)}")
    
    # Split des données
    real_train, real_temp = train_test_split(
        real_files, test_size=1-split_ratio[0], random_state=seed
    )
    real_val, real_test = train_test_split(
        real_temp, test_size=split_ratio[2]/(split_ratio[1]+split_ratio[2]), 
        random_state=seed
    )
    
    fake_train, fake_temp = train_test_split(
        fake_files, test_size=1-split_ratio[0], random_state=seed
    )
    fake_val, fake_test = train_test_split(
        fake_temp, test_size=split_ratio[2]/(split_ratio[1]+split_ratio[2]), 
        random_state=seed
    )
    
    # Création des splits
    splits = {
        'train': {
            'real': [str(f) for f in real_train],
            'fake': [str(f) for f in fake_train]
        },
        'val': {
            'real': [str(f) for f in real_val],
            'fake': [str(f) for f in fake_val]
        },
        'test': {
            'real': [str(f) for f in real_test],
            'fake': [str(f) for f in fake_test]
        }
    }
    
    # Sauvegarde des splits
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    
    logger.info(f"Splits sauvegardés dans {output_dir / 'splits.json'}")
    
    return splits

def preprocess_all(
    data_dir: Path,
    output_dir: Path,
    args
) -> None:
    """
    Prétraite toutes les données.
    """
    # Initialisation des composants
    video_processor = VideoProcessor(
        target_size=(args.image_size, args.image_size)
    )
    
    face_detector = create_face_detector(
        method=args.face_detector,
        image_size=args.image_size
    )
    
    preprocessor = create_preprocessor(
        image_size=(args.image_size, args.image_size)
    )
    
    # Création des répertoires
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "real").mkdir(exist_ok=True)
    (output_dir / "fake").mkdir(exist_ok=True)
    
    # Traitement des vidéos réelles
    real_dir = data_dir / "train" / "real"
    if real_dir.exists():
        logger.info("Traitement des vidéos réelles...")
        extract_frames_from_videos(
            real_dir,
            output_dir / "real",
            video_processor,
            args.num_frames,
            face_detector
        )
    
    # Traitement des vidéos fake
    fake_dir = data_dir / "train" / "fake"
    if fake_dir.exists():
        logger.info("Traitement des vidéos fake...")
        extract_frames_from_videos(
            fake_dir,
            output_dir / "fake",
            video_processor,
            args.num_frames,
            face_detector
        )
    
    # Création des splits
    create_dataset_splits(output_dir, output_dir.parent / "splits")
    
    logger.info("Prétraitement terminé")

def main():
    """Fonction principale."""
    args = parse_args()
    
    # Configuration du logger
    logger = setup_logger(
        name="preprocess",
        log_file="./results/logs/preprocess.log"
    )
    
    # Conversion des chemins
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    logger.info("Démarrage du prétraitement")
    logger.info(f"Données: {data_dir}")
    logger.info(f"Sortie: {output_dir}")
    
    # Prétraitement
    preprocess_all(data_dir, output_dir, args)
    
    logger.info("Prétraitement terminé avec succès")

if __name__ == "__main__":
    main()