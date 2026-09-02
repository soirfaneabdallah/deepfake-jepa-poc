#!/usr/bin/env python3
"""
Reprise du téléchargement interrompu avec support de resume.
"""

import os
import sys
import logging
from pathlib import Path
import requests
from tqdm import tqdm
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_with_resume(url, output_path, max_retries=5):
    """
    Télécharge un fichier avec support de reprise.
    """
    output_path = Path(output_path)
    temp_path = output_path.with_suffix('.part')
    
    # Vérifier si un téléchargement partiel existe
    resume_pos = 0
    if temp_path.exists():
        resume_pos = temp_path.stat().st_size
        logger.info(f"Reprise à partir de {resume_pos / (1024**3):.2f} GB")
    
    headers = {}
    if resume_pos > 0:
        headers['Range'] = f'bytes={resume_pos}-'
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Tentative {attempt + 1}/{max_retries}")
            
            response = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=60,
                allow_redirects=True
            )
            
            if response.status_code == 416:
                # Déjà complet
                logger.info("Fichier déjà complet")
                temp_path.rename(output_path)
                return True
            
            response.raise_for_status()
            
            # Taille totale
            total_size = int(response.headers.get('content-length', 0))
            if resume_pos > 0:
                total_size += resume_pos
            
            # Mode d'ouverture
            mode = 'ab' if resume_pos > 0 else 'wb'
            
            with tqdm(
                total=total_size,
                initial=resume_pos,
                unit='B',
                unit_scale=True,
                desc=output_path.name
            ) as pbar:
                with open(temp_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                            resume_pos += len(chunk)
            
            # Vérification
            if total_size > 0 and resume_pos >= total_size:
                logger.info("Téléchargement complet")
                temp_path.rename(output_path)
                return True
            else:
                logger.warning("Téléchargement incomplet, nouvelle tentative...")
                headers['Range'] = f'bytes={resume_pos}-'
                
        except Exception as e:
            logger.error(f"Erreur : {e}")
            if attempt < max_retries - 1:
                import time
                wait_time = 5 * (attempt + 1)
                logger.info(f"Attente de {wait_time} secondes...")
                time.sleep(wait_time)
            else:
                logger.error("Échec après toutes les tentatives")
                return False
    
    return False

def resume_kaggle_download():
    """
    Reprend le téléchargement du dataset Kaggle interrompu.
    """
    # Le fichier partiel est dans le cache kagglehub
    cache_dir = Path.home() / ".cache" / "kagglehub" / "datasets" / "xhlulu" / "140k-real-and-fake-faces"
    
    logger.info(f"Recherche dans : {cache_dir}")
    
    if cache_dir.exists():
        # Lister les fichiers
        for file in cache_dir.iterdir():
            logger.info(f"Trouvé : {file.name} ({file.stat().st_size / (1024**3):.2f} GB)")
            
            if file.suffix == '.archive' or file.name.endswith('.part'):
                logger.info(f"Fichier partiel détecté : {file}")
                
                # URL de téléchargement (à récupérer)
                # Pour l'instant, essayons de continuer avec le cache
                
    return None

def main():
    """
    Fonction principale avec options.
    """
    print("=" * 60)
    print("Reprise du téléchargement interrompu")
    print("=" * 60)
    
    # Option 1 : Reprendre depuis le cache
    print("\n1. Reprendre depuis le cache kagglehub")
    print("2. Télécharger un dataset plus petit")
    print("3. Créer un dataset synthétique")
    print("4. Télécharger depuis une autre source")
    
    choice = input("\nChoisissez une option (1-4) : ")
    
    if choice == '1':
        resume_kaggle_download()
    elif choice == '2':
        download_smaller_dataset()
    elif choice == '3':
        create_synthetic()
    elif choice == '4':
        download_alternative()
    else:
        print("Choix invalide")

def download_smaller_dataset():
    """
    Télécharge un dataset plus petit et plus fiable.
    """
    logger.info("Téléchargement d'un dataset plus petit...")
    
    # Dataset Deepfake Detection (plus petit, ~500 MB)
    try:
        import kagglehub
        path = kagglehub.dataset_download("bugraokcu/deepfake-detection")
        logger.info(f"✓ Téléchargé : {path}")
        
        # Copie
        target = Path("./data/raw/deepfake_detection")
        shutil.copytree(path, target, dirs_exist_ok=True)
        logger.info(f"✓ Copié vers : {target}")
        
    except Exception as e:
        logger.error(f"Erreur : {e}")
        logger.info("Essayez le dataset synthétique")

def create_synthetic():
    """
    Crée un dataset synthétique.
    """
    logger.info("Création du dataset synthétique...")
    
    import cv2
    import numpy as np
    
    output_dir = Path("./data/processed")
    num_videos = 50
    num_frames = 16
    image_size = (224, 224)
    
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            num_videos_split = int(num_videos * {'train': 0.7, 'val': 0.15, 'test': 0.15}[split])
            
            for i in tqdm(range(num_videos_split), desc=f"{split}/{label}"):
                video_path = output_dir / split / label / f"video_{i:03d}.mp4"
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(video_path), fourcc, 10, image_size)
                
                for frame_idx in range(num_frames):
                    frame = np.zeros((*image_size, 3), dtype=np.uint8)
                    
                    if label == 'real':
                        # Visage réel simulé
                        cv2.ellipse(frame, (112, 112), (80, 100), 0, 0, 360, (200, 160, 140), -1)
                        cv2.circle(frame, (85, 90), 10, (255, 255, 255), -1)
                        cv2.circle(frame, (140, 90), 10, (255, 255, 255), -1)
                    else:
                        # Visage fake avec artefacts
                        cv2.ellipse(frame, (112, 112), (80, 100), 0, 0, 360, (200, 160, 140), -1)
                        cv2.circle(frame, (85, 90), 12, (255, 255, 255), -1)
                        cv2.circle(frame, (140, 90), 8, (255, 255, 255), -1)
                        noise = np.random.normal(0, 20, frame.shape)
                        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
                    
                    writer.write(frame)
                
                writer.release()
    
    logger.info(f"✓ Dataset créé : {output_dir}")

def download_alternative():
    """
    Télécharge depuis Hugging Face.
    """
    logger.info("Téléchargement depuis Hugging Face...")
    
    try:
        from datasets import load_dataset
        
        # Dataset plus petit
        dataset = load_dataset("prithivirajdamodaran/deepfake-detection", split="train")
        logger.info(f"✓ Chargé : {len(dataset)} échantillons")
        
    except Exception as e:
        logger.error(f"Erreur : {e}")

if __name__ == "__main__":
    main()