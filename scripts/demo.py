#!/usr/bin/env python3
"""
Script de démonstration en temps réel.
Détecte les deepfakes dans des vidéos en direct ou des fichiers.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional, Tuple
import time

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import cv2
import numpy as np

from src.models.jepa import VJEPAModel, VJEPAConfig
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.data.face_detector import create_face_detector
from src.data.preprocessing import create_preprocessor
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

def parse_args():
    """Parse les arguments."""
    parser = argparse.ArgumentParser(description="Démo de détection de deepfakes")
    
    parser.add_argument("--model_checkpoint", type=str, required=True,
                       help="Checkpoint du modèle")
    parser.add_argument("--source", type=str, default="0",
                       help="Source vidéo (0 pour webcam, ou chemin fichier)")
    parser.add_argument("--output", type=str, default=None,
                       help="Sauvegarder la vidéo de sortie")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Seuil de détection")
    parser.add_argument("--num_frames", type=int, default=16,
                       help="Nombre de frames pour l'analyse")
    parser.add_argument("--display", action="store_true",
                       help="Afficher la vidéo en direct")
    
    return parser.parse_args()

class DeepfakeDetector:
    """
    Détecteur de deepfakes en temps réel.
    """
    
    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.5,
        num_frames: int = 16,
        device: str = 'cuda'
    ):
        self.model = model.to(device).eval()
        self.threshold = threshold
        self.num_frames = num_frames
        self.device = device
        
        # Détecteur de visages
        self.face_detector = create_face_detector(method='mtcnn')
        
        # Buffer de frames
        self.frame_buffer = []
        
        # Statistiques
        self.num_detections = 0
        self.num_fakes = 0
        self.processing_times = []
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """
        Traite une frame et retourne la prédiction.
        
        Args:
            frame: Frame BGR (H, W, C)
            
        Returns:
            frame annotée, is_fake, confidence
        """
        start_time = time.time()
        
        # Détection du visage
        detections = self.face_detector.detect(frame)
        
        if not detections:
            # Pas de visage détecté
            return frame, False, 0.0
        
        # Prendre le visage le plus grand
        detection = max(detections, key=lambda d: (d['bbox'][2]-d['bbox'][0]) * (d['bbox'][3]-d['bbox'][1]))
        bbox = detection['bbox']
        
        # Extraction du visage
        x1, y1, x2, y2 = bbox
        face = frame[y1:y2, x1:x2]
        
        # Redimensionnement
        face = cv2.resize(face, (224, 224))
        
        # Conversion en tensor
        face_tensor = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
        face_tensor = face_tensor.unsqueeze(0)  # (1, C, H, W)
        
        # Ajout au buffer
        self.frame_buffer.append(face_tensor)
        
        if len(self.frame_buffer) > self.num_frames:
            self.frame_buffer.pop(0)
        
        # Prédiction si assez de frames
        if len(self.frame_buffer) == self.num_frames:
            # Création du clip vidéo
            clip = torch.stack(self.frame_buffer, dim=2)  # (1, C, T, H, W)
            clip = clip.to(self.device)
            
            # Prédiction
            with torch.no_grad():
                logits = self.model(clip)
                probabilities = F.softmax(logits, dim=1)
                
                fake_score = probabilities[0, 1].item()
                is_fake = fake_score > self.threshold
                
            # Mise à jour des statistiques
            self.num_detections += 1
            if is_fake:
                self.num_fakes += 1
            
            confidence = fake_score
        else:
            is_fake = False
            confidence = 0.0
        
        # Annotation de la frame
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Dessin de la boîte
        color = (0, 0, 255) if is_fake else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Texte
        label = "FAKE" if is_fake else "REAL"
        cv2.putText(
            frame,
            f"{label}: {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
        
        # FPS
        avg_time = np.mean(self.processing_times[-30:]) if self.processing_times else 0
        fps = 1 / avg_time if avg_time > 0 else 0
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        return frame, is_fake, confidence
    
    def get_statistics(self) -> Dict:
        """Retourne les statistiques de détection."""
        return {
            'total_detections': self.num_detections,
            'fake_detections': self.num_fakes,
            'fake_ratio': self.num_fakes / max(1, self.num_detections),
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0
        }

def main():
    """Fonction principale."""
    args = parse_args()
    
    # Logger
    logger = setup_logger(name="demo", log_file="./results/logs/demo.log")
    
    # Chargement du modèle
    checkpoint = torch.load(args.model_checkpoint, map_location='cpu')
    vjepa_config = VJEPAConfig()
    jepa_model = VJEPAModel(vjepa_config)
    detector_model = HybridDeepfakeDetector(
        vjepa_model=jepa_model,
        forensic_analyzer=None,
        jepa_dim=512,
        forensic_dim=0,
        num_classes=2
    )
    detector_model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Modèle chargé depuis {args.model_checkpoint}")
    
    # Création du détecteur
    detector = DeepfakeDetector(
        model=detector_model,
        threshold=args.threshold,
        num_frames=args.num_frames
    )
    
    # Ouverture de la source vidéo
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        logger.error(f"Impossible d'ouvrir la source: {source}")
        return
    
    logger.info(f"Source vidéo ouverte: {source}")
    
    # Configuration de la sortie
    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Boucle de traitement
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Traitement
            annotated_frame, is_fake, confidence = detector.process_frame(frame)
            
            # Affichage
            if args.display:
                cv2.imshow('Deepfake Detection', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Sauvegarde
            if writer:
                writer.write(annotated_frame)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                stats = detector.get_statistics()
                logger.info(f"Frame {frame_count}: {stats}")
    
    except KeyboardInterrupt:
        logger.info("Interruption par l'utilisateur")
    
    finally:
        # Nettoyage
        cap.release()
        if writer:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
        
        # Statistiques finales
        stats = detector.get_statistics()
        logger.info(f"Statistiques finales: {stats}")
        logger.info("Démo terminée")

if __name__ == "__main__":
    main()