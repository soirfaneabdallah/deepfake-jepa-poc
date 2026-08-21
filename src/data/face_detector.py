"""
Détection et alignement des visages pour le prétraitement.
Supporte MTCNN, RetinaFace et MediaPipe.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Optional, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class FaceDetector(ABC):
    """Classe abstraite pour les détecteurs de visages."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
    
    @abstractmethod
    def detect(
        self,
        image: np.ndarray
    ) -> List[dict]:
        """
        Détecte les visages dans une image.
        
        Args:
            image: Image BGR (H, W, C)
            
        Returns:
            Liste de dictionnaires contenant:
                - 'bbox': (x1, y1, x2, y2)
                - 'confidence': float
                - 'landmarks': (5, 2) si disponible
        """
        pass
    
    def extract_faces(
        self,
        frames: torch.Tensor,
        margin: int = 20,
        align: bool = True
    ) -> torch.Tensor:
        """
        Extrait et aligne les visages d'un clip vidéo.
        
        Args:
            frames: (C, T, H, W) - Clip vidéo
            margin: Marge autour du visage
            align: Aligner les visages
            
        Returns:
            faces: (C, T, H', W') - Visages extraits
        """
        C, T, H, W = frames.shape
        
        # Conversion en numpy pour la détection
        frames_np = frames.permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
        frames_np = (frames_np * 255).astype(np.uint8)
        
        faces = []
        for t in range(T):
            frame = frames_np[t]
            face = self._extract_single_face(frame, margin, align)
            if face is not None:
                faces.append(face)
            else:
                # Utiliser la frame originale
                faces.append(frame)
        
        # Conversion en tensor
        faces = np.stack(faces)  # (T, H', W', C)
        faces = torch.from_numpy(faces).permute(3, 0, 1, 2).float() / 255.0
        
        return faces
    
    def _extract_single_face(
        self,
        image: np.ndarray,
        margin: int,
        align: bool
    ) -> Optional[np.ndarray]:
        """
        Extrait un seul visage d'une image.
        """
        detections = self.detect(image)
        
        if not detections:
            return None
        
        # Prendre la détection avec la plus haute confiance
        detection = max(detections, key=lambda x: x['confidence'])
        bbox = detection['bbox']
        
        # Ajouter la marge
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        # Extraire le visage
        face = image[y1:y2, x1:x2]
        
        # Alignement si demandé
        if align and 'landmarks' in detection:
            face = self._align_face(face, detection['landmarks'])
        
        return face
    
    def _align_face(
        self,
        face: np.ndarray,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Aligne le visage basé sur les landmarks.
        """
        # Points de référence (yeux)
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        # Calcul de l'angle de rotation
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Centre des yeux
        center = ((left_eye[0] + right_eye[0]) / 2,
                  (left_eye[1] + right_eye[1]) / 2)
        
        # Matrice de rotation
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Application de la rotation
        aligned = cv2.warpAffine(
            face, M, (face.shape[1], face.shape[0]),
            flags=cv2.INTER_CUBIC
        )
        
        return aligned

class MTCNNDetector(FaceDetector):
    """
    Détecteur de visages MTCNN.
    Utilise facenet-pytorch pour une implémentation efficace.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        margin: int = 20,
        min_face_size: int = 50,
        thresholds: List[float] = [0.6, 0.7, 0.7],
        factor: float = 0.709,
        device: str = 'cuda'
    ):
        super().__init__(device)
        
        from facenet_pytorch import MTCNN
        
        self.detector = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            thresholds=thresholds,
            factor=factor,
            keep_all=True,
            device=self.device
        )
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Détecte les visages avec MTCNN.
        """
        # Conversion en tensor
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Détection
        boxes, probs, landmarks = self.detector.detect(img_tensor, landmarks=True)
        
        detections = []
        if boxes is not None:
            for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                detections.append({
                    'bbox': box.astype(int).tolist(),
                    'confidence': float(prob),
                    'landmarks': landmark
                })
        
        return detections

class RetinaFaceDetector(FaceDetector):
    """
    Détecteur de visages RetinaFace.
    Utilise l'implémentation de insightface.
    """
    
    def __init__(self, device: str = 'cuda'):
        super().__init__(device)
        
        try:
            from insightface.app import FaceAnalysis
            self.detector = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider'] if 'cuda' in device else ['CPUExecutionProvider']
            )
            self.detector.prepare(ctx_id=0 if 'cuda' in device else -1)
        except ImportError:
            logger.error("InsightFace non installé. Utilisez MTCNN à la place.")
            self.detector = None
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Détecte les visages avec RetinaFace.
        """
        if self.detector is None:
            return []
        
        # Conversion BGR en RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Détection
        faces = self.detector.get(rgb_image)
        
        detections = []
        for face in faces:
            bbox = face.bbox.astype(int)
            detections.append({
                'bbox': bbox.tolist(),
                'confidence': float(face.det_score),
                'landmarks': face.kps if hasattr(face, 'kps') else None
            })
        
        return detections

class MediaPipeDetector(FaceDetector):
    """
    Détecteur de visages MediaPipe.
    Rapide et précis pour les visages frontaux.
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        device: str = 'cuda'
    ):
        super().__init__(device)
        
        import mediapipe as mp
        
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 = courte distance, 1 = longue distance
            min_detection_confidence=min_detection_confidence
        )
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Détecte les visages avec MediaPipe.
        """
        # Conversion BGR en RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Détection
        results = self.detector.process(rgb_image)
        
        detections = []
        if results.detections:
            h, w = image.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Conversion en coordonnées absolues
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                
                # Landmarks (6 points)
                landmarks = []
                for keypoint in detection.location_data.relative_keypoints:
                    landmarks.append([
                        keypoint.x * w,
                        keypoint.y * h
                    ])
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(detection.score[0]),
                    'landmarks': np.array(landmarks) if landmarks else None
                })
        
        return detections

def create_face_detector(
    method: str = 'mtcnn',
    **kwargs
) -> FaceDetector:
    """
    Factory pour créer un détecteur de visages.
    """
    detectors = {
        'mtcnn': MTCNNDetector,
        'retinaface': RetinaFaceDetector,
        'mediapipe': MediaPipeDetector
    }
    
    if method not in detectors:
        raise ValueError(f"Détecteur non supporté: {method}")
    
    return detectors[method](**kwargs)