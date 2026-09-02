#!/usr/bin/env python3
"""
Script d'évaluation complète du modèle.
Teste les performances, la robustesse et la généralisation.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional
import json

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report

from src.models.jepa import VJEPAModel, VJEPAConfig
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.data.video_dataset import VideoFaceDataset
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.roc_analysis import ROCAnalyzer, ThresholdOptimizer
from src.evaluation.robustness import RobustnessEvaluator, PerturbationEvaluator
from src.evaluation.cross_dataset import CrossDatasetEvaluator
from src.utils.logger import setup_logger
from src.utils.visualize import plot_confusion_matrix, plot_metrics_comparison

logger = logging.getLogger(__name__)

def parse_args():
    """Parse les arguments."""
    parser = argparse.ArgumentParser(description="Évaluation du modèle")
    
    parser.add_argument("--model_checkpoint", type=str, required=True,
                       help="Checkpoint du modèle")
    parser.add_argument("--data_dir", type=str, default="./data/processed",
                       help="Répertoire des données")
    parser.add_argument("--output_dir", type=str, default="./results/evaluation",
                       help="Répertoire de sortie")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_robustness", action="store_true",
                       help="Tester la robustesse")
    parser.add_argument("--cross_dataset", action="store_true",
                       help="Évaluation cross-dataset")
    
    return parser.parse_args()

def load_model(checkpoint_path: str) -> nn.Module:
    """
    Charge le modèle depuis un checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Création du modèle
    vjepa_config = VJEPAConfig()
    jepa_model = VJEPAModel(vjepa_config)
    detector = HybridDeepfakeDetector(
        vjepa_model=jepa_model,
        forensic_analyzer=None,
        jepa_dim=512,
        forensic_dim=0,
        num_classes=2
    )
    
    # Chargement des poids
    detector.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Modèle chargé depuis {checkpoint_path}")
    
    return detector

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Évalue le modèle sur le jeu de test.
    """
    model.eval()
    all_preds = []
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                x = batch['frames']
                y = batch['label']
            else:
                x, y = batch
            
            x = x.to(device)
            
            # Forward pass
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(y.numpy())
    
    # Calcul des métriques
    metrics = compute_classification_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_scores)
    )
    
    return metrics.to_dict(), all_labels, all_preds, all_scores

def main():
    """Fonction principale."""
    args = parse_args()
    
    # Logger
    logger = setup_logger(
        name="evaluate",
        log_file="./results/logs/evaluate.log"
    )
    
    # Création du répertoire de sortie
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Démarrage de l'évaluation")
    
    # Chargement du modèle
    model = load_model(args.model_checkpoint)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Création du dataloader de test
    test_dataset = VideoFaceDataset(
        data_root=args.data_dir,
        split='test',
        num_frames=16,
        image_size=(224, 224)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Évaluation
    logger.info("Évaluation sur le jeu de test...")
    metrics, y_true, y_pred, y_scores = evaluate_model(
        model, test_loader, device
    )
    
    logger.info("Métriques de test:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Sauvegarde des métriques
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Matrice de confusion
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(
        cm,
        save_path=str(output_dir / "confusion_matrix.png")
    )
    
    # Courbe ROC
    roc_analyzer = ROCAnalyzer(save_dir=str(output_dir))
    roc_analyzer.plot_roc(
        np.array(y_true),
        np.array(y_scores),
        save_path=str(output_dir / "roc_curve.png")
    )
    
    # Optimisation du seuil
    threshold_optimizer = ThresholdOptimizer()
    optimal_thresholds = threshold_optimizer.optimize(
        np.array(y_true),
        np.array(y_scores)
    )
    
    with open(output_dir / "optimal_thresholds.json", "w") as f:
        json.dump(optimal_thresholds, f, indent=2)
    
    # Test de robustesse
    if args.test_robustness:
        logger.info("Test de robustesse...")
        
        robust_evaluator = RobustnessEvaluator(
            model,
            device=device,
            save_dir=str(output_dir / "robustness")
        )
        
        # Perturbations
        perturb_evaluator = PerturbationEvaluator()
        
        # Création des loaders perturbés
        perturbed_loaders = {}
        
        # TODO: Implémenter les perturbations sur le dataset
        
        robustness_results = robust_evaluator.evaluate(
            test_loader,
            perturbed_loaders
        )
    
    # Évaluation cross-dataset
    if args.cross_dataset:
        logger.info("Évaluation cross-dataset...")
        
        cross_evaluator = CrossDatasetEvaluator(
            model,
            device=device,
            save_dir=str(output_dir / "cross_dataset")
        )
        
        # TODO: Charger les autres datasets
        
    logger.info("Évaluation terminée")
    logger.info(f"Résultats sauvegardés dans {output_dir}")

if __name__ == "__main__":
    main()