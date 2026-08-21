"""
Configuration du logging pour le projet.
Fournit des loggers pour la console, TensorBoard et Weights & Biases.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import json
import torch
from torch.utils.tensorboard import SummaryWriter

@dataclass
class LoggerConfig:
    """Configuration du logging."""
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_file: Optional[str] = None
    use_tensorboard: bool = False
    tensorboard_dir: str = "./logs/tensorboard"
    use_wandb: bool = False
    wandb_project: str = "deepfake-vjepa"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    capture_warnings: bool = True

class ColorFormatter(logging.Formatter):
    """
    Formatter avec couleurs pour la console.
    """
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Vert
        'WARNING': '\033[33m',   # Jaune
        'ERROR': '\033[31m',     # Rouge
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Ajout de la couleur
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        
        return super().format(record)

def setup_logger(
    name: str = "deepfake_vjepa",
    config: Optional[LoggerConfig] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure et retourne un logger.
    
    Args:
        name: Nom du logger
        config: Configuration du logging
        log_file: Chemin du fichier de log
        
    Returns:
        Logger configuré
    """
    config = config or LoggerConfig()
    
    # Création du logger
    logger = logging.getLogger(name)
    logger.setLevel(config.level)
    logger.propagate = False
    
    # Suppression des handlers existants
    logger.handlers.clear()
    
    # Handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(config.level)
    console_formatter = ColorFormatter(config.format, config.date_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Handler fichier
    if log_file or config.log_file:
        file_path = log_file or config.log_file
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(config.level)
        file_formatter = logging.Formatter(config.format, config.date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Capture des warnings
    if config.capture_warnings:
        logging.captureWarnings(True)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Retourne un logger existant ou en crée un nouveau.
    """
    return logging.getLogger(name)

class TensorBoardLogger:
    """
    Logger TensorBoard pour le suivi des métriques.
    """
    
    def __init__(self, log_dir: str = "./logs/tensorboard"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Création d'un répertoire avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(self.log_dir / timestamp)
        
        self.step = 0
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        """Log une valeur scalaire."""
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: Optional[int] = None) -> None:
        """Log plusieurs valeurs scalaires."""
        if step is None:
            step = self.step
        self.writer.add_scalars(tag, values, step)
    
    def log_image(self, tag: str, image: torch.Tensor, step: Optional[int] = None) -> None:
        """Log une image."""
        if step is None:
            step = self.step
        self.writer.add_image(tag, image, step)
    
    def log_video(self, tag: str, video: torch.Tensor, step: Optional[int] = None) -> None:
        """Log une vidéo."""
        if step is None:
            step = self.step
        self.writer.add_video(tag, video, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: Optional[int] = None) -> None:
        """Log un histogramme."""
        if step is None:
            step = self.step
        self.writer.add_histogram(tag, values, step)
    
    def log_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> None:
        """Log le graphe du modèle."""
        self.writer.add_graph(model, input_tensor)
    
    def increment_step(self) -> None:
        """Incrémente le compteur de pas."""
        self.step += 1
    
    def close(self) -> None:
        """Ferme le writer."""
        self.writer.close()

class WandbLogger:
    """
    Logger Weights & Biases pour le suivi des expériences.
    """
    
    def __init__(
        self,
        project: str = "deepfake-vjepa",
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        try:
            import wandb
            
            self.wandb = wandb
            
            # Initialisation
            self.run = wandb.init(
                project=project,
                entity=entity,
                config=config,
                tags=tags or [],
                name=name
            )
        except ImportError:
            raise ImportError("wandb n'est pas installé. Utilisez 'pip install wandb'")
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log des métriques."""
        self.wandb.log(metrics, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = "", step: Optional[int] = None) -> None:
        """Log des métriques avec préfixe."""
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        self.log(metrics, step)
    
    def log_image(self, image: torch.Tensor, caption: str = "") -> None:
        """Log une image."""
        self.wandb.log({caption: self.wandb.Image(image)})
    
    def log_video(self, video: torch.Tensor, caption: str = "") -> None:
        """Log une vidéo."""
        self.wandb.log({caption: self.wandb.Video(video)})
    
    def log_table(self, table_name: str, data: List[List[Any]], columns: List[str]) -> None:
        """Log une table."""
        table = self.wandb.Table(data=data, columns=columns)
        self.wandb.log({table_name: table})
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "model") -> None:
        """Log un artefact."""
        artifact = self.wandb.Artifact(
            name=f"{artifact_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            type=artifact_type
        )
        artifact.add_file(artifact_path)
        self.wandb.log_artifact(artifact)
    
    def finish(self) -> None:
        """Termine le run."""
        self.wandb.finish()

class MultiLogger:
    """
    Logger multiple combinant TensorBoard et Wandb.
    """
    
    def __init__(
        self,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        tensorboard_dir: str = "./logs/tensorboard",
        wandb_project: str = "deepfake-vjepa",
        wandb_config: Optional[Dict[str, Any]] = None
    ):
        self.loggers = []
        
        if use_tensorboard:
            self.tensorboard_logger = TensorBoardLogger(tensorboard_dir)
            self.loggers.append(self.tensorboard_logger)
        else:
            self.tensorboard_logger = None
        
        if use_wandb:
            self.wandb_logger = WandbLogger(
                project=wandb_project,
                config=wandb_config
            )
            self.loggers.append(self.wandb_logger)
        else:
            self.wandb_logger = None
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log des métriques sur tous les loggers."""
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.log_scalar(key, value, step)
            elif isinstance(logger, WandbLogger):
                logger.log(metrics, step)
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        """Log une valeur scalaire."""
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.log_scalar(tag, value, step)
            elif isinstance(logger, WandbLogger):
                logger.log({tag: value}, step)
    
    def close(self) -> None:
        """Ferme tous les loggers."""
        for logger in self.loggers:
            if hasattr(logger, 'close'):
                logger.close()
            elif hasattr(logger, 'finish'):
                logger.finish()