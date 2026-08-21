"""
Gestion de la configuration du projet.
Fournit des outils pour charger, valider et manipuler les configurations.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration d'expérience."""
    name: str = "experiment"
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "./outputs"
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

class ConfigManager:
    """
    Gestionnaire de configuration avec fusion et validation.
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config: Optional[DictConfig] = None
        self.config_files: List[str] = []
    
    def load_config(
        self,
        config_files: Optional[List[str]] = None,
        overrides: Optional[Dict[str, Any]] = None,
        validate: bool = True
    ) -> DictConfig:
        """
        Charge et fusionne les fichiers de configuration.
        
        Args:
            config_files: Liste des fichiers YAML à charger
            overrides: Dictionnaire de surcharges
            validate: Valider la configuration
            
        Returns:
            Configuration fusionnée
        """
        if config_files is None:
            config_files = ["default.yaml"]
        
        self.config_files = config_files
        
        # Chargement de base
        config = OmegaConf.create()
        
        # Fusion des fichiers
        for config_file in config_files:
            file_path = self.config_dir / config_file
            if file_path.exists():
                file_config = OmegaConf.load(file_path)
                config = OmegaConf.merge(config, file_config)
                logger.info(f"Configuration chargée: {config_file}")
            else:
                logger.warning(f"Fichier de configuration manquant: {config_file}")
        
        # Application des surcharges
        if overrides:
            override_config = OmegaConf.create(overrides)
            config = OmegaConf.merge(config, override_config)
        
        # Validation
        if validate:
            self._validate_config(config)
        
        self.config = config
        return config
    
    def _validate_config(self, config: DictConfig) -> None:
        """
        Valide la configuration.
        """
        # Vérification des chemins
        if hasattr(config, 'paths'):
            for path_name, path_value in config.paths.items():
                if isinstance(path_value, str):
                    Path(path_value).parent.mkdir(parents=True, exist_ok=True)
        
        # Vérification des hyperparamètres
        if hasattr(config, 'training'):
            if config.training.batch_size <= 0:
                raise ValueError("batch_size doit être positif")
            
            if config.training.learning_rate <= 0:
                raise ValueError("learning_rate doit être positif")
        
        logger.info("Configuration validée avec succès")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur de configuration.
        
        Args:
            key: Clé (ex: "training.learning_rate")
            default: Valeur par défaut
            
        Returns:
            Valeur de configuration
        """
        if self.config is None:
            return default
        
        return OmegaConf.select(self.config, key, default=default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Définit une valeur de configuration.
        """
        if self.config is None:
            raise RuntimeError("Configuration non chargée")
        
        OmegaConf.update(self.config, key, value)
    
    def save_config(self, output_path: str) -> None:
        """
        Sauvegarde la configuration fusionnée.
        """
        if self.config is None:
            raise RuntimeError("Configuration non chargée")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        OmegaConf.save(self.config, output_path)
        logger.info(f"Configuration sauvegardée: {output_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit la configuration en dictionnaire Python.
        """
        if self.config is None:
            raise RuntimeError("Configuration non chargée")
        
        return OmegaConf.to_container(self.config, resolve=True)
    
    def to_yaml(self) -> str:
        """
        Convertit la configuration en YAML.
        """
        if self.config is None:
            raise RuntimeError("Configuration non chargée")
        
        return OmegaConf.to_yaml(self.config)

def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None
) -> DictConfig:
    """
    Charge une configuration depuis un fichier YAML.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable: {config_path}")
    
    config = OmegaConf.load(config_path)
    
    if overrides:
        override_config = OmegaConf.create(overrides)
        config = OmegaConf.merge(config, override_config)
    
    return config

def save_config(
    config: DictConfig,
    output_path: Union[str, Path]
) -> None:
    """
    Sauvegarde une configuration en YAML.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    OmegaConf.save(config, output_path)

def merge_configs(
    base_config: DictConfig,
    override_config: DictConfig
) -> DictConfig:
    """
    Fusionne deux configurations.
    """
    return OmegaConf.merge(base_config, override_config)

class ConfigValidator:
    """
    Validateur de configuration.
    """
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        self.schema = schema or {}
    
    def validate(self, config: DictConfig) -> List[str]:
        """
        Valide la configuration contre le schéma.
        
        Returns:
            Liste des erreurs de validation
        """
        errors = []
        
        for key, expected_type in self.schema.items():
            if OmegaConf.select(config, key) is None:
                errors.append(f"Clé manquante: {key}")
            elif not isinstance(OmegaConf.select(config, key), expected_type):
                errors.append(
                    f"Type incorrect pour {key}: "
                    f"attendu {expected_type}, "
                    f"obtenu {type(OmegaConf.select(config, key))}"
                )
        
        return errors