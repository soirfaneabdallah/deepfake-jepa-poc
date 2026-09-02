import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import kagglehub
import os

class FaceDataset(Dataset):
    def __init__(self, root=None, split='train', transform=None, use_colab_cache=True):
        """
        Dataset pour détection de deepfakes
        Args:
            root: chemin du dataset (si None, utilise le cache Colab)
            split: 'train', 'val', 'test'
            transform: transformations
            use_colab_cache: utiliser le cache Kaggle dans Colab
        """
        if root is None and use_colab_cache:
            # Télécharger depuis le cache Colab
            try:
                root = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
                print(f"📦 Using Colab cache: {root}")
            except:
                # Fallback: utiliser le chemin local
                root = Path('./data/raw')
        
        self.root = Path(root)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Charger les images
        self._load_images(split)
        
        print(f"✅ Dataset loaded: {len(self.images)} images")
        print(f"   Real: {sum(1 for l in self.labels if l == 0)}")
        print(f"   Fake: {sum(1 for l in self.labels if l == 1)}")
    
    def _load_images(self, split):
        """Charge les images avec recherche récursive"""
        # Recherche récursive pour trouver toutes les images
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in self.root.rglob(ext):
                parent = img_path.parent.name.lower()
                if 'real' in parent:
                    self.images.append(img_path)
                    self.labels.append(0)
                elif 'fake' in parent:
                    self.images.append(img_path)
                    self.labels.append(1)
        
        # Si on a un split, on peut filtrer (optionnel)
        # Pour l'instant, on garde tout
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def get_dataloaders(batch_size=32, use_colab_cache=True):
    """Crée les DataLoaders pour entraînement et validation"""
    
    # Transformations
    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Créer le dataset complet
    dataset = FaceDataset(transform=train_transform, use_colab_cache=use_colab_cache)
    
    # Split en train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Appliquer les bonnes transformations
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader