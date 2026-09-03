import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List


class FaceDataset(Dataset):
    """
    Dataset d'images pour la détection de deepfakes.
    Dataset: xhlulu/140k-real-and-fake-faces
    """
    
    def __init__(
        self,
        root: Optional[Path] = None,
        transform: Optional[T.Compose] = None,
        use_colab_cache: bool = True
    ):
        if root is None and use_colab_cache:
            try:
                root = Path(kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces"))
                print(f"Using Colab cache: {root}")
            except:
                root = Path('./data/raw')
        
        self.root = Path(root)
        self.transform = transform or self._default_transform()
        self.images = []
        self.labels = []
        self._load_images()
        
        print(f"Loaded: {len(self.images)} images")
        print(f"Real: {sum(1 for l in self.labels if l == 0)}")
        print(f"Fake: {sum(1 for l in self.labels if l == 1)}")
    
    def _default_transform(self) -> T.Compose:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_images(self):
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in self.root.rglob(ext):
                parent = img_path.parent.name.lower()
                if 'real' in parent:
                    self.images.append(img_path)
                    self.labels.append(0)
                elif 'fake' in parent:
                    self.images.append(img_path)
                    self.labels.append(1)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class VideoDataset(Dataset):
    """
    Convertit des images en séquences vidéo pour v-JEPA.
    Chaque image est répétée pour former une vidéo de T frames.
    """
    
    def __init__(
        self,
        image_dataset: FaceDataset,
        num_frames: int = 8,
        transform: Optional[T.Compose] = None
    ):
        self.image_dataset = image_dataset
        self.num_frames = num_frames
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.image_dataset[idx]
        # (C, H, W) -> (C, T, H, W)
        video = img.unsqueeze(1).repeat(1, self.num_frames, 1, 1)
        return video, label


def get_dataloaders(
    batch_size: int = 32,
    use_colab_cache: bool = True,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Crée les DataLoaders pour l'entraînement.
    """
    dataset = FaceDataset(use_colab_cache=use_colab_cache)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_video_dataloaders(
    batch_size: int = 16,
    num_frames: int = 8,
    use_colab_cache: bool = True,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Crée les DataLoaders pour v-JEPA (vidéos synthétiques).
    """
    image_dataset = FaceDataset(use_colab_cache=use_colab_cache)
    video_dataset = VideoDataset(image_dataset, num_frames=num_frames)
    
    train_size = int(0.8 * len(video_dataset))
    val_size = len(video_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(video_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def show_sample(dataset: FaceDataset, num_images: int = 8):
    """Affiche un échantillon d'images."""
    indices = torch.randperm(len(dataset))[:num_images]
    
    fig, axes = plt.subplots(1, num_images, figsize=(15, 4))
    if num_images == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        img_np = img.numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"{'Real' if label == 0 else 'Fake'}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def show_video_sample(dataset: VideoDataset, idx: int = 0, num_display: int = 8):
    """Affiche un échantillon de vidéo synthétique."""
    video, label = dataset[idx]
    # video: (C, T, H, W)
    
    fig, axes = plt.subplots(1, num_display, figsize=(15, 4))
    if num_display == 1:
        axes = [axes]
    
    T = video.size(1)
    indices = np.linspace(0, T - 1, num_display, dtype=int)
    
    for i, t in enumerate(indices):
        img = video[:, t, :, :].numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f'Frame {t}')
        axes[i].axis('off')
    
    plt.suptitle(f"{'Real' if label == 0 else 'Fake'}")
    plt.tight_layout()
    plt.show()