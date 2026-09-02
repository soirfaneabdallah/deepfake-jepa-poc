import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import kagglehub
import os
import matplotlib.pyplot as plt
import numpy as np

class FaceDataset(Dataset):
    def __init__(self, root=None, split='train', transform=None, use_colab_cache=True):
        if root is None and use_colab_cache:
            try:
                root = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
                print(f"Using Colab cache: {root}")
            except:
                root = Path('./data/raw')
        
        self.root = Path(root)
        self.transform = transform
        self.images = []
        self.labels = []
        
        self._load_images(split)
        
        print(f"Dataset loaded: {len(self.images)} images")
        print(f"Real: {sum(1 for l in self.labels if l == 0)}")
        print(f"Fake: {sum(1 for l in self.labels if l == 1)}")
    
    def _load_images(self, split):
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in self.root.rglob(ext):
                parent = img_path.parent.name.lower()
                if 'real' in parent:
                    self.images.append(img_path)
                    self.labels.append(0)
                elif 'fake' in parent:
                    self.images.append(img_path)
                    self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]
    
    # === NOUVELLES MÉTHODES DE VISUALISATION ===
    
    def show_sample(self, num_images=8, figsize=(15, 6)):
        """Affiche un échantillon aléatoire d'images"""
        indices = torch.randperm(len(self))[:num_images]
        
        fig, axes = plt.subplots(1, num_images, figsize=figsize)
        if num_images == 1:
            axes = [axes]
        
        for i, idx in enumerate(indices):
            img, label = self[idx]
            
            if isinstance(img, torch.Tensor):
                img_np = img.numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)
            else:
                img_np = img
            
            axes[i].imshow(img_np)
            axes[i].set_title(f"{'Real' if label == 0 else 'Fake'}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def show_real_vs_fake(self, num_samples=4, figsize=(12, 8)):
        """Affiche des images réelles et fakes côte à côte"""
        real_indices = [i for i, l in enumerate(self.labels) if l == 0][:num_samples]
        fake_indices = [i for i, l in enumerate(self.labels) if l == 1][:num_samples]
        
        fig, axes = plt.subplots(2, num_samples, figsize=figsize)
        
        for idx, img_idx in enumerate(real_indices):
            img, _ = self[img_idx]
            if isinstance(img, torch.Tensor):
                img_np = img.numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)
            else:
                img_np = img
            axes[0, idx].imshow(img_np)
            axes[0, idx].set_title(f'Real {idx+1}')
            axes[0, idx].axis('off')
        
        for idx, img_idx in enumerate(fake_indices):
            img, _ = self[img_idx]
            if isinstance(img, torch.Tensor):
                img_np = img.numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)
            else:
                img_np = img
            axes[1, idx].imshow(img_np)
            axes[1, idx].set_title(f'Fake {idx+1}')
            axes[1, idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def show_class_distribution(self, figsize=(8, 5)):
        """Affiche la distribution des classes"""
        real_count = sum(1 for l in self.labels if l == 0)
        fake_count = sum(1 for l in self.labels if l == 1)
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(['Real', 'Fake'], [real_count, fake_count], color=['green', 'red'])
        ax.set_ylabel('Number of images')
        ax.set_title('Class Distribution')
        
        for bar, count in zip(bars, [real_count, fake_count]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                   f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def show_image_with_path(self, idx, figsize=(6, 6)):
        """Affiche une image avec son chemin"""
        img, label = self[idx]
        
        if isinstance(img, torch.Tensor):
            img_np = img.numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
        else:
            img_np = img
        
        plt.figure(figsize=figsize)
        plt.imshow(img_np)
        label_text = 'Real' if label == 0 else 'Fake'
        path = str(self.images[idx])
        plt.title(f'{label_text}\n{path}')
        plt.axis('off')
        plt.show()

def get_dataloaders(batch_size=32, use_colab_cache=True):
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
    
    dataset = FaceDataset(transform=train_transform, use_colab_cache=use_colab_cache)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

def show_dataloader_batch(dataloader, num_images=8, figsize=(15, 6)):
    """Affiche un batch du dataloader"""
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]
    
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    if num_images == 1:
        axes = [axes]
    
    for i in range(num_images):
        img_np = images[i].numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"{'Real' if labels[i] == 0 else 'Fake'}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()