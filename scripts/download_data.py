import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import kagglehub

# Download the dataset
path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
root = Path(path)

class FaceDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = Path(root)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Common structures for this dataset
        possible_structures = [
            # Direct real/fake folders
            lambda: [(self.root / 'real', 0), (self.root / 'fake', 1)],
            # train/val split
            lambda: [(self.root / split / 'real', 0), (self.root / split / 'fake', 1)],
            # Dataset folder
            lambda: [(self.root / 'dataset' / split / 'real', 0), (self.root / 'dataset' / split / 'fake', 1)],
            # All in one folder with subfolders
            lambda: [(self.root / 'real' / split, 0), (self.root / 'fake' / split, 1)],
        ]
        
        found_any = False
        for structure_fn in possible_structures:
            try:
                for cls_dir, label in structure_fn():
                    if cls_dir and cls_dir.exists() and cls_dir.is_dir():
                        # Check if directory has images
                        images = list(cls_dir.glob('*.*'))
                        images = [img for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                        if images:
                            self.images.extend(images)
                            self.labels.extend([label] * len(images))
                            found_any = True
                            print(f"✓ Found {len(images)} images in {cls_dir}")
            except Exception as e:
                continue
        
        if not found_any:
            print("❌ No images found! Trying to search recursively...")
            # Fallback: search recursively for image files
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in self.root.rglob(ext):
                    # Determine label from parent folder name
                    parent = img_path.parent.name.lower()
                    if 'real' in parent:
                        self.images.append(img_path)
                        self.labels.append(0)
                    elif 'fake' in parent:
                        self.images.append(img_path)
                        self.labels.append(1)
            
            if self.images:
                print(f"✓ Found {len(self.images)} images via recursive search")
        
        # Print summary
        real_count = sum(1 for l in self.labels if l == 0)
        fake_count = sum(1 for l in self.labels if l == 1)
        print(f"Real images: {real_count}")
        print(f"Fake images: {fake_count}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Transformations
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset
dataset = FaceDataset(root, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"\nTotal images: {len(dataset)}")
print(f"Batches: {len(dataloader)}")