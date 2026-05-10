"""
train_cnn.py - MobileNetV3 for Fire/Smoke Detection (Windows Fixed)
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import timm
import pickle
import random
from sklearn.model_selection import train_test_split

# ====================== IMPORTANT FOR WINDOWS ======================
if __name__ == '__main__':
    # This guard is REQUIRED on Windows
    torch.multiprocessing.freeze_support()

os.makedirs("models", exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16          # Reduced for CPU stability
EPOCHS = 6
NUM_WORKERS = 0          # Set to 0 on Windows to avoid multiprocessing issues
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

class FireDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Transforms
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_image_paths(root="dataset/images"):
    paths, labels = [], []
    for cls in ["fire", "nofire"]:
        folder = os.path.join(root, cls)
        if not os.path.isdir(folder):
            continue
        label = 1 if cls == "fire" else 0
        for f in os.listdir(folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(folder, f))
                labels.append(label)
    return paths, labels

# Load and balance dataset
image_paths, labels = get_image_paths()
print(f"Total images found - Fire: {sum(labels)} | NoFire: {len(labels)-sum(labels)}")

# Balance the dataset (important for good training)
fire_paths = [p for p, l in zip(image_paths, labels) if l == 1]
nofire_paths = [p for p, l in zip(image_paths, labels) if l == 0]

min_size = min(len(fire_paths), len(nofire_paths), 12000)
fire_paths = fire_paths[:min_size]
nofire_paths = nofire_paths[:min_size]

balanced_paths = fire_paths + nofire_paths
balanced_labels = [1] * len(fire_paths) + [0] * len(nofire_paths)

print(f"Balanced dataset: {len(balanced_paths)} images (50% fire)")

# Split
train_paths, val_paths, train_labels, val_labels = train_test_split(
    balanced_paths, balanced_labels, test_size=0.2, stratify=balanced_labels, random_state=42)

train_ds = FireDataset(train_paths, train_labels, transform_train)
val_ds = FireDataset(val_paths, val_labels, transform_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Model
model = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

print("\nStarting CNN Training...\n")

best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Validation Accuracy: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "models/cnn_fire_model.pth")
        print(f"   → New best model saved! ({acc:.2f}%)")

print(f"\n✅ Training Finished! Best Accuracy: {best_acc:.2f}%")

# Save metadata
with open("models/cnn_meta.pkl", "wb") as f:
    pickle.dump({"img_size": IMG_SIZE}, f)

print("Model saved to models/cnn_fire_model.pth")