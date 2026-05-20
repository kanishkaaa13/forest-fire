"""
train_cnn.py  –  Train MobileNetV3 CNN for forest fire detection
Usage:
    python train_cnn.py
    python train_cnn.py --model efficientnet_b0 --epochs 15 --batch 32
"""

import os, random, argparse, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import classification_report

# ── Args ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",      default="mobilenetv3_small_075")
parser.add_argument("--img-size",   type=int, default=224)
parser.add_argument("--epochs",     type=int, default=10)
parser.add_argument("--batch",      type=int, default=32)
parser.add_argument("--lr",         type=float, default=1e-3)
parser.add_argument("--max-per-class", type=int, default=10000)
parser.add_argument("--image-root", default="dataset/images")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("models", exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

class FireDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (args.img_size, args.img_size))
        return self.transform(img), label


def load_samples(root, max_per_class):
    """
    Expects:
        root/fire/   → label 1
        root/nofire/ → label 0   (also accepts 'no_fire', 'normal', 'not_fire')
    Returns balanced list of (path, label) tuples.
    """
    FIRE_NAMES   = {"fire"}
    NOFIRE_NAMES = {"nofire", "no_fire", "normal", "not_fire", "nofire"}

    samples = {0: [], 1: []}

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Image root not found: {root}")

    for cls_dir in sorted(os.listdir(root)):
        full  = os.path.join(root, cls_dir)
        cname = cls_dir.lower().strip()
        if not os.path.isdir(full):
            continue
        if cname in FIRE_NAMES:
            label = 1
        elif cname in NOFIRE_NAMES:
            label = 0
        else:
            print(f"  ⚠  Unknown class folder '{cls_dir}' — skipping")
            continue

        files = [os.path.join(full, f) for f in os.listdir(full)
                 if f.lower().endswith(EXTS)]
        random.shuffle(files)
        files = files[:max_per_class]
        samples[label].extend(files)
        print(f"  '{cls_dir}' → label={label}  ({len(files)} images loaded)")

    # ── Class balance check ────────────────────────────────────────
    n0, n1 = len(samples[0]), len(samples[1])
    print(f"\n  Class balance:  NO FIRE={n0}  |  FIRE={n1}")
    if n0 == 0 or n1 == 0:
        raise ValueError("One class has 0 images — check your dataset/images/ folders.")
    if max(n0, n1) / (min(n0, n1) + 1e-6) > 3:
        print("  ⚠  Classes are heavily imbalanced (>3x ratio). Consider balancing.")

    # Pair up and shuffle
    combined = [(p, 0) for p in samples[0]] + [(p, 1) for p in samples[1]]
    random.shuffle(combined)
    return combined


# ── Transforms ─────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Train ───────────────────────────────────────────────────────────
def train_model():
    print(f"\n🚀 Training on: {DEVICE} | Model: {args.model}")
    print(f"   Images per class: {args.max_per_class} | Total: {args.max_per_class * 2}\n")

    # Load & split
    all_samples = load_samples(args.image_root, args.max_per_class)
    split       = int(len(all_samples) * 0.8)
    train_s, val_s = all_samples[:split], all_samples[split:]

    train_ds = FireDataset(train_s, train_tf)
    val_ds   = FireDataset(val_s,   val_tf)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)

    # Model
    model = timm.create_model(args.model, pretrained=True, num_classes=2)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc  = 0.0
    best_path = "models/cnn_fire_model.pth"

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for step, (imgs, labels) in enumerate(train_dl, 1):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if step % 50 == 0:
                avg = train_loss / step
                print(f"  Epoch {epoch}/{args.epochs}  step {step}/{len(train_dl)}  loss={avg:.4f}", end="\r")

        scheduler.step()

        # ── Val ──
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += len(labels)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_acc = correct / total * 100
        print(f"✨ Epoch {epoch} | Loss: {train_loss/len(train_dl):.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  💾 Saved best model ({val_acc:.2f}%)")

    # ── Final report ───────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Best Val Accuracy: {best_acc:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=["No Fire", "Fire"]))

    # ── Save metadata ──────────────────────────────────────────────
    meta = {
        "model_name": args.model,
        "img_size":   args.img_size,
        "num_classes": 2,
        "classes":    ["nofire", "fire"],   # index 0 = nofire, index 1 = fire
        "best_val_acc": best_acc,
    }
    with open("models/cnn_meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print("✅ Metadata saved to models/cnn_meta.pkl")
    print(f"✅ Model saved  to {best_path}")


if __name__ == "__main__":
    train_model()