"""
train_cnn.py  –  Train MobileNetV3-Small CNN for forest fire detection
Uses torchvision (NOT timm) so saved weights are directly compatible with app.py.

Usage:
    python train_cnn.py
    python train_cnn.py --epochs 20 --batch 32 --lr 1e-4
"""

import os, random, argparse, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
from sklearn.metrics import classification_report

# ── Args ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--img-size",      type=int,   default=224)
parser.add_argument("--epochs",        type=int,   default=15)
parser.add_argument("--batch",         type=int,   default=32)
parser.add_argument("--lr",            type=float, default=1e-4)
parser.add_argument("--max-per-class", type=int,   default=10000)
parser.add_argument("--image-root",    default="dataset/images")
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
    Returns list of (path, label) tuples, split into (train, val).
    """
    FIRE_NAMES   = {"fire"}
    NOFIRE_NAMES = {"nofire", "no_fire", "normal", "not_fire"}

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

    n0, n1 = len(samples[0]), len(samples[1])
    print(f"\n  Class balance:  NO FIRE={n0}  |  FIRE={n1}")
    if n0 == 0 or n1 == 0:
        raise ValueError("One class has 0 images — check your dataset/images/ folders.")

    ratio = max(n0, n1) / (min(n0, n1) + 1e-6)
    if ratio > 3:
        print(f"  ⚠  Classes are imbalanced ({ratio:.1f}x ratio) — class weights will be applied.")

    combined = [(p, 0) for p in samples[0]] + [(p, 1) for p in samples[1]]
    random.shuffle(combined)
    return combined, n0, n1


# ── Transforms ─────────────────────────────────────────────────────
# ImageNet normalization — MUST match app.py exactly
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
val_tf = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ── Build model ────────────────────────────────────────────────────
def build_model():
    """
    Load torchvision MobileNetV3-Small with ImageNet weights,
    then replace the final linear layer for 2-class (fire / no-fire) output.

    This architecture is IDENTICAL to what app.py loads, so state_dict
    keys will always match perfectly (strict=True).
    """
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    # Replace the final classification head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)

    return model


# ── Training loop ───────────────────────────────────────────────────
def train_model():
    print(f"\n🚀 Training on: {DEVICE}")
    print(f"   Architecture : torchvision mobilenet_v3_small (ImageNet pretrained)")
    print(f"   Epochs: {args.epochs}  |  Batch: {args.batch}  |  LR: {args.lr}")
    print(f"   Input size: {args.img_size}x{args.img_size}\n")

    # Load & split
    all_samples, n0, n1 = load_samples(args.image_root, args.max_per_class)
    split  = int(len(all_samples) * 0.8)
    train_s, val_s = all_samples[:split], all_samples[split:]
    print(f"\n  Train: {len(train_s)}  |  Val: {len(val_s)}\n")

    train_ds = FireDataset(train_s, train_tf)
    val_ds   = FireDataset(val_s,   val_tf)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)

    # Model
    model = build_model()
    model.to(DEVICE)

    # Class-weighted loss to handle imbalance
    total  = n0 + n1
    w0     = total / (2.0 * n0) if n0 > 0 else 1.0
    w1     = total / (2.0 * n1) if n1 > 0 else 1.0
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(DEVICE)
    print(f"  Class weights → NO FIRE: {w0:.3f}  |  FIRE: {w1:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Cosine annealing over full training run
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc  = 0.0
    best_path = "models/cnn_fire_model.pth"

    # Store last epoch predictions for final report
    last_preds, last_labels = [], []

    for epoch in range(1, args.epochs + 1):
        # ── Train phase ──
        model.train()
        train_loss = 0.0
        for step, (imgs, labels) in enumerate(train_dl, 1):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if step % 20 == 0 or step == len(train_dl):
                avg = train_loss / step
                print(f"  Epoch {epoch}/{args.epochs}  step {step}/{len(train_dl)}  loss={avg:.4f}", end="\r")

        scheduler.step()

        # ── Validation phase ──
        model.eval()
        correct, total_val = 0, 0
        preds_ep, labels_ep = [], []
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(dim=1)
                correct     += (preds == labels).sum().item()
                total_val   += len(labels)
                preds_ep.extend(preds.cpu().tolist())
                labels_ep.extend(labels.cpu().tolist())

        val_acc = correct / total_val * 100
        print(f"\n✨ Epoch {epoch}/{args.epochs} | Loss: {train_loss/len(train_dl):.4f} | Val Acc: {val_acc:.2f}%")

        # Per-class accuracy
        for cls_idx, cls_name in enumerate(["NO FIRE", "FIRE"]):
            cls_mask = [l == cls_idx for l in labels_ep]
            if any(cls_mask):
                cls_correct = sum(p == l for p, l in zip(preds_ep, labels_ep) if l == cls_idx)
                cls_total   = sum(cls_mask)
                print(f"   {cls_name}: {cls_correct}/{cls_total} = {cls_correct/cls_total*100:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  💾 Saved best model ({val_acc:.2f}%)")

        last_preds, last_labels = preds_ep, labels_ep

    # ── Final report ───────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"✅ Best Val Accuracy: {best_acc:.2f}%")
    print("\nFinal validation classification report:")
    print(classification_report(last_labels, last_preds, target_names=["No Fire", "Fire"]))

    # ── Verification: reload and check strict loading works ────────
    print("🔍 Verifying saved weights load cleanly into app.py architecture...")
    verify_model = mobilenet_v3_small(weights=None)
    verify_model.classifier[3] = nn.Linear(
        verify_model.classifier[3].in_features, 2
    )
    state = torch.load(best_path, map_location="cpu")
    missing, unexpected = verify_model.load_state_dict(state, strict=True)
    if missing or unexpected:
        print(f"  ❌ PROBLEM: missing={missing}, unexpected={unexpected}")
    else:
        print("  ✅ Weights verified — strict load successful, app.py will load this model correctly.")

    # ── Save metadata ──────────────────────────────────────────────
    meta = {
        "model":        "mobilenet_v3_small",   # torchvision identifier
        "input_size":   args.img_size,
        "classes":      ["nofire", "fire"],      # index 0 = nofire, index 1 = fire
        "best_val_acc": best_acc,
        "trained_with": "torchvision",           # NOT timm
    }
    with open("models/cnn_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"\n✅ Metadata saved  → models/cnn_meta.pkl")
    print(f"✅ Model weights   → {best_path}")
    print(f"\n📌 Next steps:")
    print(f"   1. Run: python test_false_positives.py  (validate on test images)")
    print(f"   2. git add models/cnn_fire_model.pth models/cnn_meta.pkl && git commit -m 'retrain: torchvision CNN'")
    print(f"   3. git push  →  Render auto-deploys")


if __name__ == "__main__":
    train_model()