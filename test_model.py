"""
test_model.py  –  Test your trained CNN on any image
Usage:
    python test_model.py path/to/image.jpg
    python test_model.py                      # runs on all jpg/png in current folder
"""

import sys, os, pickle
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms

DEVICE = torch.device("cpu")

# ── Load metadata ──────────────────────────────────────────────────
try:
    with open("models/cnn_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    model_name = meta.get("model_name", "mobilenetv3_small_075")
    img_size   = meta.get("img_size", 224)
except FileNotFoundError:
    model_name = "mobilenetv3_small_075"
    img_size   = 224
    print("⚠  cnn_meta.pkl not found – using defaults")

# ── Load model ─────────────────────────────────────────────────────
print(f"Loading model: {model_name}  (img_size={img_size})")
model = timm.create_model(model_name, pretrained=False, num_classes=2)
model.load_state_dict(torch.load("models/cnn_fire_model.pth", map_location=DEVICE))
model.eval()
model.to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Inference function ─────────────────────────────────────────────
def predict(image_path: str) -> dict:
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    fire_prob   = float(probs[1])
    nofire_prob = float(probs[0])
    label       = "🔥 FIRE" if fire_prob > 0.5 else "✅ NO FIRE"

    return {
        "path":       image_path,
        "label":      label,
        "fire_prob":  fire_prob,
        "nofire_prob": nofire_prob,
    }

# ── Main ───────────────────────────────────────────────────────────
def main():
    paths = sys.argv[1:]

    # If no args, scan current directory
    if not paths:
        exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        paths = [f for f in os.listdir(".") if f.lower().endswith(exts)]
        if not paths:
            print("Usage: python test_model.py image1.jpg image2.jpg ...")
            return

    print(f"\n{'─'*60}")
    print(f"{'Image':<35} {'Label':<12} {'Fire %':>8}  {'NoFire %':>9}")
    print(f"{'─'*60}")

    for p in paths:
        if not os.path.isfile(p):
            print(f"  ⚠  File not found: {p}")
            continue
        r = predict(p)
        name = os.path.basename(r["path"])[:34]
        print(f"  {name:<34} {r['label']:<12} {r['fire_prob']*100:>7.2f}%  {r['nofire_prob']*100:>8.2f}%")

    print(f"{'─'*60}\n")

if __name__ == "__main__":
    main()