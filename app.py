"""
app.py - Forest Fire Detection Flask App
Ensemble: CNN (MobileNetV3) + GradientBoosting + Color Heuristic
"""

import os, io, base64, numpy as np, pickle, torch, json
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from PIL import Image
import timm
import torchvision.transforms as transforms

app = Flask(__name__)
DEVICE = torch.device("cpu")

# ── Load CNN Model (architecture read from saved metadata) ────────
cnn_model     = None
cnn_transform = None

try:
    with open("models/cnn_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    model_name = meta.get("model_name", "mobilenetv3_small_075")
    img_size   = meta.get("img_size", 224)

    cnn_model = timm.create_model(model_name, pretrained=False, num_classes=2)
    cnn_model.load_state_dict(torch.load("models/cnn_fire_model.pth", map_location=DEVICE))
    cnn_model.eval()
    cnn_model.to(DEVICE)

    cnn_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print(f"✅ CNN loaded  →  {model_name}  (img_size={img_size})")

except FileNotFoundError:
    print("⚠ CNN model not found. Run: python train_cnn.py")
except Exception as e:
    print(f"⚠ CNN model failed to load: {e}")

# ── Load GradientBoosting Image Model ────────────────────────────
gb_bundle = None
try:
    with open("models/image_model.pkl", "rb") as f:
        gb_bundle = pickle.load(f)
    print("✅ GradientBoosting model loaded")
except FileNotFoundError:
    print("⚠ GB model not found. Run: python train_models.py")
except Exception as e:
    print(f"⚠ GB model failed to load: {e}")

# ── Feature extractor (must match train_models.py exactly) ───────
def extract_features(img_array):
    img = img_array.astype(np.float32) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    feats = []

    for ch in (r, g, b):
        h, _ = np.histogram(ch, bins=16, range=(0, 1))
        feats.extend(h / (h.sum() + 1e-6))

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v    = maxc
    s    = (maxc - minc) / (maxc + 1e-6)   # fixed: no divide-by-zero
    diff = maxc - minc + 1e-6
    hue  = np.where(maxc == r, (g - b) / diff % 6,
           np.where(maxc == g, (b - r) / diff + 2,
                               (r - g) / diff + 4)) / 6.0

    for ch in (hue, s, v):
        feats += [float(ch.mean()), float(ch.std())]
        h, _ = np.histogram(ch, bins=8, range=(0, 1))
        feats.extend(h / (h.sum() + 1e-6))

    fire   = (r > 0.48) & (g < r * 0.92) & (b < r * 0.62)
    smoke  = (s < 0.45) & (v > 0.25) & (v < 0.95) & \
             (np.abs(r - g) < 0.18) & (np.abs(g - b) < 0.18) & (r.mean() > 0.25)
    bright = (r > 0.80) & (g > 0.35) & (b < 0.30)
    cool   = (r < 0.35) & (g > r) & (b < 0.4)
    rd     = r - 0.5 * (g + b)

    feats += [
        float(fire.mean()), float(smoke.mean()),
        float(bright.mean()), float(cool.mean()),
        float(rd.mean()), float(np.percentile(rd, 75)),
        float(np.percentile(rd, 90)),
        float(r.var()), float(g.var()), float(b.var())
    ]
    return np.array(feats, dtype=np.float32)

# ── Color heuristic (fast pixel-level fire check) ─────────────────
def color_heuristic(img_array):
    """Returns (is_fire: bool, confidence: float 0-1)"""
    img = img_array.astype(np.float32) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    fire_core   = (r > 0.50) & (g < r * 0.90) & (b < r * 0.55)
    bright_fire = (r > 0.80) & (g > 0.40) & (b < 0.30)

    fire_ratio = float((fire_core | bright_fire).mean())
    is_fire    = fire_ratio > 0.03           # >3% of pixels look like fire
    confidence = min(fire_ratio / 0.03, 1.0) # saturates at 3%
    return is_fire, confidence

# ── In-memory fire event log ──────────────────────────────────────
fire_events = []

# ── Routes ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/map")
def map_page():
    return render_template("map.html")

@app.route("/predict_image", methods=["POST"])
def predict_image():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image uploaded"})

        file    = request.files["image"]
        img_pil = Image.open(file.stream).convert("RGB")
        lat     = request.form.get("lat")
        lng     = request.form.get("lng")

        # votes = (is_fire, fire_confidence 0-1, source, weight)
        votes = []

        # ── Vote 1: Color heuristic (weight=2, always runs) ──────
        arr_256  = np.array(img_pil.resize((256, 256)), dtype=np.uint8)
        h_fire, h_conf = color_heuristic(arr_256)
        votes.append((h_fire, h_conf, "Heuristic", 2))
        print(f"[Heuristic] fire={h_fire}, conf={h_conf:.3f}")

        # ── Vote 2: GradientBoosting (weight=2) ──────────────────
        if gb_bundle is not None:
            arr_128      = np.array(img_pil.resize((128, 128)), dtype=np.uint8)
            feats        = extract_features(arr_128).reshape(1, -1)
            feats_scaled = gb_bundle["scaler"].transform(feats)
            gb_probs     = gb_bundle["model"].predict_proba(feats_scaled)[0]
            gb_fire_prob = float(gb_probs[1])       # label 1 = fire
            gb_fire      = gb_fire_prob > 0.45       # slightly below 0.5: fire-safe bias
            votes.append((gb_fire, gb_fire_prob, "GB", 2))
            print(f"[GB]  fire_prob={gb_fire_prob:.3f} → {'FIRE' if gb_fire else 'NO FIRE'}")

        # ── Vote 3: CNN (weight=1) ────────────────────────────────
        if cnn_model is not None and cnn_transform is not None:
            input_tensor = cnn_transform(img_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output    = cnn_model(input_tensor)
                prob      = torch.softmax(output, dim=1)[0]
                fire_prob = float(prob[1])             # index 1 = fire (matches training labels)
                cnn_fire  = fire_prob > 0.75            # high threshold: CNN can overcall
            votes.append((cnn_fire, fire_prob, "CNN", 1))
            print(f"[CNN] fire_prob={fire_prob:.3f} → {'FIRE' if cnn_fire else 'NO FIRE'}")

        # ── Weighted ensemble ─────────────────────────────────────
        total_weight = sum(w for _, _, _, w in votes)
        fire_weight  = sum(w for is_f, _, _, w in votes if is_f)

        # Tie → FIRE  (safety bias: false alarm is safer than missed fire)
        final_is_fire = fire_weight >= (total_weight - fire_weight)

        winning_confs = [c for is_f, c, _, _ in votes if is_f == final_is_fire]
        confidence    = round(float(np.mean(winning_confs)) * 100, 1) if winning_confs else 0.0

        label = "🔥 FIRE DETECTED" if final_is_fire else "✅ NO FIRE DETECTED"
        print(f"[Ensemble] fire_weight={fire_weight}/{total_weight} → {label} ({confidence}%)\n")

        # ── Log fire event ────────────────────────────────────────
        if final_is_fire and confidence > 60:
            fire_events.append({
                "lat":        float(lat) if lat else 18.52,
                "lng":        float(lng) if lng else 73.85,
                "confidence": confidence,
                "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "thumbnail":  ""
            })
            if len(fire_events) > 50:
                fire_events.pop(0)

        # ── Thumbnail ─────────────────────────────────────────────
        buf = io.BytesIO()
        img_pil.resize((300, 200)).save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            "success":    True,
            "label":      label,
            "class":      int(final_is_fire),
            "confidence": confidence,
            "thumbnail":  f"data:image/jpeg;base64,{img_b64}",
            "lat":        lat,
            "lng":        lng,
            "votes": [
                {"source": s, "fire": f, "conf": round(c * 100, 1), "weight": w}
                for f, c, s, w in votes
            ]
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/get_fire_events", methods=["GET"])
def get_fire_events():
    return jsonify({"events": fire_events})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)