"""
train_models.py  –  Forest Fire Prediction
Trains:
  1. Tabular classifier + regressor  (Algerian Forest Fires CSV)
  2. Image-based GradientBoosting    (handcrafted color/HSV features)
Usage:
    python train_models.py
"""

import os, warnings, pickle, random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, classification_report
warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
TABULAR_FEATURES = ["Temperature","RH","Ws","Rain","FFMC","DMC","DC","ISI","BUI","FWI"]


# ══════════════════════════════════════════════════════════════════
#  TABULAR  –  Algerian Forest Fires CSV
# ══════════════════════════════════════════════════════════════════

def load_algerian_csv(path="dataset/Algerian_forest_fires_dataset.csv"):
    print(f"  Reading: {path}")
    with open(path, encoding="utf-8", errors="ignore") as f:
        raw_lines = f.readlines()

    COLS = ["day","month","year","Temperature","RH","Ws","Rain",
            "FFMC","DMC","DC","ISI","BUI","FWI","Classes"]
    rows = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if any(k in line.lower() for k in ["bejaia","sidi","region","dataset"]):
            continue
        if parts[0].lower() == "day":
            continue
        if len(parts) < 14:
            continue
        try:
            int(parts[0])
        except ValueError:
            continue
        rows.append(parts[:14])

    df = pd.DataFrame(rows, columns=COLS)
    for col in TABULAR_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Classes"] = df["Classes"].astype(str).str.strip().str.lower()
    df["Classes"] = df["Classes"].map({"fire": 1, "not fire": 0})
    df = df.dropna()
    df["Classes"] = df["Classes"].astype(int)
    print(f"  Final shape: {df.shape}")
    return df


def _synthetic(n=300):
    """Fallback when CSV is missing."""
    np.random.seed(42)
    rows = []
    for _ in range(n):
        t    = np.random.uniform(15, 42)
        rh   = np.random.uniform(20, 90)
        ws   = np.random.uniform(4,  29)
        rain = np.random.exponential(0.5)
        ffmc = np.random.uniform(28, 96)
        dmc  = np.random.uniform(1,  65)
        dc   = np.random.uniform(7,  220)
        isi  = np.random.uniform(0,  56)
        bui  = np.random.uniform(1,  68)
        fwi  = max(0, 0.3*isi + 0.2*bui + np.random.normal(0, 2))
        cls  = 1 if (t > 30 and rh < 45 and fwi > 15) else (1 if np.random.rand() < 0.3 else 0)
        rows.append([t, rh, ws, rain, ffmc, dmc, dc, isi, bui, fwi, cls])
    return pd.DataFrame(rows, columns=TABULAR_FEATURES + ["Classes"])


def train_tabular(csv_path="dataset/Algerian_forest_fires_dataset.csv"):
    print("\n─── Training Tabular Models ───")
    df = load_algerian_csv(csv_path) if os.path.exists(csv_path) else _synthetic()
    print(f"  Class distribution:\n{df['Classes'].value_counts().to_string()}")

    X     = df[TABULAR_FEATURES].values
    y_cls = df["Classes"].values
    y_reg = df["FWI"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # ── Classifier ──
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    clf = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    clf.fit(X_tr, y_tr)
    print(f"\n  Classifier Accuracy: {accuracy_score(y_te, clf.predict(X_te))*100:.1f}%")
    print(classification_report(y_te, clf.predict(X_te), target_names=["No Fire","Fire"]))

    # ── Regressor (FWI) ──
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_sc, y_reg, test_size=0.2, random_state=42)
    reg = Ridge(alpha=1.0)
    reg.fit(Xr_tr, yr_tr)
    print(f"  Regressor R²: {r2_score(yr_te, reg.predict(Xr_te)):.3f}")

    with open("models/scaler.pkl",     "wb") as f: pickle.dump(scaler, f)
    with open("models/classifier.pkl", "wb") as f: pickle.dump(clf,    f)
    with open("models/regressor.pkl",  "wb") as f: pickle.dump(reg,    f)
    print("  ✅ Tabular models saved.")


# ══════════════════════════════════════════════════════════════════
#  IMAGE  –  Handcrafted features + GradientBoosting
# ══════════════════════════════════════════════════════════════════

def extract_features(img_array):
    """
    Returns a 1-D float32 feature vector from a uint8 HxWx3 array.
    Must stay identical to the version used in app.py.
    """
    img = img_array.astype(np.float32) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    feats = []

    # RGB histograms
    for ch in (r, g, b):
        h, _ = np.histogram(ch, bins=16, range=(0, 1))
        feats.extend(h / (h.sum() + 1e-6))

    # HSV statistics
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v    = maxc
    s    = np.where(maxc > 0, (maxc - minc) / maxc, 0)
    diff = maxc - minc + 1e-6
    hue  = np.where(maxc == r, (g - b) / diff % 6,
           np.where(maxc == g, (b - r) / diff + 2,
                               (r - g) / diff + 4)) / 6.0

    for ch in (hue, s, v):
        feats += [float(ch.mean()), float(ch.std())]
        h, _ = np.histogram(ch, bins=8, range=(0, 1))
        feats.extend(h / (h.sum() + 1e-6))

    # Semantic masks
    fire   = (r > 0.48) & (g < r * 0.92) & (b < r * 0.62)
    smoke  = (s < 0.45) & (v > 0.25) & (v < 0.95) & \
             (np.abs(r - g) < 0.18) & (np.abs(g - b) < 0.18) & (r.mean() > 0.25)
    bright = (r > 0.80) & (g > 0.35) & (b < 0.30)
    cool   = (r < 0.35) & (g > r) & (b < 0.4)
    rd     = r - 0.5 * (g + b)

    feats += [
        float(fire.mean()),  float(smoke.mean()),
        float(bright.mean()), float(cool.mean()),
        float(rd.mean()), float(np.percentile(rd, 75)), float(np.percentile(rd, 90)),
        float(r.var()), float(g.var()), float(b.var()),
    ]
    return np.array(feats, dtype=np.float32)


def diagnose_images(image_root="dataset/images"):
    print("\n─── Image Folder Diagnostics ───")
    if not os.path.isdir(image_root):
        print(f"  ❌ {image_root} does not exist!")
        return
    for cls_dir in sorted(os.listdir(image_root)):
        full = os.path.join(image_root, cls_dir)
        if not os.path.isdir(full):
            continue
        img_files = [f for f in os.listdir(full)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
        print(f"  {cls_dir}/  →  {len(img_files)} image files")


def train_image_model(image_root="dataset/images", max_per_class=6000):
    print("\n─── Training Image Model ───")
    from PIL import Image as PILImage

    IMG_SIZE = 128
    EXTS     = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    X, y     = [], []

    if not os.path.isdir(image_root):
        print(f"  ❌ Folder not found: {image_root}")
        return None

    counts = {0: 0, 1: 0}
    for cls_dir in sorted(os.listdir(image_root)):
        full  = os.path.join(image_root, cls_dir)
        cname = cls_dir.lower().strip()
        if not os.path.isdir(full):
            continue

        # Accept common naming conventions
        if cname in ("fire",):
            label = 1
        elif cname in ("nofire", "no_fire", "normal", "not_fire"):
            label = 0
        else:
            print(f"  ⚠  Unknown folder '{cls_dir}' — skipping")
            continue

        files = [os.path.join(full, f) for f in os.listdir(full)
                 if f.lower().endswith(EXTS)]
        print(f"  '{cls_dir}' → label={label}  ({len(files)} files)")

        random.seed(42)
        random.shuffle(files)
        files = files[:max_per_class]

        ok, fail = 0, 0
        for fp in files:
            try:
                with PILImage.open(fp) as im:
                    im  = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                    arr = np.array(im, dtype=np.uint8)
                X.append(extract_features(arr))
                y.append(label)
                ok += 1
            except Exception as ex:
                fail += 1
                if fail <= 3:
                    print(f"    ⚠  {os.path.basename(fp)}: {ex}")
        counts[label] += ok
        print(f"    Loaded: {ok}  |  Failed: {fail}")

    if len(X) < 100:
        print(f"\n  ❌ Only {len(X)} images loaded — aborting image model training.")
        return None

    X, y = np.array(X), np.array(y)
    print(f"\n  Feature vector : {X.shape[1]} dims")
    print(f"  Total samples  : {len(X)}  |  Fire: {counts[1]}  |  No-fire: {counts[0]}")

    # Balance warning
    if counts[0] > 0 and counts[1] > 0:
        ratio = max(counts[0], counts[1]) / min(counts[0], counts[1])
        if ratio > 3:
            print(f"  ⚠  Class imbalance ratio = {ratio:.1f}x  — consider balancing your dataset")

    sc = StandardScaler()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr = sc.fit_transform(X_tr)
    X_te  = sc.transform(X_te)

    clf = GradientBoostingClassifier(
        n_estimators=250, learning_rate=0.08, max_depth=5,
        random_state=42, n_iter_no_change=10, validation_fraction=0.1)
    clf.fit(X_tr, y_tr)

    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"  Image Accuracy : {acc*100:.1f}%")
    print(classification_report(y_te, clf.predict(X_te), target_names=["No Fire","Fire"]))

    bundle = {"model": clf, "scaler": sc}
    with open("models/image_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print("  ✅ Image model saved → models/image_model.pkl")
    return clf


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def train_all():
    diagnose_images()
    train_tabular()
    train_image_model()
    print("\n🎉 All models trained!  Run:  python app.py")


if __name__ == "__main__":
    train_all()