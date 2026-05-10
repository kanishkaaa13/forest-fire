"""
train_models.py  –  Forest Fire Prediction (with added smoke dataset support)
"""

import os, warnings, pickle, glob, random
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

# ... (keep your existing load_algerian_csv and train_tabular functions unchanged) ...

def load_algerian_csv(path="dataset/Algerian_forest_fires_dataset.csv"):
    # (Your existing function - copy it from previous version)
    print(f"  Reading: {path}")
    with open(path, encoding="utf-8", errors="ignore") as f:
        raw_lines = f.readlines()
    COLS = ["day","month","year","Temperature","RH","Ws","Rain",
            "FFMC","DMC","DC","ISI","BUI","FWI","Classes"]
    rows = []
    for line in raw_lines:
        line = line.strip()
        if not line: continue
        parts = [p.strip() for p in line.split(",")]
        if any(k in line.lower() for k in ["bejaia","sidi","region","dataset"]): continue
        if parts[0].lower() == "day": continue
        if len(parts) < 14: continue
        try: int(parts[0])
        except ValueError: continue
        rows.append(parts[:14])
    df = pd.DataFrame(rows, columns=COLS)
    for col in TABULAR_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Classes"] = df["Classes"].astype(str).str.strip().str.lower()
    df["Classes"] = df["Classes"].map({"fire":1,"not fire":0})
    df = df.dropna()
    df["Classes"] = df["Classes"].astype(int)
    print(f"  Final shape: {df.shape}")
    return df

def train_tabular(csv_path="dataset/Algerian_forest_fires_dataset.csv"):
    # (Your existing tabular training function - keep it as is)
    print("\n─── Training Tabular Models ───")
    df = load_algerian_csv(csv_path) if os.path.exists(csv_path) else _synthetic()
    print(f"  Class distribution:\n{df['Classes'].value_counts().to_string()}")
    X = df[TABULAR_FEATURES].values
    y_cls = df["Classes"].values
    y_reg = df["FWI"].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    X_tr,X_te,y_tr,y_te = train_test_split(X_sc,y_cls,test_size=0.2,random_state=42,stratify=y_cls)
    clf = GradientBoostingClassifier(n_estimators=200,learning_rate=0.05,max_depth=4,random_state=42)
    clf.fit(X_tr,y_tr)
    print(f"\n  Classifier Accuracy: {accuracy_score(y_te,clf.predict(X_te))*100:.1f}%")
    print(classification_report(y_te,clf.predict(X_te),target_names=["No Fire","Fire"]))
    Xr_tr,Xr_te,yr_tr,yr_te = train_test_split(X_sc,y_reg,test_size=0.2,random_state=42)
    reg = Ridge(alpha=1.0)
    reg.fit(Xr_tr,yr_tr)
    print(f"  Regressor R²: {r2_score(yr_te,reg.predict(Xr_te)):.3f}")
    with open("models/scaler.pkl","wb") as f: pickle.dump(scaler,f)
    with open("models/classifier.pkl","wb") as f: pickle.dump(clf,f)
    with open("models/regressor.pkl","wb") as f: pickle.dump(reg,f)
    print("  ✅ Tabular models saved.")

def _synthetic(n=300):
    # (keep your existing synthetic function)
    np.random.seed(42)
    rows=[]
    for _ in range(n):
        t=np.random.uniform(15,42);rh=np.random.uniform(20,90)
        ws=np.random.uniform(4,29);rain=np.random.exponential(0.5)
        ffmc=np.random.uniform(28,96);dmc=np.random.uniform(1,65)
        dc=np.random.uniform(7,220);isi=np.random.uniform(0,56)
        bui=np.random.uniform(1,68);fwi=max(0,0.3*isi+0.2*bui+np.random.normal(0,2))
        cls=1 if(t>30 and rh<45 and fwi>15) else(1 if np.random.rand()<0.3 else 0)
        rows.append([t,rh,ws,rain,ffmc,dmc,dc,isi,bui,fwi,cls])
    return pd.DataFrame(rows,columns=TABULAR_FEATURES+["Classes"])

# Improved Feature Extractor (fixed division warning + better smoke)
def extract_features(img_array):
    img = img_array.astype(np.float32) / 255.0
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    feats = []
    
    for ch in (r, g, b):
        h, _ = np.histogram(ch, bins=16, range=(0,1))
        feats.extend(h / (h.sum() + 1e-6))
    
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    s = np.where(maxc > 0, (maxc - minc) / maxc, 0)
    diff = maxc - minc + 1e-6
    hue = np.where(maxc==r, (g-b)/diff % 6,
          np.where(maxc==g, (b-r)/diff + 2, (r-g)/diff + 4)) / 6.0
    
    for ch in (hue, s, v):
        feats += [float(ch.mean()), float(ch.std())]
        h, _ = np.histogram(ch, bins=8, range=(0,1))
        feats.extend(h / (h.sum() + 1e-6))
    
    # Enhanced smoke & fire detection
    fire  = (r > 0.48) & (g < r*0.92) & (b < r*0.62)
    smoke = (s < 0.45) & (v > 0.25) & (v < 0.95) & \
            (np.abs(r - g) < 0.18) & (np.abs(g - b) < 0.18) & (r.mean() > 0.25)
    bright = (r > 0.80) & (g > 0.35) & (b < 0.30)
    cool   = (r < 0.35) & (g > r) & (b < 0.4)
    rd = r - 0.5 * (g + b)
    
    feats += [float(fire.mean()), float(smoke.mean()), float(bright.mean()), float(cool.mean()),
              float(rd.mean()), float(np.percentile(rd, 75)), float(np.percentile(rd, 90)),
              float(r.var()), float(g.var()), float(b.var())]
    
    return np.array(feats, dtype=np.float32)

# Image training (keep your current version but use the new extract_features)
def train_image_model(image_root="dataset/images"):
    print("\n─── Training Image Model ───")
    from PIL import Image as PILImage

    IMG_SIZE = 128
    X, y = [], []

    if not os.path.isdir(image_root):
        print(f"  ❌ Folder not found: {image_root}")
        return None

    for cls_dir in sorted(os.listdir(image_root)):
        full = os.path.join(image_root, cls_dir)
        if not os.path.isdir(full): continue
        label = 1 if cls_dir.lower() == "fire" else 0

        files = [os.path.join(full, f) for f in os.listdir(full) 
                 if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))]
        print(f"  '{cls_dir}' → label={label}  ({len(files)} files)")

        random.seed(42)
        random.shuffle(files)
        files = files[:6000]

        ok, fail = 0, 0
        for fp in files:
            try:
                with PILImage.open(fp) as im:
                    im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                    arr = np.array(im, dtype=np.uint8)
                X.append(extract_features(arr))
                y.append(label)
                ok += 1
            except Exception as ex:
                fail += 1
                if fail <= 3:
                    print(f"    ⚠  {os.path.basename(fp)}: {ex}")
        print(f"    Loaded: {ok}  |  Failed: {fail}")

    if len(X) < 100:
        print(f"\n  ❌ Only {len(X)} images loaded.")
        return None

    X, y = np.array(X), np.array(y)
    print(f"\n  Feature vector : {X.shape[1]} dims")
    print(f"  Total samples  : {len(X)}  |  Fire: {sum(y)}  |  No-fire: {sum(1-y)}")

    sc = StandardScaler()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    clf = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.08,
        max_depth=5,
        random_state=42,
        n_iter_no_change=10,
        validation_fraction=0.1
    )
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"  Image Accuracy : {acc*100:.1f}%")
    print(classification_report(y_te, clf.predict(X_te), target_names=["No Fire","Fire"]))

    bundle = {"model": clf, "scaler": sc}
    with open("models/image_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print("  ✅ Image model saved.")
    return clf

# Diagnostics and main (keep as before)
def diagnose_images(image_root="dataset/images"):
    print("\n─── Image Folder Diagnostics ───")
    if not os.path.isdir(image_root):
        print(f"  ❌ {image_root} does not exist!")
        return
    for cls_dir in os.listdir(image_root):
        full = os.path.join(image_root, cls_dir)
        if not os.path.isdir(full): continue
        img_files = [f for f in os.listdir(full) if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        print(f"  {cls_dir}/  →  {len(img_files)} image files")

def train_all():
    diagnose_images()
    train_tabular()
    train_image_model()
    return "Done!"

if __name__ == "__main__":
    train_all()
    print("\n🎉 Finished! Run:  python app.py")