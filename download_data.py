"""
download_data.py  –  Downloads both datasets using the correct Kaggle IDs
Run this from your project root folder:
    python download_data.py
"""

import os, shutil, glob

os.makedirs("dataset", exist_ok=True)
os.makedirs("dataset/images/fire",   exist_ok=True)
os.makedirs("dataset/images/nofire", exist_ok=True)
os.makedirs("models",    exist_ok=True)
os.makedirs("static",    exist_ok=True)
os.makedirs("templates", exist_ok=True)

print("=" * 55)
print("  Forest Fire – Dataset Downloader")
print("=" * 55)

import kagglehub

# ── 1. Algerian Forest Fires CSV ──────────────────────────────────
print("\n[1/2] Downloading Algerian Forest Fires dataset (CSV)...")
try:
    path1 = kagglehub.dataset_download("nitinchoudhary012/algerian-forest-fires-dataset")
    print(f"  Kaggle cached at: {path1}")

    copied = 0
    for csv in glob.glob(os.path.join(path1, "**", "*.csv"), recursive=True):
        dest = os.path.join("dataset", os.path.basename(csv))
        shutil.copy(csv, dest)
        print(f"  Copied: {dest}")
        copied += 1

    if copied == 0:
        for f in os.listdir(path1):
            if f.endswith(".csv"):
                shutil.copy(os.path.join(path1, f), os.path.join("dataset", f))
                print(f"  Copied: dataset/{f}")
                copied += 1

    print(f"  CSV files copied: {copied}")
except Exception as e:
    print(f"  ERROR: {e}")
    print("  -> Make sure kaggle.json is in C:\\Users\\<YourName>\\.kaggle\\")

# ── 2. Forest Fire Images ─────────────────────────────────────────
print("\n[2/2] Downloading Forest Fire Image dataset...")
try:
    path2 = kagglehub.dataset_download("datascientist97/forest-fire")
    print(f"  Kaggle cached at: {path2}")

    fire_count   = 0
    nofire_count = 0

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for img in glob.glob(os.path.join(path2, "**", ext), recursive=True):
            parent = os.path.basename(os.path.dirname(img)).lower()
            fname  = os.path.basename(img)

            if "fire" in parent and "no" not in parent and "non" not in parent:
                dest = os.path.join("dataset", "images", "fire", fname)
                fire_count += 1
            else:
                dest = os.path.join("dataset", "images", "nofire", fname)
                nofire_count += 1

            shutil.copy(img, dest)

    print(f"  Fire images copied:    {fire_count}")
    print(f"  No-fire images copied: {nofire_count}")

    if fire_count == 0 and nofire_count == 0:
        print("\n  Could not auto-sort. Showing folder structure:")
        for root, dirs, files in os.walk(path2):
            level = root.replace(path2, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"  {indent}{os.path.basename(root)}/")
            if level < 3:
                subindent = ' ' * 2 * (level + 1)
                for f in files[:3]:
                    print(f"  {subindent}{f}")

except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 55)
print("  DONE! Now run:  python train_models.py")
print("=" * 55)

csv_files  = glob.glob("dataset/*.csv")
img_fire   = glob.glob("dataset/images/fire/*")
img_nofire = glob.glob("dataset/images/nofire/*")
print(f"\n  Summary:")
print(f"  CSV files:      {len(csv_files)} — {[os.path.basename(f) for f in csv_files]}")
print(f"  Fire images:    {len(img_fire)}")
print(f"  No-fire images: {len(img_nofire)}")