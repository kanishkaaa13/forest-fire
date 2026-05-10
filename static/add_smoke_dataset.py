"""
add_smoke_dataset.py - Adds real smoke + fire images from Kaggle
Run: python add_smoke_dataset.py
"""

import os
import shutil
import kagglehub
from pathlib import Path

print("Downloading Fire/Smoke Detection YOLO v9 dataset...")

# Download the dataset
path = kagglehub.dataset_download("roscoekerby/firesmoke-detection-yolo-v9")
print(f"Dataset downloaded to: {path}")

# Create target folders
fire_dest = Path("dataset/images/fire")
nofire_dest = Path("dataset/images/nofire")
fire_dest.mkdir(parents=True, exist_ok=True)
nofire_dest.mkdir(parents=True, exist_ok=True)

fire_count = 0
smoke_count = 0
other_count = 0

print("Copying images... (this may take a few minutes)")

for root, dirs, files in os.walk(path):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(root, file)
            rel_path = os.path.relpath(root, path).lower()
            
            # Logic based on dataset structure (fire or smoke)
            if 'fire' in rel_path or 'flame' in rel_path or file.lower().startswith('fire'):
                dest = fire_dest / file
                shutil.copy(src, dest)
                fire_count += 1
            elif 'smoke' in rel_path or 'smog' in rel_path:
                dest = fire_dest / f"smoke_{file}"   # treat smoke as fire class for our binary model
                shutil.copy(src, dest)
                smoke_count += 1
            else:
                # Unknown → put in nofire to be safe (you can move later)
                dest = nofire_dest / file
                shutil.copy(src, dest)
                other_count += 1

print(f"\n✅ Added:")
print(f"   Fire images     : {fire_count}")
print(f"   Smoke images    : {smoke_count}  (moved to fire/ folder)")
print(f"   Other images    : {other_count} (moved to nofire/)")
print(f"\nTotal fire folder now has many more real smoke examples!")
print("Next step: Run `python train_models.py` to retrain.")