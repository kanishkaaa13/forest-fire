"""
add_smoke_dataset.py - Adds real smoke + fire images from Kaggle
Run this file with: python add_smoke_dataset.py
"""

import os
import shutil
import kagglehub
from pathlib import Path

print("=" * 60)
print("Downloading Fire & Smoke Detection Dataset from Kaggle...")
print("=" * 60)

# Download the dataset
path = kagglehub.dataset_download("roscoekerby/firesmoke-detection-yolo-v9")
print(f"Dataset downloaded to: {path}")

# Target folders
fire_dest = Path("dataset/images/fire")
nofire_dest = Path("dataset/images/nofire")
fire_dest.mkdir(parents=True, exist_ok=True)
nofire_dest.mkdir(parents=True, exist_ok=True)

fire_count = 0
smoke_count = 0
other_count = 0

print("\nCopying images... This may take 2-5 minutes depending on your internet and PC speed.")

for root, dirs, files in os.walk(path):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(root, path).lower()
            
            # Classify based on folder name or filename
            if any(keyword in rel_path for keyword in ['fire', 'flame', 'burn']):
                dest = fire_dest / file
                shutil.copy2(src_path, dest)
                fire_count += 1
            elif any(keyword in rel_path for keyword in ['smoke', 'smog', 'haze']):
                # Rename slightly to avoid overwriting
                new_name = f"smoke_{file}"
                dest = fire_dest / new_name
                shutil.copy2(src_path, dest)
                smoke_count += 1
            else:
                # Put unknown images in nofire (you can review later)
                dest = nofire_dest / file
                shutil.copy2(src_path, dest)
                other_count += 1

print("\n" + "=" * 60)
print("✅ Dataset Integration Complete!")
print("=" * 60)
print(f"   Fire images added      : {fire_count}")
print(f"   Smoke images added     : {smoke_count}  (treated as Fire class)")
print(f"   Other images added     : {other_count}  (went to nofire/)")
print(f"\nYour fire/ folder now has many real smoke examples.")
print("Next steps:")
print("   1. Delete old model:   del models\\image_model.pkl")
print("   2. Retrain:            python train_models.py")
print("   3. Run app:            python app.py")