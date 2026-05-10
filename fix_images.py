"""
fix_images.py  -  Sort images by FILENAME prefix
AoF = Aerial of Fire     --> fire/
WeB = Without Fire/Other --> nofire/

Run from your project root:
    python fix_images.py
"""

import os, shutil, glob

NOFIRE_SRC  = os.path.join("dataset", "images", "nofire")
FIRE_DEST   = os.path.join("dataset", "images", "fire")

os.makedirs(FIRE_DEST, exist_ok=True)

all_images = glob.glob(os.path.join(NOFIRE_SRC, "*"))
print(f"Total images found in nofire/: {len(all_images)}")

# Show first 10 filenames so we can confirm prefixes
print("\nSample filenames:")
for f in sorted(all_images)[:10]:
    print(f"  {os.path.basename(f)}")

# Count by prefix first
from collections import Counter
prefixes = Counter()
for f in all_images:
    name = os.path.basename(f)
    prefixes[name[:3].lower()] += 1

print("\nPrefix counts (first 3 letters of filename):")
for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1]):
    print(f"  '{prefix}' : {count} images")

# ── Sort: AoF --> fire,  everything else --> stays in nofire ──────
fire_count   = 0
stayed_count = 0

for fpath in all_images:
    fname  = os.path.basename(fpath)
    prefix = fname[:3].lower()

    # AoF = Aerial of Fire
    if prefix == "aof":
        dest = os.path.join(FIRE_DEST, fname)
        shutil.move(fpath, dest)
        fire_count += 1
    else:
        stayed_count += 1   # already in nofire, leave it

print(f"\nMoved to fire/:    {fire_count}")
print(f"Kept in nofire/:   {stayed_count}")
print(f"\nDone! Now run:  python train_models.py")