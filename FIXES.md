# FIXES.md — Forest Fire Detection: False Positive Remediation

## Root Causes Fixed

| # | Problem | Root Cause | Fix |
|---|---------|-----------|-----|
| 1 | CNN never used | Weights trained with `timm` but loaded into `torchvision` → missing keys → model silently disabled | Retrained with `torchvision` MobileNetV3-Small (exact same architecture as `app.py` loads) |
| 2 | GB over-triggers | Threshold `gbp > 0.45` too low — warm skin tones / cream paper / banana leaves all score ≥45% | Raised to `gbp > 0.62` |
| 3 | Heuristic not protecting | Only ran when CNN uncertain; with CNN disabled it couldn't veto GB | Now **always** runs as a **hard veto** |
| 4 | No minimum confidence | Low-confidence fire votes still shown as FIRE DETECTED | Added 50% confidence floor before declaring fire |
| 5 | Heuristic too broad | Old fire pixel mask `(r>0.48)&(g<r*0.92)&(b<r*0.62)` matched skin tones, paper, leaves | Tightened to `(r>0.72)&(g<r*0.68)&(b<r*0.38)&(r-b>0.42)&(v>0.60)` |

---

## Changes Made

### `train_cnn.py` — Full Rewrite

- **Removed** `import timm` (was the root cause of the architecture mismatch)
- **Uses** `torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)`
- **Replaces** `model.classifier[3]` with `nn.Linear(in_features, 2)` — exactly matching `app.py`
- **Augmentation**: random horizontal flip, rotation ±15°, color jitter (brightness=0.3, contrast=0.3, saturation=0.2)
- **Optimizer**: Adam, lr=1e-4, 15 epochs, batch 32
- **Class weighting**: automatically computed from dataset counts (handles imbalance)
- **Post-train verification**: reloads saved weights with `strict=True` to confirm app.py compatibility
- **Saves** `models/cnn_meta.pkl` with `{"model": "mobilenet_v3_small", "input_size": 224, "classes": ["nofire", "fire"]}`

### `app.py` — Targeted Patches

#### FIX 2a — GB threshold raised
```python
# Before
gbf = gbp > 0.45
# After
gbf = gbp > 0.62
```

#### FIX 2b — Heuristic always runs + hard veto
```python
# Before: heuristic only ran when CNN was uncertain
# After: always runs, and if heuristic says NO FIRE with conf < 0.12 → force final = False
if not hf and hc < 0.12:
    final = False  # "Heuristic VETO"
```

#### FIX 2c — Confidence floor
```python
if final and confidence < 50.0:
    final = False
```

#### FIX 2d — CNN soft vote
```python
# If CNN fire prob is between 0.15 and 0.40 (leans no-fire), reduce confidence by 15pp
if cnn_fire_prob is not None and 0.15 <= cnn_fire_prob <= 0.40:
    confidence = max(0.0, confidence - 15.0)
```

#### FIX 2e — Heuristic-only fallback warning
```python
# If neither CNN nor GB loaded, add warning in JSON response and cap confidence at 60%
resp["warning"] = "Running on heuristic only — accuracy may be low. Please check model files."
```

#### FIX 3 — Stricter heuristic

| Guard | Trigger | Effect |
|-------|---------|--------|
| **Tighter fire pixels** (3d) | `r>0.72, g<r*0.68, b<r*0.38, r-b>0.42, v>0.60` | Only true flame colors pass |
| **Skin tone** (3a) | `>12%` of pixels in HSV skin range AND fire_ratio < 10% | Portrait / people photos vetoed |
| **Document/text** (3b) | `>35%` near-white AND `>15%` near-black | Printed invitations, cards vetoed |
| **Green vegetation** (3c) | green_px > 35% AND fire_ratio < 8% | Forest / garden photos vetoed |

### `test_false_positives.py` — New File

Standalone validation script that:
- Loads the same models as `app.py` (CNN, GB, heuristic)
- Runs each image through the full ensemble with all FIX 2 logic
- Prints a colour-coded table: `Filename | CNN% | GB% | Heuristic | Final | Conf%`
- **Flags false positives in red**, false negatives in yellow (based on filename)
- Prints a summary: total, FP count, FN count, per-FP model breakdown

---

## Re-Deploy Checklist

### Step 1 — Retrain CNN (required once)
```bash
# On your local machine (CPU is fine, ~10–20 min for typical dataset sizes)
python train_cnn.py
```
Expected output:
```
✅ Weights verified — strict load successful, app.py will load this model correctly.
✅ Model weights   → models/cnn_fire_model.pth
```

### Step 2 — Validate (optional but recommended)
```bash
# Put test images in test_images/ folder
# Name them nofire_*.jpg and fire_*.jpg for automatic FP/FN counting
mkdir test_images
# copy some problematic images in...
python test_false_positives.py
```

### Step 3 — Commit & Push
```bash
git add train_cnn.py app.py test_false_positives.py FIXES.md
git add models/cnn_fire_model.pth models/cnn_meta.pkl
git commit -m "fix: retrain CNN (torchvision), tighten GB threshold + heuristic guards"
git push
```

### Step 4 — Render Auto-Deploy
Render picks up the push automatically. Check the deploy logs for:
```
✅ CNN loaded  →  mobilenet_v3_small (torchvision)
✅ GradientBoosting model loaded
```

> **Note**: If Render's free instance has limited disk space, the `.pth` file (~4 MB) is small enough to commit directly.

---

## Expected Behaviour After Fixes

| Image type | Before | After |
|------------|--------|-------|
| Portrait of two people | 97% FIRE 🔥 | ✅ NO FIRE (skin tone veto) |
| Marathi invitation card | 68% FIRE 🔥 | ✅ NO FIRE (document veto) |
| Banana leaf decoration | FIRE 🔥 | ✅ NO FIRE (green vegetation veto) |
| Actual forest fire | FIRE 🔥 | 🔥 FIRE (CNN primary signal) |
| Smoke-only image | uncertain | 🔥 FIRE (smoke+heuristic) |
