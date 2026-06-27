"""
test_false_positives.py  –  Validate ensemble against a folder of test images.

Usage:
    python test_false_positives.py
    python test_false_positives.py --folder test_images
    python test_false_positives.py --folder test_images --glob "*.jpg"

Naming convention for automatic FP/FN counting:
    *fire*   in filename → expected FIRE  (e.g. fire_001.jpg)
    *nofire* in filename → expected NO FIRE (e.g. nofire_001.jpg, nofire_scene.png)

Output:
    A table: filename | CNN_prob | GB_prob | Heuristic | Final | Confidence
    Summary: total, false positives, false negatives
"""

import os, sys, glob, argparse, pickle, pathlib
import numpy as np
from PIL import Image

# ── Colour codes ───────────────────────────────────────────────────
RED   = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"
BOLD  = "\033[1m"
YELLOW = "\033[93m"

# ── Args ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Test ensemble for false positives/negatives")
parser.add_argument("--folder", default="test_images",
                    help="Folder containing test images (default: test_images/)")
parser.add_argument("--glob",   default="*.*",
                    help="Glob pattern inside folder (default: *.*)")
args = parser.parse_args()

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ══════════════════════════════════════════════════════════════════
#  Load models (mirrors app.py load_cnn / load_gb exactly)
# ══════════════════════════════════════════════════════════════════

_cnn_model     = None
_cnn_transform = None
_gb_bundle     = None


def load_cnn():
    global _cnn_model, _cnn_transform
    try:
        import torch
        from torchvision import transforms
        from torchvision.models import mobilenet_v3_small

        device = torch.device("cpu")
        model  = mobilenet_v3_small(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)

        state = torch.load("models/cnn_fire_model.pth", map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"⚠ CNN: {len(missing)} missing keys — CNN disabled")
            return

        model.eval()
        _cnn_model = model
        _cnn_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        print("✅ CNN loaded (torchvision mobilenet_v3_small)")
    except FileNotFoundError:
        print("⚠ CNN model file not found (models/cnn_fire_model.pth)")
    except ImportError:
        print("⚠ torch/torchvision not installed — CNN disabled")
    except Exception as e:
        print(f"⚠ CNN load failed: {e}")


def load_gb():
    global _gb_bundle
    try:
        with open("models/image_model.pkl", "rb") as f:
            _gb_bundle = pickle.load(f)
        print("✅ GradientBoosting model loaded")
    except FileNotFoundError:
        print("⚠ GB model not found (models/image_model.pkl)")
    except Exception as e:
        print(f"⚠ GB load failed: {e}")


# ══════════════════════════════════════════════════════════════════
#  Heuristic & feature extractor (kept in sync with app.py)
# ══════════════════════════════════════════════════════════════════

def smart_fire_heuristic(img_array):
    img = img_array.astype(np.float32) / 255.0
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

    avg_texture = float((r.std() + g.std() + b.std()) / 3.0)
    if avg_texture < 0.04:
        return False, 0.0, "Solid color"

    maxc = np.maximum(np.maximum(r,g),b)
    minc = np.minimum(np.minimum(r,g),b)
    v    = maxc
    s    = np.where(maxc > 0, (maxc-minc)/maxc, 0)
    diff = maxc - minc + 1e-6
    hue  = np.where(maxc==r,(g-b)/diff%6,
           np.where(maxc==g,(b-r)/diff+2,(r-g)/diff+4))/6.0

    # Tightened fire pixel mask (FIX 3d)
    fire_px    = (r>0.72) & (g<r*0.68) & (b<r*0.38) & (r-b>0.42) & (v>0.60)
    fire_ratio = float(fire_px.mean())

    if fire_ratio > 0.0:
        fy, fx = np.where(fire_px)
        if len(fy) > 10:
            bbox      = (fy.max()-fy.min()+1)*(fx.max()-fx.min()+1)
            fill_rate = len(fy)/(bbox+1e-6)
            if fill_rate > 0.70 and avg_texture < 0.12:
                return False, 0.0, "Solid object"
        else:
            fire_ratio = 0.0

    # Skin tone guard (FIX 3a)
    skin_px    = (hue < 0.069) & (s > 0.20) & (s < 0.60) & (v > 0.40) & (v < 0.90)
    skin_ratio = float(skin_px.mean())
    if skin_ratio > 0.12 and fire_ratio < 0.10:
        return False, 0.0, "Skin tone"

    # Document guard (FIX 3b)
    near_white = float((v > 0.88).mean())
    near_black = float((v < 0.12).mean())
    if near_white > 0.35 and near_black > 0.15:
        return False, 0.0, "Document/text"

    # Green vegetation guard (FIX 3c)
    green_px = float(((g>r*1.05)&(g>b)&(g>0.20)).mean())
    if green_px > 0.35 and fire_ratio < 0.08:
        return False, 0.0, "Green vegetation"

    red_dom    = float((r > 0.40).mean())
    bright_var = float(v.std())
    if red_dom > 0.42 and bright_var < 0.20 and fire_ratio < 0.12:
        return False, 0.0, "Red foliage"

    smoke_px    = (s<0.22) & (v>0.38) & (v<0.88) & (np.abs(r-g)<0.09) & (np.abs(g-b)<0.09)
    smoke_ratio = float(smoke_px.mean())
    sky_px      = float(((b>0.52)&(b>r)&(b>g)).mean())
    h2, w2      = r.shape[0]//2, r.shape[1]//2
    quads       = [r[:h2,:w2].mean(),r[:h2,w2:].mean(),r[h2:,:w2].mean(),r[h2:,w2:].mean()]
    sp_var      = float(np.std(quads))

    fire_score  = min(fire_ratio/0.08, 1.0)
    smoke_score = min(smoke_ratio/0.15, 1.0)
    confidence  = float(np.clip(
        fire_score*0.55 + smoke_score*0.25
        + min(avg_texture/0.18,0.25) + min(sp_var/0.18,0.20)
        - green_px*0.40 - sky_px*0.25,
        0.0, 1.0
    ))
    is_fire = (fire_score > 0.50 and confidence > 0.35) or (smoke_score > 0.65 and fire_score > 0.15)
    return is_fire, confidence, f"fire={fire_ratio*100:.1f}% skin={skin_ratio*100:.0f}%"


def extract_features(img_array):
    img = img_array.astype(np.float32)/255.0
    r,g,b = img[:,:,0],img[:,:,1],img[:,:,2]
    feats = []
    for ch in (r,g,b):
        h,_ = np.histogram(ch, bins=16, range=(0,1))
        feats.extend(h/(h.sum()+1e-6))
    maxc=np.maximum(np.maximum(r,g),b); minc=np.minimum(np.minimum(r,g),b)
    v=maxc; s=np.where(maxc>0,(maxc-minc)/maxc,0); diff=maxc-minc+1e-6
    hue=np.where(maxc==r,(g-b)/diff%6,np.where(maxc==g,(b-r)/diff+2,(r-g)/diff+4))/6.0
    for ch in (hue,s,v):
        feats+=[float(ch.mean()),float(ch.std())]
        h,_=np.histogram(ch,bins=8,range=(0,1)); feats.extend(h/(h.sum()+1e-6))
    fire=(r>0.48)&(g<r*0.92)&(b<r*0.62)
    smoke=(s<0.45)&(v>0.25)&(v<0.95)&(np.abs(r-g)<0.18)&(np.abs(g-b)<0.18)&(r.mean()>0.25)
    bright=(r>0.80)&(g>0.35)&(b<0.30); cool=(r<0.35)&(g>r)&(b<0.4); rd=r-0.5*(g+b)
    feats+=[float(fire.mean()),float(smoke.mean()),float(bright.mean()),float(cool.mean()),
            float(rd.mean()),float(np.percentile(rd,75)),float(np.percentile(rd,90)),
            float(r.var()),float(g.var()),float(b.var())]
    return np.array(feats, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════
#  Full ensemble (mirrors predict_image logic in app.py)
# ══════════════════════════════════════════════════════════════════

def run_ensemble(img_pil):
    votes         = []
    cnn_fire_prob = None

    # CNN
    if _cnn_model is not None:
        try:
            import torch
            inp  = _cnn_transform(img_pil).unsqueeze(0)
            with torch.no_grad():
                out  = _cnn_model(inp)
                prob = torch.softmax(out, dim=1)[0]
                cnn_fire_prob = float(prob[1])
                cnn_fire = cnn_fire_prob > 0.55
            votes.append((cnn_fire, cnn_fire_prob, "CNN", 4))
        except Exception as e:
            print(f"  [CNN] error: {e}")

    # GB
    gbp = None
    if _gb_bundle is not None:
        arr   = np.array(img_pil.resize((128,128)), dtype=np.uint8)
        feats = extract_features(arr).reshape(1,-1)
        fsc   = _gb_bundle["scaler"].transform(feats)
        gp    = _gb_bundle["model"].predict_proba(fsc)[0]
        gbp   = float(gp[1]); gbf = gbp > 0.62   # FIX 2a threshold
        votes.append((gbf, gbp, "GB", 2))

    # Heuristic — always run (FIX 2b)
    arr256     = np.array(img_pil.resize((256,256)), dtype=np.uint8)
    hf, hc, hr = smart_fire_heuristic(arr256)
    votes.append((hf, hc, "Heuristic", 1))

    total_w = sum(w for _,_,_,w in votes)
    fire_w  = sum(w for f,_,_,w in votes if f)
    final   = fire_w > (total_w - fire_w)

    # Heuristic hard veto (FIX 2b)
    if not hf and hc < 0.12:
        final = False

    # CNN hard overrides
    if cnn_fire_prob is not None:
        if cnn_fire_prob > 0.88:  final = True
        elif cnn_fire_prob < 0.15: final = False

    winning    = [c for f,c,_,_ in votes if f==final]
    confidence = round(float(np.mean(winning))*100, 1) if winning else 0.0

    # CNN soft vote (FIX 2d)
    if cnn_fire_prob is not None and 0.15 <= cnn_fire_prob <= 0.40:
        confidence = max(0.0, confidence - 15.0)

    # Confidence floor (FIX 2c)
    if final and confidence < 50.0:
        final = False

    return {
        "cnn_prob":  round(cnn_fire_prob * 100, 1) if cnn_fire_prob is not None else None,
        "gb_prob":   round(gbp * 100, 1) if gbp is not None else None,
        "heuristic": f"{'FIRE' if hf else 'no'}  {round(hc*100,1)}%",
        "heuristic_reason": hr,
        "final":     final,
        "confidence": confidence,
    }


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*70)
    print(f"  Forest Fire Ensemble — False Positive / Negative Test Runner")
    print("═"*70 + "\n")

    # Discover images
    folder = args.folder
    if not os.path.isdir(folder):
        print(f"❌ Folder not found: '{folder}'")
        print(f"   Create it and add test images, or use --folder <path>")
        sys.exit(1)

    pattern = os.path.join(folder, args.glob)
    paths   = sorted([
        p for p in glob.glob(pattern, recursive=True)
        if pathlib.Path(p).suffix.lower() in EXTS
    ])

    if not paths:
        print(f"❌ No images found in '{folder}' matching '{args.glob}'")
        sys.exit(1)

    print(f"📂 Found {len(paths)} images in '{folder}'\n")

    # Load models
    load_cnn()
    load_gb()
    print()

    # Column widths
    W_NAME = max(30, max(len(os.path.basename(p)) for p in paths) + 2)
    W_CNNp = 9
    W_GBp  = 9
    W_HEUR = 16
    W_FIN  = 10
    W_CONF = 10

    # Header
    header = (f"{'Filename':<{W_NAME}} {'CNN%':>{W_CNNp}} {'GB%':>{W_GBp}}"
              f" {'Heuristic':<{W_HEUR}} {'Final':<{W_FIN}} {'Conf%':>{W_CONF}}")
    sep    = "─" * len(header)
    print(BOLD + header + RESET)
    print(sep)

    results = []
    for path in paths:
        fname = os.path.basename(path)
        try:
            img_pil = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  ⚠ Could not open {fname}: {e}")
            continue

        r = run_ensemble(img_pil)

        cnn_s  = f"{r['cnn_prob']:>7.1f}%" if r["cnn_prob"] is not None else "   N/A  "
        gb_s   = f"{r['gb_prob']:>7.1f}%"  if r["gb_prob"]  is not None else "   N/A  "
        heur_s = f"{r['heuristic']:<{W_HEUR}}"
        fin_s  = "FIRE     " if r["final"] else "no fire  "
        conf_s = f"{r['confidence']:>8.1f}%"

        # Name of expected class from filename
        fname_low = fname.lower()
        expected_fire   = "fire" in fname_low and "nofire" not in fname_low
        expected_nofire = "nofire" in fname_low or "no_fire" in fname_low or "nofire" in fname_low

        # Colour coding
        is_fp = r["final"]  and expected_nofire   # false positive
        is_fn = not r["final"] and expected_fire  # false negative
        row = f"{fname:<{W_NAME}} {cnn_s:>{W_CNNp}} {gb_s:>{W_GBp}} {heur_s} {fin_s} {conf_s}"

        if is_fp:
            print(RED + row + f"  ← FALSE POSITIVE" + RESET)
        elif is_fn:
            print(YELLOW + row + f"  ← FALSE NEGATIVE" + RESET)
        elif r["final"] and r["confidence"] > 50:
            print(GREEN + row + RESET)
        else:
            print(row)

        results.append({
            "fname": fname,
            "result": r,
            "expected_fire": expected_fire,
            "expected_nofire": expected_nofire,
            "is_fp": is_fp,
            "is_fn": is_fn,
        })

    # ── Summary ───────────────────────────────────────────────────
    print(sep)
    total      = len(results)
    fire_pred  = sum(1 for x in results if x["result"]["final"])
    fp_count   = sum(1 for x in results if x["is_fp"])
    fn_count   = sum(1 for x in results if x["is_fn"])
    labeled    = sum(1 for x in results if x["expected_fire"] or x["expected_nofire"])

    print(f"\n{'─'*40}")
    print(f"  Total images tested : {total}")
    print(f"  Predicted FIRE      : {fire_pred}")
    print(f"  Predicted NO FIRE   : {total - fire_pred}")
    if labeled > 0:
        print(f"\n  Labeled images      : {labeled}")
        print(RED    + f"  False Positives     : {fp_count}" + RESET
              + f"  (nofire image predicted FIRE)")
        print(YELLOW + f"  False Negatives     : {fn_count}" + RESET
              + f"  (fire image predicted NO FIRE)")

        # Per-model breakdown for FP images
        fp_results = [x["result"] for x in results if x["is_fp"]]
        if fp_results:
            print(f"\n  {RED}False Positive breakdown:{RESET}")
            for x in results:
                if x["is_fp"]:
                    r = x["result"]
                    cnn = f"CNN={r['cnn_prob']}%" if r["cnn_prob"] is not None else "CNN=N/A"
                    gb  = f"GB={r['gb_prob']}%"  if r["gb_prob"]  is not None else "GB=N/A"
                    print(f"    {RED}{x['fname']}{RESET}")
                    print(f"      {cnn}  {gb}  Heuristic={r['heuristic']}")
                    print(f"      Heuristic reason: {r['heuristic_reason']}")
    else:
        print(f"\n  ℹ No labelled images found.")
        print(f"    Name files with 'fire' or 'nofire' to enable FP/FN counting.")

    print(f"{'─'*40}\n")


if __name__ == "__main__":
    main()
