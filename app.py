"""
app.py - Forest Fire Detection Flask App
Ensemble: CNN (primary, weight=4) + GradientBoosting (weight=2) + Heuristic (fallback, weight=1)
Map data: NASA FIRMS satellite active fire data (real world, last 24h)
"""

import os, io, base64, numpy as np, pickle, json, urllib.request
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__)

# ── Lazy-loaded model references (nothing loads at startup) ────────
_cnn_model     = None
_cnn_transform = None
_cnn_loaded    = False   # tracks whether load was attempted

_gb_bundle     = None
_gb_loaded     = False

_tab_scaler     = None
_tab_classifier = None
_tab_regressor  = None
_tab_loaded     = False


def load_cnn():
    """Load CNN model on first use, not at startup."""
    global _cnn_model, _cnn_transform, _cnn_loaded
    if _cnn_loaded:
        return
    _cnn_loaded = True
    try:
        import torch, timm
        import torchvision.transforms as transforms

        with open("models/cnn_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        model_name = meta.get("model_name", "mobilenetv3_small_075")
        img_size   = meta.get("img_size", 224)

        model = timm.create_model(model_name, pretrained=False, num_classes=2)
        model.load_state_dict(
            torch.load("models/cnn_fire_model.pth", map_location=torch.device("cpu"))
        )
        model.eval()

        _cnn_model = model
        _cnn_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        print(f"✅ CNN loaded  →  {model_name}  (img_size={img_size})")
    except FileNotFoundError:
        print("⚠ CNN model not found. Run: python train_cnn.py")
    except Exception as e:
        print(f"⚠ CNN model failed to load: {e}")


def load_gb():
    """Load GradientBoosting model on first use."""
    global _gb_bundle, _gb_loaded
    if _gb_loaded:
        return
    _gb_loaded = True
    try:
        with open("models/image_model.pkl", "rb") as f:
            _gb_bundle = pickle.load(f)
        print("✅ GradientBoosting model loaded")
    except FileNotFoundError:
        print("⚠ GB model not found. Run: python train_models.py")
    except Exception as e:
        print(f"⚠ GB model failed to load: {e}")


def load_tabular():
    """Load tabular models on first use."""
    global _tab_scaler, _tab_classifier, _tab_regressor, _tab_loaded
    if _tab_loaded:
        return
    _tab_loaded = True
    try:
        with open("models/scaler.pkl",     "rb") as f: _tab_scaler     = pickle.load(f)
        with open("models/classifier.pkl", "rb") as f: _tab_classifier = pickle.load(f)
        with open("models/regressor.pkl",  "rb") as f: _tab_regressor  = pickle.load(f)
        print("✅ Tabular models loaded")
    except FileNotFoundError:
        print("⚠ Tabular models not found. Run: python train_models.py")
    except Exception as e:
        print(f"⚠ Tabular models failed to load: {e}")


# ══════════════════════════════════════════════════════════════════
#  HEURISTIC  (fallback — only used when CNN is uncertain or absent)
#  Much stricter than before. Rejects red foliage, autumn scenes.
# ══════════════════════════════════════════════════════════════════

def smart_fire_heuristic(img_array):
    img = img_array.astype(np.float32) / 255.0
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

    avg_texture = float((r.std() + g.std() + b.std()) / 3.0)
    if avg_texture < 0.04:
        return False, 0.0, "Solid color — not fire"

    maxc = np.maximum(np.maximum(r,g),b)
    minc = np.minimum(np.minimum(r,g),b)
    v    = maxc
    s    = np.where(maxc > 0, (maxc-minc)/maxc, 0)
    diff = maxc - minc + 1e-6
    hue  = np.where(maxc==r,(g-b)/diff%6,
           np.where(maxc==g,(b-r)/diff+2,(r-g)/diff+4))/6.0

    # Strict fire: BRIGHT warm pixels only (not muted autumn red)
    fire_px    = (r>0.68) & (g<r*0.72) & (b<r*0.42) & (r-b>0.38) & (v>0.65)
    fire_ratio = float(fire_px.mean())

    # Reject solid object (car/wall)
    if fire_ratio > 0.0:
        fy, fx = np.where(fire_px)
        if len(fy) > 10:
            bbox      = (fy.max()-fy.min()+1)*(fx.max()-fx.min()+1)
            fill_rate = len(fy)/(bbox+1e-6)
            if fill_rate > 0.70 and avg_texture < 0.12:
                return False, 0.0, "Solid object (car/wall) — not fire"
        else:
            fire_ratio = 0.0

    # Reject broad red/orange foliage scenes (autumn, red trees)
    red_dom    = float((r > 0.40).mean())
    bright_var = float(v.std())
    if red_dom > 0.42 and bright_var < 0.20 and fire_ratio < 0.12:
        return False, 0.0, "Red foliage / natural scene — not fire"

    smoke_px    = (s<0.22) & (v>0.38) & (v<0.88) & (np.abs(r-g)<0.09) & (np.abs(g-b)<0.09)
    smoke_ratio = float(smoke_px.mean())

    green_px = float(((g>r*1.05)&(g>b)&(g>0.20)).mean())
    sky_px   = float(((b>0.52)&(b>r)&(b>g)).mean())
    h2, w2   = r.shape[0]//2, r.shape[1]//2
    quads    = [r[:h2,:w2].mean(),r[:h2,w2:].mean(),r[h2:,:w2].mean(),r[h2:,w2:].mean()]
    sp_var   = float(np.std(quads))

    fire_score  = min(fire_ratio/0.08, 1.0)
    smoke_score = min(smoke_ratio/0.15, 1.0)
    confidence  = float(np.clip(
        fire_score*0.55 + smoke_score*0.25
        + min(avg_texture/0.18,0.25) + min(sp_var/0.18,0.20)
        - green_px*0.40 - sky_px*0.25,
        0.0, 1.0
    ))
    is_fire = (fire_score > 0.50 and confidence > 0.35) or (smoke_score > 0.65 and fire_score > 0.15)
    reason  = (f"fire_px={fire_ratio*100:.1f}% smoke={smoke_ratio*100:.1f}% "
               f"tex={avg_texture:.3f} red_dom={red_dom*100:.0f}%")
    return is_fire, confidence, reason


# ── Tabular override ───────────────────────────────────────────────
def tabular_heuristic_override(features, model_pred, model_conf):
    f    = {k: float(v) for k,v in features.items()}
    T    = f.get("Temperature",20); RH   = f.get("RH",50)
    rain = f.get("Rain",0);         fwi  = f.get("FWI",0)
    ffmc = f.get("FFMC",60);        isi  = f.get("ISI",0)
    if rain >= 3.0:               return 0,  5.0, f"Heavy rain {rain}mm"
    if rain >= 1.0 and fwi < 5:  return 0, 10.0, f"Rain + low FWI"
    if T < 12:                    return 0,  8.0, f"Temp {T}°C too low"
    if ffmc < 40:                 return 0,  8.0, f"FFMC {ffmc} too low"
    if RH > 88 and rain > 0:      return 0,  7.0, f"High humidity + rain"
    if fwi>=30 and T>=32 and RH<=30 and rain==0: return 1,97.0,f"Extreme FWI={fwi}"
    if fwi>=20 and ffmc>=88 and isi>=10:         return 1,92.0,f"Critical indices"
    if ffmc>=92 and T>=35 and RH<=25 and rain==0:return 1,90.0,f"Extreme dryness"
    if model_pred==1 and fwi<5 and RH>70:        return model_pred,min(model_conf,45.0),"Mild — reduced"
    if model_pred==0 and fwi>=15 and T>=28 and rain==0: return model_pred,max(model_conf,60.0),"Elevated — boosted"
    return model_pred, model_conf, None


# ── Feature extractor (must match train_models.py) ─────────────────
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


def fwi_danger(fwi):
    if fwi<5.2:  return "Low"
    if fwi<11.2: return "Moderate"
    if fwi<21.3: return "High"
    if fwi<38.0: return "Very High"
    return "Extreme"


# ══════════════════════════════════════════════════════════════════
#  NASA FIRMS  — real-world active satellite fire data
#  No API key needed. Public CSV. Refreshed every hour.
# ══════════════════════════════════════════════════════════════════
FIRMS_MAP_KEY = "7a866dce41eddfabb75cb6d797b170fd"
FIRMS_TTL     = 1800   # 30 min cache

_firms_cache = {"data": [], "fetched_at": None}

def fetch_nasa_firms():
    import time as _t, random
    now = _t.time()
    if _firms_cache["fetched_at"] and (now - _firms_cache["fetched_at"]) < FIRMS_TTL:
        return _firms_cache["data"]

    fires = []
    try:
        url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
               f"{FIRMS_MAP_KEY}/VIIRS_NOAA20_NRT/world/1")

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read().decode("utf-8")

        lines = raw.strip().split("\n")
        if len(lines) < 2:
            raise ValueError("Empty response")

        header = [h.strip() for h in lines[0].split(",")]

        def col(name):
            return header.index(name) if name in header else None

        lat_i  = col("latitude");   lng_i  = col("longitude")
        conf_i = col("confidence"); acq_i  = col("acq_date")
        tim_i  = col("acq_time");   brt_i  = col("bright_ti4") or col("brightness")

        data_lines = lines[1:]
        if len(data_lines) > 1500:
            data_lines = random.sample(data_lines, 1500)

        for line in data_lines:
            c = line.split(",")
            try:
                lat  = float(c[lat_i])
                lng  = float(c[lng_i])
                conf = c[conf_i].strip() if conf_i else "n"
                date = c[acq_i].strip()  if acq_i  else ""
                time = c[tim_i].strip()  if tim_i  else ""
                brt  = float(c[brt_i])   if brt_i and c[brt_i].strip() else 0

                if   conf in ("h","high"):    conf_pct = 90
                elif conf in ("l","low"):     conf_pct = 50
                else:
                    try:    conf_pct = int(float(conf))
                    except: conf_pct = 70

                time_fmt = f"{time[:2]}:{time[2:]}" if len(time)==4 else time
                fires.append({
                    "lat": lat, "lng": lng,
                    "confidence": conf_pct,
                    "time": f"{date} {time_fmt}".strip(),
                    "source": "NASA FIRMS (VIIRS NOAA-20)",
                    "brightness": round(brt, 1),
                })
            except Exception:
                continue

        print(f"✅ NASA FIRMS API: {len(fires)} fires loaded")
        _firms_cache["data"]       = fires
        _firms_cache["fetched_at"] = now

    except Exception as e:
        print(f"⚠ NASA FIRMS API failed: {e}")
        if _firms_cache["data"]:
            return _firms_cache["data"]

    return fires


# ── In-memory local detection log ─────────────────────────────────
local_detections = []

# ── Routes ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/map")
def map_page():
    return render_template("map.html")


@app.route("/predict_tabular", methods=["POST"])
def predict_tabular():
    load_tabular()  # lazy load on first request
    try:
        if _tab_classifier is None:
            return jsonify({"success":False,"error":"Tabular models not loaded. Run: python train_models.py"})
        FEATURES = ["Temperature","RH","Ws","Rain","FFMC","DMC","DC","ISI","BUI","FWI"]
        data     = request.get_json()
        row      = np.array([[float(data[f]) for f in FEATURES]])
        row_sc   = _tab_scaler.transform(row)
        pred     = int(_tab_classifier.predict(row_sc)[0])
        proba    = _tab_classifier.predict_proba(row_sc)[0]
        conf     = round(float(proba[pred])*100, 1)
        final_pred, final_conf, override = tabular_heuristic_override(data, pred, conf)
        label    = "🔥 FIRE RISK DETECTED" if final_pred==1 else "✅ NO FIRE RISK"
        fwi_pred = float(_tab_regressor.predict(row_sc)[0]) if _tab_regressor else float(data["FWI"])
        return jsonify({"success":True,
            "classification":{"class":final_pred,"label":label,"confidence":round(final_conf,1),"override_reason":override},
            "regression":{"FWI":round(fwi_pred,2),"danger_level":fwi_danger(fwi_pred)}})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success":False,"error":str(e)})


@app.route("/predict_image", methods=["POST"])
def predict_image():
    load_cnn()  # lazy load on first request
    load_gb()   # lazy load on first request
    try:
        import torch

        if "image" not in request.files:
            return jsonify({"success":False,"error":"No image uploaded"})

        file    = request.files["image"]
        img_pil = Image.open(file.stream).convert("RGB")
        lat     = request.form.get("lat")
        lng     = request.form.get("lng")
        votes   = []

        cnn_fire_prob = None

        # ── CNN  (weight=4, PRIMARY) ───────────────────────────────
        if _cnn_model is not None and _cnn_transform is not None:
            inp  = _cnn_transform(img_pil).unsqueeze(0)
            with torch.no_grad():
                out  = _cnn_model(inp)
                prob = torch.softmax(out, dim=1)[0]
                cnn_fire_prob = float(prob[1])
                cnn_fire = cnn_fire_prob > 0.55
            votes.append((cnn_fire, cnn_fire_prob, "CNN", 4))
            print(f"[CNN] fire_prob={cnn_fire_prob:.3f} → {'FIRE' if cnn_fire else 'NO FIRE'}")

        # ── GradientBoosting  (weight=2) ──────────────────────────
        if _gb_bundle is not None:
            arr   = np.array(img_pil.resize((128,128)), dtype=np.uint8)
            feats = extract_features(arr).reshape(1,-1)
            fsc   = _gb_bundle["scaler"].transform(feats)
            gp    = _gb_bundle["model"].predict_proba(fsc)[0]
            gbp   = float(gp[1]); gbf = gbp > 0.45
            votes.append((gbf, gbp, "GB", 2))
            print(f"[GB]  fire_prob={gbp:.3f} → {'FIRE' if gbf else 'NO FIRE'}")

        # ── Heuristic (weight=1, ONLY when CNN uncertain or absent) ──
        cnn_confident = cnn_fire_prob is not None and (cnn_fire_prob > 0.65 or cnn_fire_prob < 0.35)
        if not cnn_confident:
            arr256 = np.array(img_pil.resize((256,256)), dtype=np.uint8)
            hf, hc, hr = smart_fire_heuristic(arr256)
            votes.append((hf, hc, "Heuristic", 1))
            print(f"[Heuristic] fire={hf} conf={hc:.3f} | {hr}")
        else:
            print(f"[Heuristic] Skipped — CNN confident ({cnn_fire_prob:.3f})")

        # ── Ensemble ───────────────────────────────────────────────
        total_w = sum(w for _,_,_,w in votes)
        fire_w  = sum(w for f,_,_,w in votes if f)
        final   = fire_w > (total_w - fire_w)

        # Hard CNN override
        if cnn_fire_prob is not None:
            if cnn_fire_prob > 0.88:   final = True;  print(f"[Override] CNN FIRE ({cnn_fire_prob:.3f})")
            elif cnn_fire_prob < 0.15: final = False; print(f"[Override] CNN NO FIRE ({cnn_fire_prob:.3f})")

        winning    = [c for f,c,_,_ in votes if f==final]
        confidence = round(float(np.mean(winning))*100,1) if winning else 0.0
        label      = "🔥 FIRE DETECTED" if final else "✅ NO FIRE DETECTED"
        print(f"[Ensemble] {fire_w}/{total_w} → {label} ({confidence}%)\n")

        if final and confidence > 65:
            local_detections.append({
                "lat":        float(lat) if lat else 18.52,
                "lng":        float(lng) if lng else 73.85,
                "confidence": confidence,
                "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source":     "Local detection",
                "brightness": 0,
            })
            if len(local_detections) > 50: local_detections.pop(0)

        buf = io.BytesIO()
        img_pil.resize((300,200)).save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            "success":True,"label":label,"class":int(final),
            "confidence":confidence,"thumbnail":f"data:image/jpeg;base64,{img_b64}",
            "lat":lat,"lng":lng,
            "votes":[{"source":s,"fire":f,"conf":round(c*100,1),"weight":w} for f,c,s,w in votes]
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success":False,"error":str(e)})


@app.route("/get_fire_events")
def get_fire_events():
    nasa     = fetch_nasa_firms()
    combined = nasa + local_detections
    return jsonify({"events":combined,"nasa_count":len(nasa),
                    "local_count":len(local_detections),"total":len(combined)})

@app.route("/get_local_detections")
def get_local_detections():
    return jsonify({"events": local_detections})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)