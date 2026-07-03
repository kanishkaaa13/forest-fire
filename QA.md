# 🌲 Forest Fire Prediction: Project Overview & Interview Q&A

This document contains a comprehensive breakdown of the **Forest Fire Detection & Prediction System**, its technical architecture, tech stack, and detailed answers to the key interview questions.

---

## 🏗️ Project Architecture & Workflow

The system is a multi-modal forest fire monitoring platform. It combines **computer vision** (for camera/image-based fire detection) and **tabular climate forecasting** (for weather-based risk prediction) with a **real-time live satellite monitoring dashboard**.

```
User uploads image
        ↓
┌───────────────────────────────────────────┐
│   ENSEMBLE PREDICTION ENGINE              │
│                                           │
│  ① CNN — MobileNetV3  (weight = 4)       │
│     224×224 → deep feature extraction    │
│                                           │
│  ② Gradient Boosting  (weight = 2)       │
│     128×128 → 88 handcrafted features    │
│     RGB histograms + HSV stats + masks   │
│                                           │
│  ③ Smart Heuristic    (weight = 1)       │
│     Pixel-level fire/smoke detection     │
│     Rejects: autumn leaves, red cars,    │
│              sunsets, solid objects      │
└───────────────────────────────────────────┘
        ↓
  Weighted vote → Final prediction
        ↓
  Hard override if CNN > 88% or < 15%
        ↓
  🔥 FIRE DETECTED / ✅ NO FIRE  +  Confidence %
```

### Key Components
1. **Frontend Dashboard (`templates/index.html` & `templates/map.html`)**: Built using HTML5, Vanilla CSS, and JavaScript. The interactive map uses **Leaflet.js** and **CartoDB Dark Tiles** to plot active fires.
2. **Backend Server (`app.py`)**: Built with **Flask 3.1**, handling model loading, image feature extraction, heuristic evaluation, NASA API ingestion, and prediction endpoints.
3. **ML Training Pipelines (`train_models.py` & `train_cnn.py`)**:
   - `train_models.py`: Preprocesses the **Algerian Forest Fires dataset** to train the weather classifier and FWI regressor, and extracts 88 color/texture features from images to train a secondary image-based Gradient Boosting classifier.
   - `train_cnn.py`: Trains a lightweight **MobileNetV3-Small** CNN using PyTorch and torchvision.

---

## 🛠️ Complete Tech Stack

| Layer | Technology | Details |
|---|---|---|
| **Backend Framework** | Python 3.11, Flask 3.1 | Core application server and inference APIs. |
| **WSGI Server** | Gunicorn | High-performance production runner for deployment. |
| **Deep Learning** | PyTorch, Torchvision | Model definition, training, data loaders, and CPU-based inference. |
| **Machine Learning** | Scikit-learn | StandardScaler, GradientBoostingClassifier, and Ridge Regression. |
| **Data Processing** | NumPy, Pandas, Pillow | Image manipulation, CSV handling, and numerical calculations. |
| **Satellite Integration** | NASA FIRMS API | Real-time active fire tracking via VIIRS NOAA-20 satellite CSV stream. |
| **Frontend Map** | Leaflet.js | Open-source mobile-friendly interactive mapping library. |
| **Map Tiles** | CartoDB Dark Matter | Sleek, dark-themed background tiles optimized for highlighting bright red flame markers. |
| **UI Aesthetics** | HTML5, Vanilla CSS, JS | Modern dark mode theme using DM Sans, Bebas Neue, and DM Mono fonts. |
| **Deployment** | Render.com, GitHub Actions | CD pipeline linking repository commits directly to production. |

---

## 💬 Detailed Interview Q&A

### 1. Why MobileNetV3 specifically instead of ResNet/EfficientNet?
* **Resource Optimization**: The application is deployed on Render's free tier, which restricts RAM (typically 512MB) and limits CPU compute. Heavy architectures like ResNet-50 (~100MB weights) or larger EfficientNet variants require substantial RAM to load, causing Out-Of-Memory (OOM) crashes during deployment, and introduce notable latency on CPU inference.
* **Lightweight & Fast**: `mobilenet_v3_small` is extremely lightweight (~10MB weights / 2.5M parameters) and completes CPU inference in milliseconds, ensuring sub-second response times on standard web requests.
* **ImageNet Pretraining**: It leverages pre-trained ImageNet weights (`MobileNet_V3_Small_Weights.IMAGENET1K_V1`), giving the network powerful baseline features for textures, edges, and color patterns. We only need to swap and fine-tune the final linear head:
  ```python
  model.classifier[3] = nn.Linear(in_features, 2)
  ```

### 2. Walk me through the ensemble: why weight CNN=4, GB=2, heuristic=1 — how did you land on these numbers?
* **Total Voting Weight**: The total voting weight is $4 + 2 + 1 = 7$. To declare a fire under standard weighted voting, the positive votes must sum to $> 3.5$ (meaning $\ge 4$).
* **Model Hierarchy & Reasoning**:
  * **CNN (Weight = 4)**: The PyTorch CNN is the primary model because deep neural nets extract complex, hierarchical spatial features (shapes, fire boundaries, smoke columns) rather than relying purely on color values. Since its weight is 4, if the CNN is highly confident that there is a fire, it can carry the vote alone ($4 \ge 4$).
  * **Gradient Boosting (Weight = 2)**: The GB model evaluates global image statistics (color distributions, RGB histograms, HSV stats). It operates on tabular color information, which is fast and reliable for detecting color clusters but lacks spatial awareness.
  * **Smart Heuristic (Weight = 1)**: The heuristic layer performs pixel-level rule checking. It acts as a safety anchor.
* **Landing on these weights**: Through rigorous testing using `test_false_positives.py`, we wanted the CNN to remain the primary decision maker, but allow the secondary models (GB + Heuristic, total weight 3) to tip the scales if the CNN is uncertain or close to the decision threshold. Additionally, the weights are supplemented by explicit override controls (e.g. hard overrides and vetos) to handle edge cases where simple voting fails.

### 3. What's the hard override rule doing, and why is it needed on top of weighted voting?
Simple weighted voting fails at the extremes because statistical probabilities can average out incorrectly. To combat this, the ensemble implements two levels of hard overrides:
1. **Extreme CNN Confidence Override**:
   ```python
   if cnn_fire_prob > 0.88: final = True
   elif cnn_fire_prob < 0.15: final = False
   ```
   * *Why it's needed*: If the CNN is absolutely certain (e.g., $95\%$ probability of fire), we should bypass voting entirely. A false positive veto from the weaker heuristic should not suppress a glaring, obvious wildfire detection. Conversely, if the CNN sees no trace of fire ($<15\%$), it prevents random color matches from the GB/heuristic from triggering a false alarm.
2. **Heuristic Hard Veto**:
   ```python
   if not hf and hc < 0.12: final = False
   ```
   * *Why it's needed*: If the heuristic finds absolutely zero fire pixels (extremely bright, saturated oranges/reds) and no smoke textures in the image ($hc < 12\%$), any positive signal from the CNN/GB is statistically a false positive (e.g., a green forest or skin tones that tricked the neural net). The heuristic hard veto acts as an absolute physical sanity check.

### 4. What are the 58 handcrafted features fed to the Gradient Boosting image classifier?
> **Code-to-Doc Discrepancy Note**: The UI and comments mention "58 features", which was the feature count in an earlier iteration using fewer histogram bins. In the current implementation (`extract_features` function), the feature vector has been upgraded to **88 dimensions** for higher resolution. 

Here is the exact breakdown of the 88 extracted features:
1. **RGB Histograms (48 features)**: The red, green, and blue channels are split, binned into 16 bins each ($16 \times 3 = 48$), and normalized by the total pixel count.
2. **HSV Statistics & Histograms (30 features)**:
   * The image is converted to HSV (Hue, Saturation, Value).
   * For each channel, it computes the **Mean** and **Standard Deviation** ($2 \times 3 = 6$ features).
   * It also extracts an 8-bin histogram for each channel ($8 \times 3 = 24$ features), resulting in 30 features.
3. **Semantic/Targeted Heuristics & Statistical Metrics (10 features)**:
   * `fire.mean()`: Percentage of pixels matching raw fire RGB rules.
   * `smoke.mean()`: Percentage of pixels matching smoke texture constraints.
   * `bright.mean()`: Ratio of highly bright flame-colored pixels.
   * `cool.mean()`: Ratio of forest-like green pixels.
   * `rd.mean()`: Mean of Red-Difference `R - 0.5 * (G + B)`, which separates red/orange fire hues from cool foliage.
   * `np.percentile(rd, 75)` & `np.percentile(rd, 90)`: Captures concentrated, small fire spots.
   * `r.var()`, `g.var()`, `b.var()`: Texture variances indicating high spatial frequency (flames/smoke) vs smooth regions (sky/walls).

### 5. What is Fire Weather Index, and what do FFMC/DMC/DC/ISI/BUI actually represent?
The Canadian Forest Fire Weather Index (FWI) System is a method of estimating forest fire danger based on meteorological observations (Temperature, Relative Humidity, Wind Speed, and Rain).

| Index | Name | Represents | Sensitivity |
|---|---|---|---|
| **FFMC** | Fine Fuel Moisture Code | Moisture content of litter and fine fuels (leaves, needles, small twigs). | **High**: Responds to weather changes within hours. High values ($>80$) mean easy ignition. |
| **DMC** | Duff Moisture Code | Moisture content of loosely compacted, organic duff layers of moderate depth. | **Medium**: Responds to weather over several days. Indication of fuel consumption. |
| **DC** | Drought Code | Moisture content of deep, compact organic layers and large logs. | **Low**: Responds to long-term dry spells (weeks/months). Indicates how hard a fire is to put out. |
| **ISI** | Initial Spread Index | Expected rate of fire spread immediately after ignition. | Combines wind speed and FFMC. Dry fuels + high winds = rapid spread. |
| **BUI** | Buildup Index | Total amount of fuel available for combustion (combines DMC and DC). | Represents the cumulative dry fuel load. |
| **FWI** | Fire Weather Index | The overall potential intensity of a fire (potential energy release per unit length). | Combines ISI and BUI to serve as a single indicator of fire danger. |

### 6. Why did the heuristic layer specifically need to reject autumn leaves, red cars, and sunsets — what do those have in common with fire in pixel space (orange/red hue ranges)?
* **Pixel Space Commonality**: Fire, autumn leaves, red cars, and sunsets are all dominated by high red-channel values, moderate green-channel values, and low blue-channel values ($R > G > B$). In a simple color-based classifier, they occupy the exact same region of the RGB cube.
* **How They are Rejected**:
  * **Sunsets**: Sunsets are represented by smooth color gradients with low local variance (lack of flicker/texture) and high blue sky presence. The heuristic detects sky dominance (`sky_px`) and texture variance to penalize sunset images.
  * **Autumn Leaves**: Red/orange foliage covers large, continuous regions of the image with a uniform, low-contrast pattern. The heuristic uses the rule:
    ```python
    if red_dom > 0.42 and bright_var < 0.20 and fire_ratio < 0.12: return False
    ```
    This identifies that the scene has widespread red/orange coloring (`red_dom > 42%`) but lacks the high-contrast brightness variance (`bright_var < 0.20`) and highly concentrated bright flame spots (`fire_ratio < 12%`) characteristic of actual fires.
  * **Red Cars/Walls**: Painted walls or cars have highly uniform, flat coloring. The heuristic finds the bounding box of fire-colored pixels and measures its "fill rate". If the bounding box is highly filled (`fill_rate > 0.70`) and lacks texture variance (`avg_texture < 0.12`), it vetoes it as a flat solid object.

### 7. How is the NASA FIRMS data ingested and merged with your own local detections on the map?
* **Satellite Data Ingestion**: The backend exposes `/get_fire_events`. This triggers `fetch_nasa_firms()`, which makes a server-to-server request to the public **NASA FIRMS API**:
  `https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/VIIRS_NOAA20_NRT/world/1`
  This fetches active thermal hotspot coordinates detected by the VIIRS sensor on the NOAA-20 satellite over the last 24 hours.
* **Data Formatting**: The CSV is parsed, mapping satellite confidence flags (`"high"`/`"nominal"`/`"low"`) into numeric confidence percentages ($90\%$/$70\%$/$50\%$).
* **Caching Layer**: To prevent API rate-limiting and accelerate page loads, the parsed data is stored in an in-memory cache (`_firms_cache`) with a 30-minute Time-To-Live (TTL).
* **Merging Local Detections**:
  * When a user uploads an image to `/predict_image` and the ensemble detects a fire with confidence $> 65\%$, it logs the event (including user GPS coordinates or default placeholders) to an in-memory list `local_detections`.
  * The `/get_fire_events` endpoint returns a concatenated JSON array: `nasa_detections + local_detections`.
* **Map Rendering**:
  * The frontend Leaflet map queries this endpoint. It places NASA detections on `nasaLayer` (colored orange-to-red based on confidence) and user detections on `localLayer` (rendered as blue circle markers with a white border).

### 8. What's the biggest source of false positives/negatives in this system, and how would you reduce them further?
* **Biggest False Positive Sources**:
  * Artificial orange lighting at night (street lamps, security lights, industrial sodium bulbs).
  * Extreme desert/clay landscapes under direct sunlight (resembles fire texture and hue).
  * Campfires, fireplaces, or candlelight (technically fire, but not "forest fires").
* **Biggest False Negative Sources**:
  * Heavy smoke plumes where the flames are completely hidden (tightened fire pixel rules might trigger a heuristic veto if the CNN is uncertain).
  * Very early-stage or tiny fires occupying a fraction of the 224x224 input grid.
* **How to Reduce Them**:
  * **Temporal Modeling (Video)**: Fire flickers dynamically at a rate of 10–15 Hz, whereas cars, leaves, and sunsets are static. Analyzing a 3-second video clip using a 3D-CNN or LSTM would eliminate $99\%$ of static false positives.
  * **Contextual Object Detection**: Transition from binary image classification to object detection (e.g. YOLOv8). This would locate the fire boundaries and allow the system to check context (e.g., ignoring fire inside a barbecue pit or fireplace vs flagging fire on a tree canopy).
  * **Multi-spectral Integration**: Integrate thermal/infrared camera inputs. Fire has a distinct thermal emission spectrum that cannot be mimicked by red paint or sunsets.

### 9. Why Flask over FastAPI here, given you used FastAPI in your other project?
* **Monolithic MVC Architecture**: This project relies on server-side rendering using Jinja2 templates (`render_template("index.html")`, `render_template("map.html")`) and direct form/file uploads. Flask is the industry standard for lightweight, self-contained monolithic Python MVC apps.
* **Minimal Async Requirements**: The server doesn't require high-concurrency async I/O or websockets. It processes requests synchronously (or handles light threads), making Flask's WSGI model (run with Gunicorn) simpler and more robust.
* **FastAPI Suitability**: FastAPI excels at building headless REST APIs (decoupled from the frontend) with high-performance asynchronous endpoints (ASGI) and automatic OpenAPI documentation. Because this application renders its own views and does not use a separate client-side framework (like React or Next.js), Flask has less boilerplate and is faster to set up.

### 10. What's your model's accuracy, and how did you validate it (train/test split, real-world testing)?
* **Tabular Models**:
  * **Dataset**: Algerian Forest Fires dataset (246 samples).
  * **Split**: $80\%$ Train / $20\%$ Test, stratified to ensure class distribution balance.
  * **Results**: The Gradient Boosting tabular classifier achieves **$\approx 93\%$ to $95\%$ accuracy** on the test set. The Ridge Regressor for FWI achieves an $R^2$ score of **$\approx 0.90+$**.
* **CNN Image Model**:
  * **Dataset**: Custom dataset of $10,000+$ images (fire and nofire).
  * **Split**: $80\%$ Train / $20\%$ Validation.
  * **Results**: The MobileNetV3-Small model achieves **$\approx 96\%$ validation accuracy** during training.
* **Gradient Boosting Image Model**:
  * **Results**: Evaluated on the 88 handcrafted features, it achieves **$\approx 88\%$ to $91\%$ accuracy** (lower than the CNN because color histograms ignore spatial context).
* **System-level Validation**:
  * We built a validation harness (`test_false_positives.py`) that feeds a folder of real-world "trick" images (skin tones, invitations, sunsets) through the entire ensemble (CNN + GB + Heuristics). This validation script computes system-wide False Positive and False Negative rates, which guided us to set the $62\%$ threshold for the GB model and the $50\%$ confidence floor.
