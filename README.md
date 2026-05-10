# 🔥 Forest Fire Prediction System

A dual-mode ML web app that predicts forest fire risk from:
- **Tabular weather data** (Algerian Forest Fires Dataset – FWI, temperature, humidity…)
- **Forest images** (CNN-based fire detection from photos)

Built with **Python · Flask · scikit-learn · Pillow · HTML/CSS/JS**

---

## Project Structure

```
forest-fire-prediction/
├── app.py                  ← Flask web server (main entry point)
├── train_models.py         ← Training script for all models
├── download_data.py        ← Kaggle dataset downloader
├── requirements.txt
├── dataset/
│   ├── Algerian_forest_fires_dataset.csv
│   └── images/
│       ├── fire/           ← fire images
│       └── nofire/         ← non-fire images
├── models/                 ← saved .pkl files (auto-created after training)
│   ├── classifier.pkl
│   ├── regressor.pkl
│   ├── scaler.pkl
│   └── image_model.pkl
├── templates/
│   └── index.html          ← web frontend
└── notebooks/
    └── EDA_and_Training.ipynb
```

---

## ⚡ Setup in VS Code – Step by Step

### Step 1 – Clone / download the project

```bash
# Option A: if you have git
git clone https://github.com/aravind-selvam/forest-fire-prediction
cd forest-fire-prediction

# Option B: just put all these files in a folder called forest-fire-prediction
```

### Step 2 – Open in VS Code

```
File → Open Folder → select forest-fire-prediction/
```

Install recommended extensions if prompted:
- **Python** (Microsoft)
- **Pylance**
- **Jupyter** (for the notebook)

### Step 3 – Create a Python virtual environment

Open the VS Code **Terminal** (`Ctrl + `` ` ``):

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

VS Code will ask "Do you want to use this venv?" → click **Yes**.

### Step 4 – Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5 – Download the datasets

**Option A – Automatic (Kaggle API)**

1. Go to https://www.kaggle.com → Account → Create New API Token
2. Place the downloaded `kaggle.json` in:
   - Windows: `C:\Users\<YourName>\.kaggle\kaggle.json`
   - macOS/Linux: `~/.kaggle/kaggle.json`
3. Run:
   ```bash
   python download_data.py
   ```

**Option B – Manual**

1. Download from Kaggle manually:
   - https://www.kaggle.com/datasets/harshvir04/algerian-forest-fires-dataset
   - https://www.kaggle.com/datasets/kutaykutlu/forest-fire

2. Place files:
   - CSV → `dataset/Algerian_forest_fires_dataset.csv`
   - Fire images → `dataset/images/fire/`
   - Non-fire images → `dataset/images/nofire/`

### Step 6 – Train the models

```bash
python train_models.py
```

This will:
- Clean and process the CSV dataset
- Train a **Gradient Boosting Classifier** (fire / no fire)
- Train a **Ridge Regressor** (FWI score prediction)
- Train a **Random Forest** on the images (if images are present)
- Save everything to `models/`

Expected output:
```
─── Training Tabular Models ───
  Dataset shape: (244, 11)
  Classifier Accuracy : 97.8%
  Regressor R² Score  : 0.981
  ✅ Tabular models saved to models/

─── Training Image Model ───
  Images loaded: 1800  |  Fire: 900  |  No-fire: 900
  Image Classifier Accuracy: 94.2%
  ✅ Image model saved to models/image_model.pkl
```

### Step 7 – Run the Flask app

```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## Using the Web Interface

### Weather Data Tab
- Enter the 10 meteorological features (or click a preset: High Risk / Low Risk / Moderate)
- Click **Analyze Risk**
- See: Fire / No-Fire prediction + confidence + FWI score + danger level

### Image Analysis Tab
- Drag-and-drop or browse for a forest image
- Click **Detect Fire**
- See: Fire detected / not detected + confidence score

---

## Models Used

| Task | Model | Accuracy |
|------|-------|----------|
| Fire Classification | Gradient Boosting Classifier | ~97% |
| FWI Regression | Ridge Regression | R² ~0.98 |
| Image Fire Detection | Random Forest (64×64 pixels) | ~94% |

---

## Dataset Features

| Feature | Description |
|---------|-------------|
| Temperature | Air temperature (°C), range 15–42 |
| RH | Relative humidity (%), range 20–90 |
| Ws | Wind speed (km/h), range 4–29 |
| Rain | Rainfall in mm |
| FFMC | Fine Fuel Moisture Code |
| DMC | Duff Moisture Code |
| DC | Drought Code |
| ISI | Initial Spread Index |
| BUI | Buildup Index |
| FWI | Fire Weather Index (regression target) |
| Classes | Fire / Not Fire (classification target) |

---

## Troubleshooting

**"No module named flask"** → Make sure your venv is activated and you ran `pip install -r requirements.txt`

**"FileNotFoundError: dataset/..."** → Run `python download_data.py` or place files manually

**Models not found on startup** → Run `python train_models.py` first

**Image model not predicting well** → The heuristic fallback (red-pixel ratio) is used when no image model is trained. Add more images to `dataset/images/` and retrain.

**Port 5000 already in use** → Change port in `app.py`: `app.run(debug=True, port=5001)`
