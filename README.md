# 🔥 Forest Fire Detection & Prediction System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-WebApp-black)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Ensemble-green)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-orange)
![Deployment](https://img.shields.io/badge/Deployment-Render-success)

An AI-powered web application that detects and predicts forest fires using:

- 🌡️ Weather & environmental data
- 🖼️ Forest image analysis
- 🤖 Ensemble Machine Learning + CNN models

The system combines:
- **MobileNetV3 CNN**
- **Gradient Boosting Classifier**
- **Color-based Fire Heuristic**

to improve detection reliability and reduce false negatives.

---

# 🌐 Live Demo

👉 https://forest-fire-1-xcci.onrender.com/

---

# 📸 Screenshots

## Home Page
(Add screenshot here)

## Fire Detection Result
(Add screenshot here)

## Fire Map
(Add screenshot here)

---

# ✨ Features

- 🔥 Forest fire image detection
- 🌡️ Weather-based fire risk prediction
- 🤖 Ensemble AI prediction system
- 🧠 CNN + ML hybrid architecture
- 📍 Fire event map visualization
- 📊 Confidence scoring
- ⚡ Real-time web predictions
- 📱 Responsive UI
- ☁️ Cloud deployment on Render

---

# 🧠 AI / ML Architecture

The application uses a weighted ensemble prediction system:

| Model | Purpose |
|------|------|
| MobileNetV3 CNN | Deep-learning image fire detection |
| Gradient Boosting | Feature-based image classification |
| Color Heuristic | Fast pixel-level fire detection |

### Ensemble Logic
- Each model casts a weighted vote
- Final prediction is determined using weighted confidence
- Tie cases are biased toward FIRE detection for safety

---

# 🛠 Tech Stack

## Backend
- Python
- Flask

## Machine Learning
- PyTorch
- torchvision
- timm
- scikit-learn
- NumPy

## Frontend
- HTML
- CSS
- JavaScript

## Deployment
- Render
- GitHub

---

# 📂 Project Structure

```bash
forest-fire/
│
├── app.py
├── requirements.txt
├── train_models.py
├── download_data.py
│
├── dataset/
│   ├── Algerian_forest_fires_dataset.csv
│   └── images/
│       ├── fire/
│       └── nofire/
│
├── models/
│   ├── cnn_meta.pkl
│   ├── cnn_fire_model.pth
│   ├── image_model.pkl
│   ├── classifier.pkl
│   ├── regressor.pkl
│   └── scaler.pkl
│
├── templates/
│   ├── index.html
│   └── map.html
│
├── static/
│
└── notebooks/
    └── EDA_and_Training.ipynb
