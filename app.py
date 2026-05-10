import os, io, base64, numpy as np, pickle, torch, json
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from PIL import Image
import timm
import torchvision.transforms as transforms

app = Flask(__name__)

# Load CNN Model
DEVICE = torch.device("cpu")
model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=2)
try:
    model.load_state_dict(torch.load("models/cnn_fire_model.pth", map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    print("✅ CNN Model loaded successfully")
except:
    print("⚠ CNN model not found. Using fallback.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Store recent fire events (in memory for demo)
fire_events = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/map")
def map_page():
    return render_template("map.html")

@app.route("/predict_image", methods=["POST"])
def predict_image():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image uploaded"})

        file = request.files["image"]
        img = Image.open(file.stream).convert("RGB")

        # Get location if provided (for live cameras)
        lat = request.form.get("lat")
        lng = request.form.get("lng")

        # CNN Prediction
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)[0]
            pred = torch.argmax(prob).item()
            confidence = round(float(prob[pred]) * 100, 1)

        label = "🔥 FIRE DETECTED" if pred == 1 else "✅ NO FIRE DETECTED"

        # Save fire event if detected
        if pred == 1 and confidence > 60:   # Only save high confidence fires
            event = {
                "lat": float(lat) if lat else 18.52,
                "lng": float(lng) if lng else 73.85,
                "confidence": confidence,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "thumbnail": ""  # can add base64 later if needed
            }
            fire_events.append(event)
            # Keep only last 50 events
            if len(fire_events) > 50:
                fire_events.pop(0)

        # Create thumbnail
        buf = io.BytesIO()
        img.resize((300, 200)).save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            "success": True,
            "label": label,
            "class": pred,
            "confidence": confidence,
            "thumbnail": f"data:image/jpeg;base64,{img_b64}",
            "lat": lat,
            "lng": lng
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/get_fire_events", methods=["GET"])
def get_fire_events():
    return jsonify({"events": fire_events})

if __name__ == "__main__":
    app.run(debug=True, port=5000)