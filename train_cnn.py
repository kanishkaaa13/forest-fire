import torch
import timm
from PIL import Image
import torchvision.transforms as transforms

# Load your model
model = timm.create_model("mobilenetv3_small_075", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("models/cnn_fire_model.pth", map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open("path/to/the/fire_image_in_screenshot.jpg").convert("RGB")
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.softmax(outputs[0], dim=0)
    print(f"Fire probability: {probs[1].item():.4f} | NoFire: {probs[0].item():.4f}")