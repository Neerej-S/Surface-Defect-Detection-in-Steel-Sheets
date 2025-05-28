import os
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory

# Initialize Flask App
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Pretrained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load(r"C:\Users\S NEEREJ\Desktop\Defect Dectetion\best_model.pth", map_location=device))
model.to(device).eval()

# Class Labels
CLASS_LABELS = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]

# Image Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Grad-CAM Function
def grad_cam(image_tensor):
    features, gradients = [], []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.layer4[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor.unsqueeze(0).to(device))
    class_idx = torch.argmax(output, dim=1).item()
    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0].mean(dim=[2, 3], keepdim=True)
    cam = torch.relu((features[0] * grads).sum(dim=1)).squeeze()
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().detach().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min()), class_idx

# Bounding Box & Heatmap Functions
def apply_mask(heatmap, threshold=0.6):
    return np.where(heatmap >= threshold, 255, 0).astype(np.uint8)

def get_bounding_box(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    return cv2.boundingRect(max(contours, key=cv2.contourArea))

def process_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = transform(image_pil)
    heatmap, class_idx = grad_cam(image_tensor)
    mask = apply_mask(heatmap)
    bbox = get_bounding_box(mask)

    # Draw bounding box
    image_cv = np.array(image_pil)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image_cv, CLASS_LABELS[class_idx], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    heatmap_img = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, "heatmap.jpg"), heatmap_img)
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, "bbox.jpg"), image_cv)

    return CLASS_LABELS[class_idx]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "input.jpg")
    file.save(file_path)

    defect = process_image(file_path)
    return jsonify({
        "defect": defect,
        "heatmap": "/uploads/heatmap.jpg",
        "bbox": "/uploads/bbox.jpg"
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
