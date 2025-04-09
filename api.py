from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

class GrapeDiseaseModel(nn.Module):
    def __init__(self, num_classes=4):
        super(GrapeDiseaseModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GrapeDiseaseModel().to(device)
model_path = "grape_disease_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Black Measles', 'Black Rot', 'Healthy', 'Isariopsis Leaf Spot'])

def predict_image(image):
    image = image.convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    return label_encoder.inverse_transform([predicted_class.cpu().item()])[0]

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        image = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    prediction = predict_image(image)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
