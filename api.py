import torch
from torchvision import transforms
from PIL import Image
import os
from model import GarbageClassifier

class GarbagePredictor:
    def __init__(self, model_path="models/GarbageSortingModel.pth", num_classes=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GarbageClassifier(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def predict(self, image_paths):
        predictions = []
        with torch.no_grad():
            for image_path in image_paths:
                image = Image.open(image_path).convert("RGB")
                image = self.transform(image).unsqueeze(0).to(self.device)
                output = self.model(image)
                _, predicted = torch.max(output, 1)
                predictions.append(predicted.item())
        return predictions