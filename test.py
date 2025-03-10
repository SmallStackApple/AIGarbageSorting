import torch
from torchvision import transforms
from PIL import Image
from model import GarbageClassifier
import os


# 加载模型
def load_model(model_path, num_classes=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GarbageClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# 预测函数
def predict_image(image, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # 添加batch维度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    return predicted_class


if __name__ == "__main__":
    model_path = "models/GarbageSortingModel.pth"
    class_names = ["Harmful", "Kitchen", "Other", "Recyclable"]  # 更新为4个类别
    model = load_model(model_path, num_classes=4)  # 明确指定类别数为4

    test_images_dir = "E:/PyProject/AIGarbageSorting/test_images"
    for image_name in os.listdir(test_images_dir):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_images_dir, image_name)
            image = Image.open(image_path).convert("RGB")
            result = predict_image(image, model, class_names)
            print(f"Image: {image_name}, Predicted class: {result}")
