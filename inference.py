import torch
from torchvision import transforms
from PIL import Image
from model import GarbageClassifier

# 加载模型
def load_model(model_path, num_classes=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GarbageClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 预测函数
def predict_image(image_path, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # 添加batch维度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    return predicted_class

if __name__ == "__main__":
    model_path = "E:/PyProject/AIGarbageSorting/model.pth"
    class_names = ["cardboard", "glass", "metal", "plastic"]  # 更新为4个类别
    model = load_model(model_path, num_classes=4)  # 明确指定类别数为4

    image_path = "E:/PyProject/AIGarbageSorting/test_image.jpg"
    result = predict_image(image_path, model, class_names)
    print(f"Predicted class: {result}")