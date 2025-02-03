import torch
import os
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import GarbageDetectionModel, MaterialAnalysisModel  # 修改: 引入新的材质分析模型
from utils import GarbageDetectionDataset, MaterialAnalysisDataset, create_data_loaders  # 修改: 引入新的材质分析数据集

# 使用 create_data_loaders 函数加载数据
data_dir = 'labeled_images'  # 更新数据路径
train_loader_detection, val_loader_detection, test_loader_detection = create_data_loaders(data_dir, batch_size=32, dataset_class=GarbageDetectionDataset)
train_loader_material, val_loader_material, test_loader_material = create_data_loaders(data_dir, batch_size=32, dataset_class=MaterialAnalysisDataset)  # 修改: 添加材质分析数据加载器

# 模型、损失函数和优化器
model_detection = GarbageDetectionModel(num_classes=len(train_loader_detection.dataset.dataset.class_labels))  # 修改: 使用类别标签数量
model_material = MaterialAnalysisModel(num_classes=len(train_loader_material.dataset.dataset.class_labels))  # 修改: 使用类别标签数量
criterion_detection = nn.MSELoss()  # 修改: 使用均方误差损失函数
criterion_material = nn.CrossEntropyLoss()  # 修改: 添加材质分析损失函数
optimizer_detection = optim.Adam(model_detection.parameters(), lr=0.001)
optimizer_material = optim.Adam(model_material.parameters(), lr=0.001)  # 修改: 添加材质分析优化器

# 新增裁剪函数
def crop_image(image, box):
    x1, y1, x2, y2 = box
    return image[:, y1:y2, x1:x2]

# 新增函数：保存裁剪后的图片到 unlabeled_images 目录
def save_cropped_image(image, epoch, index, unlabeled_dir):
    image_pil = Image.fromarray((image * 255).astype(np.uint8).transpose(1, 2, 0))
    image_path = os.path.join(unlabeled_dir, f'cropped_{epoch}_{index}.jpg')
    image_pil.save(image_path)

# 训练函数
def train_detection(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, boxes, labels in train_loader:
        images, boxes = images.to(device), boxes.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, boxes)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def train_material(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

# 验证函数
def validate_detection(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, boxes, labels in val_loader:
            images, boxes = images.to(device), boxes.to(device)
            outputs = model(images)
            loss = criterion(outputs, boxes)
            running_loss += loss.item()
    return running_loss / len(val_loader)

def validate_material(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(val_loader), correct / total

# 训练循环
device = torch.device('cpu')
num_epochs = 10
unlabeled_dir = 'unlabeled_images'  # 新增: 定义 unlabeled_images 目录
for epoch in range(num_epochs):
    train_loss_detection = train_detection(model_detection, train_loader_detection, criterion_detection, optimizer_detection, device)
    val_loss_detection = validate_detection(model_detection, val_loader_detection, criterion_detection, device)
    
    # 检测模式下先识别物体位置，再裁剪并识别材质
    for images, boxes, labels in train_loader_detection:
        images, boxes = images.to(device), boxes.to(device)
        with torch.no_grad():
            detected_boxes = model_detection(images)
        for i, box in enumerate(detected_boxes):
            cropped_image = crop_image(images[i], box.cpu().numpy())
            save_cropped_image(cropped_image.cpu().numpy(), epoch, i, unlabeled_dir)  # 新增: 保存裁剪后的图片
            cropped_image = transforms.ToTensor()(cropped_image).unsqueeze(0).to(device)
            material_label = train_loader_material.dataset.dataset.annotations[os.path.splitext(os.path.basename(train_loader_detection.dataset.dataset.images[i]))[0]]['material_labels']
            material_label = torch.tensor([material_label]).to(device)
            optimizer_material.zero_grad()
            material_output = model_material(cropped_image)
            material_loss = criterion_material(material_output, material_label)
            material_loss.backward()
            optimizer_material.step()
    
    train_loss_material = train_material(model_material, train_loader_material, criterion_material, optimizer_material, device)
    val_loss_material, val_acc_material = validate_material(model_material, val_loader_material, criterion_material, device)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss Detection: {train_loss_detection:.4f}, Val Loss Detection: {val_loss_detection:.4f}, Train Loss Material: {train_loss_material:.4f}, Val Loss Material: {val_loss_material:.4f}, Val Acc Material: {val_acc_material:.4f}')

# 新增函数：自动训练并清理 labeled_images 目录
def train_and_cleanup():
    # 使用 create_data_loaders 函数加载数据
    data_dir = 'labeled_images'
    train_loader_detection, val_loader_detection, test_loader_detection = create_data_loaders(data_dir, batch_size=32, dataset_class=GarbageDetectionDataset)
    train_loader_material, val_loader_material, test_loader_material = create_data_loaders(data_dir, batch_size=32, dataset_class=MaterialAnalysisDataset)  # 修改: 添加材质分析数据加载器

    # 模型、损失函数和优化器
    model_detection = GarbageDetectionModel(num_classes=len(train_loader_detection.dataset.dataset.class_labels))  # 修改: 使用类别标签数量
    model_material = MaterialAnalysisModel(num_classes=len(train_loader_material.dataset.dataset.class_labels))  # 修改: 使用类别标签数量
    criterion_detection = nn.MSELoss()
    criterion_material = nn.CrossEntropyLoss()  # 修改: 添加材质分析损失函数
    optimizer_detection = optim.Adam(model_detection.parameters(), lr=0.001)
    optimizer_material = optim.Adam(model_material.parameters(), lr=0.001)  # 修改: 添加材质分析优化器

    # 训练循环
    device = torch.device('cpu')
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss_detection = train_detection(model_detection, train_loader_detection, criterion_detection, optimizer_detection, device)
        val_loss_detection = validate_detection(model_detection, val_loader_detection, criterion_detection, device)
        
        # 检测模式下先识别物体位置，再裁剪并识别材质
        for images, boxes, labels in train_loader_detection:
            images, boxes = images.to(device), boxes.to(device)
            with torch.no_grad():
                detected_boxes = model_detection(images)
            for i, box in enumerate(detected_boxes):
                cropped_image = crop_image(images[i], box.cpu().numpy())
                cropped_image = transforms.ToTensor()(cropped_image).unsqueeze(0).to(device)
                material_label = train_loader_material.dataset.dataset.annotations[os.path.splitext(os.path.basename(train_loader_detection.dataset.dataset.images[i]))[0]]['material_labels']
                material_label = torch.tensor([material_label]).to(device)
                optimizer_material.zero_grad()
                material_output = model_material(cropped_image)
                material_loss = criterion_material(material_output, material_label)
                material_loss.backward()
                optimizer_material.step()
        
        train_loss_material = train_material(model_material, train_loader_material, criterion_material, optimizer_material, device)
        val_loss_material, val_acc_material = validate_material(model_material, val_loader_material, criterion_material, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss Detection: {train_loss_detection:.4f}, Val Loss Detection: {val_loss_detection:.4f}, Train Loss Material: {train_loss_material:.4f}, Val Loss Material: {val_loss_material:.4f}, Val Acc Material: {val_acc_material:.4f}')
    
    # 删除 labeled_images 目录中的所有图片
    for filename in os.listdir(data_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(data_dir, filename)
            os.remove(file_path)
            print(f'Deleted: {file_path}')

# 主程序中调用 train_and_cleanup 函数
if __name__ == "__main__":
    train_and_cleanup()