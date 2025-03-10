import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from model import GarbageClassifier

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 加载数据集
def load_data(data_dir, batch_size=32):
    # 自定义数据集类，增加异常处理
    class SafeImageFolder(datasets.ImageFolder):
        def __getitem__(self, index):
            try:
                return super().__getitem__(index)
            except Exception as e:
                print(f"Error loading image: {self.imgs[index][0]} - {e}")
                return None  # 返回 None 表示跳过该图片

    dataset = SafeImageFolder(root=data_dir, transform=transform)
    # 过滤掉返回值为 None 的样本
    dataset.samples = [sample for sample in dataset.samples if SafeImageFolder(root=data_dir, transform=transform).__getitem__(dataset.samples.index(sample)) is not None]
    class_names = dataset.classes  # 获取目录名作为类别名称
    print(f"Detected classes: {class_names}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, class_names  # 返回类别名称


# 训练函数
def train_model(data_dir, save_path="models/GarbageSortingModel.pth", epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader, class_names = load_data(data_dir)  # 获取类别名称
    # 使用weights参数替代pretrained参数
    model = GarbageClassifier(num_classes=len(class_names)).to(device)  # 动态指定类别数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}")

    # 保存模型
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    print(f"Trained classes: {class_names}")  # 打印训练的类别名称


if __name__ == "__main__":
    data_dir = "E:/PyProject/AIGarbageSorting/untrained_image"  # 确保路径指向untrained_image目录
    train_model(data_dir)
