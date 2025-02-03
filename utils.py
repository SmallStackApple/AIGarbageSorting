import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json

class GarbageDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.annotations = {os.path.splitext(img)[0]: self.load_annotations(img) for img in self.images}
        self.class_labels = ['plastic', 'paper', 'metal', 'glass', 'organic', 'battery', 'trash']  # 新增: 垃圾类别标签

    def load_annotations(self, img_name):
        annotation_path = os.path.join(self.root_dir, f'{os.path.splitext(img_name)[0]}_annotations.json')
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                return json.load(f)
        return {}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        annotation = self.annotations.get(os.path.splitext(img_name)[0], {})

        if self.transform:
            image = self.transform(image)

        return image, annotation['boxes'], annotation['labels']  # 修改: 返回图像、标注框和标签

class MaterialAnalysisDataset(Dataset):  # 修改: 添加材质分析数据集
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.annotations = {os.path.splitext(img)[0]: self.load_annotations(img) for img in self.images}
        self.class_labels = ['plastic', 'paper', 'metal', 'glass', 'organic', 'battery', 'trash']  # 新增: 垃圾类别标签

    def load_annotations(self, img_name):
        annotation_path = os.path.join(self.root_dir, f'{os.path.splitext(img_name)[0]}_annotations.json')
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                return json.load(f)
        return {}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        annotation = self.annotations.get(os.path.splitext(img_name)[0], {})

        if self.transform:
            image = self.transform(image)

        return image, annotation['material_labels']  # 返回图像和材质标签

def create_data_loaders(data_dir, batch_size, dataset_class):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    dataset = dataset_class(root_dir=data_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader  # 返回训练集、验证集和测试集