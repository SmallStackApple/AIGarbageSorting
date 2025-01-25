import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_data_loaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader