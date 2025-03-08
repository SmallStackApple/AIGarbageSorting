import torch
import torch.nn as nn
from torchvision import models

class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=4):  # 将默认类别数从6改回4
        super(GarbageClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)