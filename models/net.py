import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.net = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.net(x)
        return x