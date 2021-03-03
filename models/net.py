import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.net = torchvision.models.resnet18(pretrained=False)
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, num_classes)
        # trick from https://gist.github.com/y0ast/d91d09565462125a1eb75acc65da1469
        self.net.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.net.maxpool = nn.Identity()
    def forward(self, x):
        x = self.net(x)
        return x