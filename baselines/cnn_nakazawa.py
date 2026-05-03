"""Traditional CNN baseline (Nakazawa & Kulkarni, 2018).

5-layer CNN: 3 convolutional + 2 fully connected layers, adapted to 25x25.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


class TraditionalCNN(nn.Module):
    def __init__(self, num_classes: int = cfg.NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 25 -> 12 -> 6 -> 3 (after 3 poolings)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
