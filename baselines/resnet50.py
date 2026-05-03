"""ResNet-50 baseline (He et al., 2016) — ImageNet pre-trained, fine-tuned."""

import torch
import torch.nn as nn
from torchvision import models

from config import cfg


class ResNet50Wafer(nn.Module):
    def __init__(self, num_classes: int = cfg.NUM_CLASSES):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone

    def forward(self, x):
        # 1-channel wafer expanded to 3 channels for ImageNet weights
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)
