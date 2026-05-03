"""Convolutional autoencoder baseline (Yu et al., 2020).

Trained with reconstruction loss on unlabeled wafer maps; bottleneck
representations are used as features for downstream classification.
"""

import torch
import torch.nn as nn

from config import cfg


class WaferAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),    # 25 -> 13
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),   # 13 -> 7
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 7 -> 4
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 4), nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=0),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
