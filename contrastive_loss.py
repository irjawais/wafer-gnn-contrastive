"""Contrastive loss for self-supervised pre-training.

L_contrastive = sum_{i,j} [
    1[y_i = y_j] * d_ij^2 +
    1[y_i != y_j] * max(0, m - d_ij)^2
]

with optional class-balanced weighting w_ij = 1 / sqrt(freq_i * freq_j).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = cfg.MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, pair_labels: torch.Tensor,
                weights: torch.Tensor = None) -> torch.Tensor:
        """
        embeddings: (N, D) — graph-level embeddings
        pair_labels: (N, N) with values in {1, 0, -1} where -1 = exclude
        weights: optional (N, N) class-balance weights
        """
        N = embeddings.size(0)
        d = torch.cdist(embeddings, embeddings, p=2)

        pos_mask = (pair_labels == 1).float()
        neg_mask = (pair_labels == 0).float()
        valid_mask = pos_mask + neg_mask

        pos_loss = pos_mask * d.pow(2)
        neg_loss = neg_mask * F.relu(self.margin - d).pow(2)

        loss_matrix = pos_loss + neg_loss
        if weights is not None:
            loss_matrix = loss_matrix * weights

        denom = valid_mask.sum().clamp(min=1.0)
        return loss_matrix.sum() / denom


def class_balance_weights(class_freq: dict, batch_labels: torch.Tensor) -> torch.Tensor:
    """Return (N, N) tensor of w_ij = 1 / sqrt(freq_i * freq_j)."""
    inv_sqrt_freq = torch.tensor(
        [1.0 / (class_freq.get(int(c.item()), 1.0) ** 0.5) for c in batch_labels],
        device=batch_labels.device, dtype=torch.float32,
    )
    return inv_sqrt_freq.unsqueeze(0) * inv_sqrt_freq.unsqueeze(1)
