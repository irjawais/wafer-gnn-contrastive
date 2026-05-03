"""GIN with Jumping Knowledge for wafer defect detection.

Architecture (per paper):
    4 GINConv layers: 64 -> 128 -> 256 -> 128
    Concatenate all layer outputs -> 576-dim
    Project via FC to 128-dim embedding for contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

from config import cfg


def make_gin_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
    )


class GINJumpingKnowledge(nn.Module):
    """GIN with Jumping Knowledge concatenation."""

    def __init__(
        self,
        node_feat_dim: int = cfg.NODE_FEATURE_DIM,
        hidden_dims=tuple(cfg.GIN_HIDDEN_DIMS),
        proj_dim: int = cfg.PROJECTION_DIM,
        dropout: float = cfg.DROPOUT,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        in_dim = node_feat_dim
        for h in hidden_dims:
            self.convs.append(GINConv(make_gin_mlp(in_dim, h)))
            in_dim = h

        self.jk_dim = sum(hidden_dims)
        self.projection = nn.Linear(self.jk_dim, proj_dim)
        self.classifier = nn.Linear(proj_dim, cfg.NUM_CLASSES)

    def encode(self, x, edge_index, batch):
        """Return graph-level embedding (after JK concat + projection)."""
        layer_outs = []
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            layer_outs.append(global_mean_pool(h, batch))

        h_jk = torch.cat(layer_outs, dim=-1)
        z = self.projection(h_jk)
        return z

    def forward(self, x, edge_index, batch):
        """Return embedding and logits."""
        z = self.encode(x, edge_index, batch)
        logits = self.classifier(z)
        return z, logits
