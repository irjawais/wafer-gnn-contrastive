"""Convert wafer maps to graph representations with 8-connectivity.

Edge weights combine:
    w_geom: 1.0 for direct (H/V), 1/sqrt(2) for diagonal
    w_sem:  alpha if v_i == v_j, (1-alpha) otherwise
"""

import math
import numpy as np
import torch
from torch_geometric.data import Data

from config import cfg


def build_node_features(wafer: np.ndarray) -> np.ndarray:
    """9-dimensional node features per the paper."""
    m, n = wafer.shape
    diag = math.sqrt(m ** 2 + n ** 2) / 2
    features = np.zeros((m * n, cfg.NODE_FEATURE_DIM), dtype=np.float32)

    for i in range(m):
        for j in range(n):
            idx = i * n + j
            v = wafer[i, j]
            d_c = math.sqrt((i - m / 2) ** 2 + (j - n / 2) ** 2) / diag
            d_e = min(i, j, m - 1 - i, n - 1 - j) / max(m, n)
            theta = math.atan2(j - n / 2, i - m / 2) / math.pi

            # Local defect density in 5x5 neighborhood
            i0, i1 = max(0, i - 2), min(m, i + 3)
            j0, j1 = max(0, j - 2), min(n, j + 3)
            patch = wafer[i0:i1, j0:j1]
            density = float(np.mean(patch == 2))

            # Local entropy
            counts = np.bincount(patch.flatten(), minlength=3)
            probs = counts / max(counts.sum(), 1)
            entropy = -np.sum([p * math.log(p + 1e-9) for p in probs if p > 0])

            features[idx] = [
                i / m, j / n, float(v), d_c, d_e, theta, density, entropy, 1.0
            ]
    return features


def build_edge_index_and_weights(wafer: np.ndarray, alpha: float = cfg.ALPHA_SEM):
    """8-connectivity edges with geometric and semantic weights."""
    m, n = wafer.shape
    edges, weights = [], []
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    for i in range(m):
        for j in range(n):
            src = i * n + j
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m and 0 <= nj < n:
                        dst = ni * n + nj
                        w_geom = 1.0 if (di == 0 or dj == 0) else inv_sqrt2
                        same = (wafer[i, j] == wafer[ni, nj])
                        w_sem = alpha if same else (1 - alpha)
                        edges.append((src, dst))
                        weights.append(w_geom * w_sem)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


def wafer_to_graph(wafer: np.ndarray, label: int = -1) -> Data:
    """Convert a single wafer map to a PyG Data object."""
    x = torch.tensor(build_node_features(wafer), dtype=torch.float32)
    edge_index, edge_weight = build_edge_index_and_weights(wafer)
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
