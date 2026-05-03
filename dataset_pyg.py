"""PyTorch Geometric dataset wrapping wafer maps as graphs.

Notes:
    - When `augment=True`, the wafer is rotated/flipped per sample, so structural
      descriptors (Hu moments, Fourier) and density are recomputed on the fly to
      remain consistent with the augmented view.
    - When `augment=False` and `precompute_descriptors=True`, descriptors are
      cached once at __init__ to amortize cost during training without
      augmentation (and during contrastive pair-label construction).
    - When `precompute_descriptors=False`, no caching is done — useful for
      eval-only paths (extract_embeddings) that never read these fields.
    - RNG is per-worker via `worker_init_fn` (see `make_worker_init_fn`) to
      avoid duplicated augmentation across workers.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from data_loader import augment_wafer
from graph_construction import wafer_to_graph
from similarity import defect_density, fourier_descriptors, hu_moments


def _structural_vec(wafer: np.ndarray) -> np.ndarray:
    return np.concatenate([hu_moments(wafer), fourier_descriptors(wafer)])


class WaferGraphDataset(Dataset):
    """Each item: (PyG Data, label, raw wafer map, lot_id, structural_descriptor, density)."""

    def __init__(
        self,
        df: pd.DataFrame,
        augment: bool = False,
        seed: int = 0,
        precompute_descriptors: bool = True,
    ):
        self.wafers = df["wafer_map_resized"].tolist()
        self.labels = df["label"].tolist()
        self.lots = df["lotName"].tolist() if "lotName" in df.columns else [None] * len(df)
        self.augment = augment
        self.base_seed = seed
        self.precompute_descriptors = precompute_descriptors and not augment

        if self.precompute_descriptors:
            self._structural = [_structural_vec(w) for w in self.wafers]
            self._density = [defect_density(w) for w in self.wafers]
        else:
            self._structural = None
            self._density = None

        # Will be replaced per-worker by make_worker_init_fn; default for main thread.
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.wafers)

    def __getitem__(self, idx):
        wafer = self.wafers[idx]
        if self.augment:
            wafer = augment_wafer(wafer, self.rng)
            structural = _structural_vec(wafer)
            density = defect_density(wafer)
        elif self.precompute_descriptors:
            structural = self._structural[idx]
            density = self._density[idx]
        else:
            # Compute lazily when needed (rare in eval paths)
            structural = _structural_vec(wafer)
            density = defect_density(wafer)

        label = self.labels[idx]
        graph = wafer_to_graph(wafer, label=label)
        return graph, label, wafer, self.lots[idx], structural, density


def collate_graphs(batch):
    graphs, labels, wafers, lots, structurals, densities = zip(*batch)
    batch_graph = Batch.from_data_list(list(graphs))
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return (
        batch_graph,
        labels_tensor,
        list(wafers),
        list(lots),
        np.stack(structurals),
        np.array(densities, dtype=np.float32),
    )


def make_worker_init_fn(base_seed: int = 0):
    """Return a worker_init_fn that gives each DataLoader worker a unique RNG."""

    def _init(worker_id: int):
        info = torch.utils.data.get_worker_info()
        if info is None:
            return
        ds = info.dataset
        if isinstance(ds, WaferGraphDataset):
            ds.rng = np.random.default_rng(base_seed + worker_id + 1000 * os.getpid())

    return _init
