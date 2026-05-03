"""Table 4: feature ablation (Position+Value / +Spatial / +Contextual)."""

import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.neural_network import MLPClassifier

from config import cfg
from data_loader import prepare_dataset
from dataset_pyg import WaferGraphDataset, collate_graphs
from graph_construction import wafer_to_graph
from model import GINJumpingKnowledge
from train import train_self_supervised
import graph_construction as gc


FEATURE_GROUPS = {
    "position_value":   [True, False, False],   # i/m, j/n, value
    "with_spatial":     [True, True, False],    # + d_c, d_e, theta
    "with_contextual":  [True, True, True],     # + density, entropy, bias
}


def _build_features_subset(wafer, mask):
    full = gc.build_node_features(wafer)
    keep = []
    if mask[0]: keep += [0, 1, 2]
    if mask[1]: keep += [3, 4, 5]
    if mask[2]: keep += [6, 7, 8]
    return full[:, keep]


def run():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    train_df, _, test_df = prepare_dataset()
    results = {}

    for name, mask in FEATURE_GROUPS.items():
        print(f"\n--- Feature group: {name} ---")
        # Monkey-patch the node-feature builder to honor the ablation mask
        original = gc.build_node_features
        gc.build_node_features = lambda w, _m=mask: _build_features_subset(w, _m)

        ckpt = train_self_supervised(
            train_df=train_df.head(2000),  # small subset for ablation; remove cap for full run
            val_df=test_df.head(500),
            ckpt_name=f"feat_{name}.pt",
            epochs=20,
            augment=False,
        )
        gc.build_node_features = original

        # Evaluate
        model = GINJumpingKnowledge(
            node_feat_dim=sum([3 if m else 0 for m in mask]),
        ).to(cfg.DEVICE)
        # ... (full eval omitted for brevity in driver; rerun pretrain_main eval)
        results[name] = {"checkpoint": ckpt}

    with open(os.path.join(cfg.RESULTS_DIR, "table4_features.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run()
