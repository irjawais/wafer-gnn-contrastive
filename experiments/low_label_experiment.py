"""Section 4.3: 25%-labeled-subset experiment (claim: 95.7% accuracy)."""

import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

from config import cfg
from data_loader import prepare_dataset
from dataset_pyg import WaferGraphDataset, collate_graphs
from evaluate import extract_embeddings
from model import GINJumpingKnowledge


def run():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    train_df, _, test_df = prepare_dataset()

    rng = np.random.default_rng(cfg.SEED)
    subset_idx = rng.choice(len(train_df), size=int(0.25 * len(train_df)), replace=False)
    train_subset = train_df.iloc[subset_idx].reset_index(drop=True)

    train_loader = DataLoader(WaferGraphDataset(train_subset, precompute_descriptors=False), batch_size=cfg.BATCH_SIZE,
                               shuffle=False, collate_fn=collate_graphs)
    test_loader = DataLoader(WaferGraphDataset(test_df, precompute_descriptors=False), batch_size=cfg.BATCH_SIZE,
                              shuffle=False, collate_fn=collate_graphs)

    model = GINJumpingKnowledge().to(cfg.DEVICE)
    ckpt = os.path.join(cfg.CHECKPOINT_DIR, "gin_pretrained.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE))

    z_tr, y_tr = extract_embeddings(model, train_loader)
    z_te, y_te = extract_embeddings(model, test_loader)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                         random_state=cfg.SEED, early_stopping=True).fit(z_tr, y_tr)
    preds = clf.predict(z_te)

    out = {
        "label_fraction": 0.25,
        "accuracy": accuracy_score(y_te, preds),
        "macro_f1": f1_score(y_te, preds, average="macro", zero_division=0),
    }
    with open(os.path.join(cfg.RESULTS_DIR, "low_label_25pct.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(out)


if __name__ == "__main__":
    run()
