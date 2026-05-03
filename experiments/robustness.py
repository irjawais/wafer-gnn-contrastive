"""Section 5: robustness to Gaussian noise (sigma=0.1)."""

import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from config import cfg
from data_loader import prepare_dataset
from dataset_pyg import WaferGraphDataset, collate_graphs
from evaluate import extract_embeddings
from model import GINJumpingKnowledge


def add_gaussian_noise_to_features(z, sigma):
    return z + np.random.normal(0, sigma, z.shape).astype(np.float32)


def run():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    train_df, _, test_df = prepare_dataset()

    train_loader = DataLoader(WaferGraphDataset(train_df, precompute_descriptors=False), batch_size=cfg.BATCH_SIZE,
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

    results = {}
    for sigma in [0.0, 0.05, 0.1, 0.2]:
        z_noisy = add_gaussian_noise_to_features(z_te, sigma)
        acc = accuracy_score(y_te, clf.predict(z_noisy))
        results[f"sigma={sigma}"] = acc

    with open(os.path.join(cfg.RESULTS_DIR, "robustness.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(results)


if __name__ == "__main__":
    run()
