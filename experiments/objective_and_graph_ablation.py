"""Table 8: learning-objective and graph-construction ablations."""

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.neural_network import MLPClassifier

from config import cfg
from data_loader import prepare_dataset
from dataset_pyg import WaferGraphDataset, collate_graphs
from evaluate import extract_embeddings
from model import GINJumpingKnowledge


def _eval(model, train_set, test_set, y_test):
    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=False,
                               collate_fn=collate_graphs)
    test_loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE, shuffle=False,
                              collate_fn=collate_graphs)
    z_train, y_train = extract_embeddings(model, train_loader)
    z_test, _ = extract_embeddings(model, test_loader)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                         random_state=cfg.SEED, early_stopping=True)
    clf.fit(z_train, y_train)
    acc = accuracy_score(y_test, clf.predict(z_test))
    ari = adjusted_rand_score(
        y_test, AgglomerativeClustering(n_clusters=cfg.NUM_CLASSES).fit_predict(z_test)
    )
    return {"accuracy": acc, "ari": ari}


def run():
    """This script orchestrates the ablations; full re-pretraining for each
    variant is expensive — run the variants individually for paper-quality
    numbers. Here we provide the scaffold."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    train_df, _, test_df = prepare_dataset()

    train_set = WaferGraphDataset(train_df, precompute_descriptors=False)
    test_set = WaferGraphDataset(test_df, precompute_descriptors=False)
    y_test = np.array(test_df["label"].tolist())

    # Random initialization
    model = GINJumpingKnowledge().to(cfg.DEVICE)
    rand_init = _eval(model, train_set, test_set, y_test)

    # Pre-trained checkpoint (contrastive)
    contrastive = {"checkpoint": os.path.join(cfg.CHECKPOINT_DIR, "gin_pretrained.pt")}
    if os.path.exists(contrastive["checkpoint"]):
        model = GINJumpingKnowledge().to(cfg.DEVICE)
        model.load_state_dict(torch.load(contrastive["checkpoint"], map_location=cfg.DEVICE))
        contrastive.update(_eval(model, train_set, test_set, y_test))

    # Other learning objectives (reconstruction, rotation) require dedicated
    # training scripts — see experiments.objective_recon and objective_rotation.
    results = {
        "random_init": rand_init,
        "contrastive": contrastive,
    }
    with open(os.path.join(cfg.RESULTS_DIR, "table8_objective.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run()
