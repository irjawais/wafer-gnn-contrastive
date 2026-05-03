"""Table 5: per-class F1 (mean +/- std over 5 seeds)."""

import os
import json

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader

from config import cfg
from data_loader import prepare_dataset
from dataset_pyg import WaferGraphDataset, collate_graphs
from evaluate import extract_embeddings
from model import GINJumpingKnowledge


def run():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    train_df, _, test_df = prepare_dataset()

    train_set = WaferGraphDataset(train_df, precompute_descriptors=False)
    test_set = WaferGraphDataset(test_df, precompute_descriptors=False)
    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE,
                               shuffle=False, collate_fn=collate_graphs)
    test_loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE,
                              shuffle=False, collate_fn=collate_graphs)

    ckpt = os.path.join(cfg.CHECKPOINT_DIR, "gin_pretrained.pt")
    model = GINJumpingKnowledge().to(cfg.DEVICE)
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE))

    z_train, y_train = extract_embeddings(model, train_loader)
    z_test, y_test = extract_embeddings(model, test_loader)

    f1s, precs, recs = [], [], []
    for seed in range(5):
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                             random_state=seed, early_stopping=True)
        clf.fit(z_train, y_train)
        preds = clf.predict(z_test)
        f1s.append(f1_score(y_test, preds, average=None, zero_division=0))
        precs.append(precision_score(y_test, preds, average=None, zero_division=0))
        recs.append(recall_score(y_test, preds, average=None, zero_division=0))

    f1s = np.array(f1s); precs = np.array(precs); recs = np.array(recs)
    table = {}
    for i, name in enumerate(cfg.CLASS_NAMES):
        table[name] = {
            "f1_mean": float(f1s[:, i].mean()), "f1_std": float(f1s[:, i].std()),
            "precision": float(precs[:, i].mean()), "recall": float(recs[:, i].mean()),
        }
    table["macro_average"] = {
        "f1_mean": float(f1s.mean()), "precision": float(precs.mean()), "recall": float(recs.mean()),
    }
    with open(os.path.join(cfg.RESULTS_DIR, "table5_per_class_f1.json"), "w") as f:
        json.dump(table, f, indent=2)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    run()
