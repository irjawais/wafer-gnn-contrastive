"""Section 5: OOD detection — hold out one class, measure detection rate.

Following the protocol of energy-based OOD: train on N-1 classes, treat the
held-out class as anomalies. We use the maximum softmax probability of an
MLP classifier as the OOD score (lower = more anomalous).
"""

import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

from config import cfg
from data_loader import prepare_dataset
from dataset_pyg import WaferGraphDataset, collate_graphs
from evaluate import extract_embeddings
from model import GINJumpingKnowledge


def run():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    train_df, _, test_df = prepare_dataset()

    model = GINJumpingKnowledge().to(cfg.DEVICE)
    ckpt = os.path.join(cfg.CHECKPOINT_DIR, "gin_pretrained.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE))

    results = {}
    for held_out in range(cfg.NUM_CLASSES):
        train_in = train_df[train_df["label"] != held_out]
        test_in = test_df[test_df["label"] != held_out]
        test_ood = test_df[test_df["label"] == held_out]
        if len(test_ood) == 0:
            continue

        loaders = {
            "train_in": DataLoader(WaferGraphDataset(train_in, precompute_descriptors=False), batch_size=cfg.BATCH_SIZE,
                                    shuffle=False, collate_fn=collate_graphs),
            "test_in": DataLoader(WaferGraphDataset(test_in, precompute_descriptors=False), batch_size=cfg.BATCH_SIZE,
                                   shuffle=False, collate_fn=collate_graphs),
            "test_ood": DataLoader(WaferGraphDataset(test_ood, precompute_descriptors=False), batch_size=cfg.BATCH_SIZE,
                                    shuffle=False, collate_fn=collate_graphs),
        }
        z_tr, y_tr = extract_embeddings(model, loaders["train_in"])
        z_in, _ = extract_embeddings(model, loaders["test_in"])
        z_ood, _ = extract_embeddings(model, loaders["test_ood"])

        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                             random_state=cfg.SEED, early_stopping=True).fit(z_tr, y_tr)
        msp_in = clf.predict_proba(z_in).max(axis=1)
        msp_ood = clf.predict_proba(z_ood).max(axis=1)

        scores = np.concatenate([msp_in, msp_ood])
        labels = np.concatenate([np.zeros(len(msp_in)), np.ones(len(msp_ood))])
        # Higher score => in-distribution; flip for OOD AUC
        auc = roc_auc_score(labels, -scores)
        # 89.3% claim: rate at which OOD samples are flagged below threshold
        threshold = np.percentile(msp_in, 5)
        detection_rate = float(np.mean(msp_ood < threshold))
        results[cfg.CLASS_NAMES[held_out]] = {
            "auc": float(auc), "detection_rate@95tpr": detection_rate,
        }

    overall = float(np.mean([v["detection_rate@95tpr"] for v in results.values()]))
    results["overall_detection_rate"] = overall

    with open(os.path.join(cfg.RESULTS_DIR, "ood_detection.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()
