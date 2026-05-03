"""Figure 17: confusion matrix plot."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

from config import cfg
from data_loader import prepare_dataset
from dataset_pyg import WaferGraphDataset, collate_graphs
from evaluate import extract_embeddings
from model import GINJumpingKnowledge


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
    preds = clf.predict(z_te)
    cm = confusion_matrix(y_te, preds, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=cfg.CLASS_NAMES, yticklabels=cfg.CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized)")
    fig.tight_layout()
    out = os.path.join(cfg.RESULTS_DIR, "fig_confusion_matrix.png")
    fig.savefig(out, dpi=200)
    print(f"Saved: {out}")


if __name__ == "__main__":
    run()
