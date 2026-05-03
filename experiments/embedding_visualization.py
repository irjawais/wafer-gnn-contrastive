"""Figures 15-16: PCA and t-SNE embedding visualizations."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from config import cfg
from data_loader import prepare_dataset
from dataset_pyg import WaferGraphDataset, collate_graphs
from evaluate import extract_embeddings
from model import GINJumpingKnowledge


def _scatter(ax, coords, labels, title):
    cmap = plt.get_cmap("tab10")
    for cls in range(cfg.NUM_CLASSES):
        mask = labels == cls
        ax.scatter(coords[mask, 0], coords[mask, 1], s=4, alpha=0.6,
                   color=cmap(cls), label=cfg.CLASS_NAMES[cls])
    ax.set_title(title)
    ax.legend(fontsize=6, loc="best")


def run():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    _, _, test_df = prepare_dataset()
    loader = DataLoader(WaferGraphDataset(test_df, precompute_descriptors=False), batch_size=cfg.BATCH_SIZE,
                        shuffle=False, collate_fn=collate_graphs)

    model = GINJumpingKnowledge().to(cfg.DEVICE)
    ckpt = os.path.join(cfg.CHECKPOINT_DIR, "gin_pretrained.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE))

    z, y = extract_embeddings(model, loader)
    sample_idx = np.random.default_rng(cfg.SEED).choice(len(z), size=min(5000, len(z)), replace=False)
    z, y = z[sample_idx], y[sample_idx]

    pca = PCA(n_components=2).fit(z)
    pca_coords = pca.transform(z)
    var_explained = pca.explained_variance_ratio_.sum()

    tsne_coords = TSNE(n_components=2, perplexity=30, random_state=cfg.SEED).fit_transform(z)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _scatter(axes[0], pca_coords, y, f"PCA (var={var_explained:.1%})")
    _scatter(axes[1], tsne_coords, y, "t-SNE")
    fig.tight_layout()
    out = os.path.join(cfg.RESULTS_DIR, "fig_pca_tsne.png")
    fig.savefig(out, dpi=200)
    print(f"Saved: {out} (PCA variance: {var_explained:.3f})")


if __name__ == "__main__":
    run()
