"""Evaluation: clustering metrics + supervised classification on embeddings.

Two evaluation paths:
    1. Clustering + sklearn MLP on frozen pre-trained embeddings  (ARI 0.89)
    2. In-model logits from the fine-tuned backbone               (Acc 98.6%, ARI 0.98)
"""

import os
import random

import numpy as np
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, classification_report,
    confusion_matrix, f1_score, normalized_mutual_info_score, silhouette_score,
)
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader

from config import cfg
from data_loader import prepare_dataset
from dataset_pyg import WaferGraphDataset, collate_graphs
from model import GINJumpingKnowledge


@torch.no_grad()
def extract_embeddings(model: GINJumpingKnowledge, loader: DataLoader):
    model.eval()
    all_z, all_y = [], []
    for graphs, labels, *_ in loader:
        graphs = graphs.to(cfg.DEVICE)
        z = model.encode(graphs.x, graphs.edge_index, graphs.batch)
        all_z.append(z.cpu().numpy())
        all_y.append(labels.numpy())
    return np.concatenate(all_z), np.concatenate(all_y)


@torch.no_grad()
def predict_with_classifier(model: GINJumpingKnowledge, loader: DataLoader):
    """Use the in-model classifier head — for evaluating fine-tuned checkpoints."""
    model.eval()
    preds, ys, embeds = [], [], []
    for graphs, labels, *_ in loader:
        graphs = graphs.to(cfg.DEVICE)
        z, logits = model(graphs.x, graphs.edge_index, graphs.batch)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        ys.append(labels.numpy())
        embeds.append(z.cpu().numpy())
    return np.concatenate(embeds), np.concatenate(ys), np.concatenate(preds)


def evaluate_clustering(embeddings: np.ndarray, labels: np.ndarray):
    """Run K-means, Agglomerative, DBSCAN — report Silhouette, ARI, NMI."""
    results = {}
    for name, algo in [
        ("KMeans", KMeans(n_clusters=cfg.NUM_CLASSES, random_state=cfg.SEED, n_init=10)),
        ("Agglomerative", AgglomerativeClustering(n_clusters=cfg.NUM_CLASSES)),
        ("DBSCAN", DBSCAN(eps=0.5, min_samples=5)),
    ]:
        preds = algo.fit_predict(embeddings)
        try:
            sil = silhouette_score(embeddings, preds) if len(set(preds)) > 1 else float("nan")
        except Exception:
            sil = float("nan")
        results[name] = {
            "silhouette": sil,
            "ari": adjusted_rand_score(labels, preds),
            "nmi": normalized_mutual_info_score(labels, preds),
        }
    return results


def evaluate_classification(z_train, y_train, z_test, y_test):
    """Train an MLP on embeddings, evaluate on test set."""
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=300,
        random_state=cfg.SEED, early_stopping=True,
    )
    clf.fit(z_train, y_train)
    preds = clf.predict(z_test)
    return _classification_metrics(y_test, preds)


def _classification_metrics(y_true, preds):
    return {
        "accuracy": accuracy_score(y_true, preds),
        "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
        "per_class_f1": f1_score(y_true, preds, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "report": classification_report(
            y_true, preds, target_names=cfg.CLASS_NAMES, zero_division=0,
        ),
    }


def main(checkpoint: str = None, use_finetuned: bool = False):
    random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
    train_df, _, test_df = prepare_dataset()
    train_set = WaferGraphDataset(train_df, augment=False, precompute_descriptors=False)
    test_set = WaferGraphDataset(test_df, augment=False, precompute_descriptors=False)

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE,
                               shuffle=False, collate_fn=collate_graphs)
    test_loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE,
                              shuffle=False, collate_fn=collate_graphs)

    model = GINJumpingKnowledge().to(cfg.DEVICE)
    if checkpoint is None:
        checkpoint = os.path.join(
            cfg.CHECKPOINT_DIR,
            "gin_finetuned.pt" if use_finetuned else "gin_pretrained.pt",
        )
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=cfg.DEVICE))
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print(f"WARNING: checkpoint not found ({checkpoint}) — using random init.")

    if use_finetuned:
        # Path 2: in-model logits from fine-tuned backbone
        z_test, y_test, preds = predict_with_classifier(model, test_loader)
        print("\n=== Clustering on fine-tuned embeddings ===")
        for name, m in evaluate_clustering(z_test, y_test).items():
            print(f"{name}: Silhouette={m['silhouette']:.3f}  "
                  f"ARI={m['ari']:.3f}  NMI={m['nmi']:.3f}")
        print("\n=== Classification (in-model classifier) ===")
        cls = _classification_metrics(y_test, preds)
        print(f"Accuracy: {cls['accuracy']:.4f}  Macro-F1: {cls['macro_f1']:.4f}")
        print(cls["report"])
        return cls

    # Path 1: frozen embeddings + sklearn MLP
    z_train, y_train = extract_embeddings(model, train_loader)
    z_test, y_test = extract_embeddings(model, test_loader)

    print("\n=== Clustering Performance (frozen pre-trained embeddings) ===")
    for name, m in evaluate_clustering(z_test, y_test).items():
        print(f"{name}: Silhouette={m['silhouette']:.3f}  "
              f"ARI={m['ari']:.3f}  NMI={m['nmi']:.3f}")

    print("\n=== Classification Performance (sklearn MLP on frozen embeddings) ===")
    cls = evaluate_classification(z_train, y_train, z_test, y_test)
    print(f"Accuracy: {cls['accuracy']:.4f}  Macro-F1: {cls['macro_f1']:.4f}")
    print(cls["report"])
    return cls


if __name__ == "__main__":
    main()
