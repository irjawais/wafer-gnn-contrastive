"""Table 7: state-of-the-art comparison after supervised fine-tuning."""

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                              normalized_mutual_info_score)

from config import cfg
from data_loader import prepare_dataset
from dataset_pyg import WaferGraphDataset, collate_graphs
from model import GINJumpingKnowledge
from train import fine_tune_classifier


def run():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    train_df, val_df, test_df = prepare_dataset()

    pretrained = os.path.join(cfg.CHECKPOINT_DIR, "gin_pretrained.pt")
    fine_ckpt = fine_tune_classifier(
        pretrained, train_df, val_df, epochs=30, lr=1e-4,
        out_ckpt="gin_finetuned.pt",
    )

    model = GINJumpingKnowledge().to(cfg.DEVICE)
    model.load_state_dict(torch.load(fine_ckpt, map_location=cfg.DEVICE))
    model.eval()

    test_loader = DataLoader(WaferGraphDataset(test_df, precompute_descriptors=False), batch_size=cfg.BATCH_SIZE,
                              shuffle=False, collate_fn=collate_graphs)

    embeddings, preds, labels = [], [], []
    with torch.no_grad():
        for graphs, y, *_ in test_loader:
            graphs = graphs.to(cfg.DEVICE)
            z, logits = model(graphs.x, graphs.edge_index, graphs.batch)
            embeddings.append(z.cpu().numpy())
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(y.tolist())
    embeddings = np.concatenate(embeddings)
    labels = np.array(labels)
    preds = np.array(preds)

    cluster_pred = AgglomerativeClustering(n_clusters=cfg.NUM_CLASSES).fit_predict(embeddings)
    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "ari_finetuned": float(adjusted_rand_score(labels, cluster_pred)),
        "nmi_finetuned": float(normalized_mutual_info_score(labels, cluster_pred)),
    }
    with open(os.path.join(cfg.RESULTS_DIR, "table7_sota.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run()
