"""Transferability experiment on MixedWM38.

Runs the same self-supervised pre-training -> supervised fine-tuning pipeline
on the MixedWM38 dataset (Wang et al. 2020) to evaluate whether the method
generalizes beyond WM-811K.

Reports:
    - Overall accuracy and macro-F1 on 38-way classification
    - Macro-F1 on the 4-mixed-defect subset (the hardest classes; Wang's
      original Deformable CNN reported only 88.05% here)
    - ARI on frozen pre-trained embeddings (clustering quality)

Usage:
    export MIXEDWM38_PATH=/path/to/Wafer_Map_Datasets.npz
    python -m experiments.transferability_mixedwm38

Or:
    python -m experiments.transferability_mixedwm38 --npz /path/to/file.npz
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, f1_score,
    normalized_mutual_info_score,
)

from config import cfg
from data_loader_mixedwm38 import (
    four_mixed_class_indices, prepare_mixedwm38,
)
from dataset_pyg import WaferGraphDataset, collate_graphs
from model import GINJumpingKnowledge
from train import fine_tune_classifier, train_self_supervised


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=None,
                   help="Path to MixedWM38 .npz (else uses MIXEDWM38_PATH env).")
    p.add_argument("--pretrain-epochs", type=int, default=60,
                   help="Self-supervised pre-training epochs (default: 60).")
    p.add_argument("--finetune-epochs", type=int, default=30,
                   help="Supervised fine-tuning epochs (default: 30).")
    p.add_argument("--skip-pretrain", action="store_true",
                   help="Reuse existing checkpoint instead of pre-training.")
    p.add_argument("--subsample", type=float, default=1.0,
                   help="Stratified subsample fraction (default: 1.0 = full dataset).")
    return p.parse_args()


def _eval_test(model, test_df, class_names, four_mixed_idx):
    """Evaluate the fine-tuned model on the test set."""
    test_set = WaferGraphDataset(test_df, augment=False, precompute_descriptors=False)
    loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        collate_fn=collate_graphs)

    embeddings, preds, ys = [], [], []
    model.eval()
    with torch.no_grad():
        for graphs, y, *_ in loader:
            graphs = graphs.to(cfg.DEVICE)
            z, logits = model(graphs.x, graphs.edge_index, graphs.batch)
            embeddings.append(z.cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
            ys.append(y.numpy())
    embeddings = np.concatenate(embeddings)
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)

    # Overall metrics
    acc = float(accuracy_score(ys, preds))
    macro_f1 = float(f1_score(ys, preds, average="macro", zero_division=0))

    # 4-mixed subset metrics (the hardest classes — Wang 2020 baseline: 88.05%)
    four_mask = np.isin(ys, four_mixed_idx)
    if four_mask.any():
        four_acc = float(accuracy_score(ys[four_mask], preds[four_mask]))
        four_f1 = float(f1_score(
            ys[four_mask], preds[four_mask],
            labels=four_mixed_idx, average="macro", zero_division=0,
        ))
    else:
        four_acc = float("nan")
        four_f1 = float("nan")

    # Clustering on frozen fine-tuned embeddings
    n_clusters = len(class_names)
    cluster_pred = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeddings)
    ari = float(adjusted_rand_score(ys, cluster_pred))
    nmi = float(normalized_mutual_info_score(ys, cluster_pred))

    return {
        "num_classes": n_clusters,
        "test_size": int(len(ys)),
        "overall_accuracy": acc,
        "overall_macro_f1": macro_f1,
        "four_mixed_accuracy": four_acc,
        "four_mixed_macro_f1": four_f1,
        "four_mixed_class_count": len(four_mixed_idx),
        "clustering_ari_finetuned": ari,
        "clustering_nmi_finetuned": nmi,
        "per_class_f1": f1_score(ys, preds, average=None, zero_division=0).tolist(),
        "class_names": class_names,
        "wang_2020_baseline_overall": 0.9320,
        "wang_2020_baseline_four_mixed": 0.8805,
    }


def run():
    args = parse_args()
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    print("Loading MixedWM38...")
    train_df, val_df, test_df, class_names, num_classes = prepare_mixedwm38(args.npz)
    four_mixed_idx = four_mixed_class_indices(class_names)

    if args.subsample < 1.0:
        rng = np.random.default_rng(cfg.SEED)

        def _strat_subsample(df, frac):
            keep = []
            for label_value in df["label"].unique():
                grp = df[df["label"] == label_value]
                n_keep = max(1, int(round(len(grp) * frac)))
                idx = rng.choice(grp.index.values, size=n_keep, replace=False)
                keep.extend(idx)
            return df.loc[keep].reset_index(drop=True)

        train_df = _strat_subsample(train_df, args.subsample)
        val_df = _strat_subsample(val_df, args.subsample)
        # Keep the test set full so reported numbers are statistically meaningful
    print(f"  Train/Val/Test = {len(train_df)}/{len(val_df)}/{len(test_df)}  "
          f"({num_classes} classes; {len(four_mixed_idx)} four-mixed classes)")

    # Override config for this dataset
    cfg.NUM_CLASSES = num_classes
    cfg.CLASS_NAMES = class_names
    # Production proximity is unavailable -> reweight remaining criteria
    cfg.SIM_WEIGHTS = {"sp": 0.45, "den": 0.33, "str": 0.22, "pr": 0.0}

    pretrained_ckpt = os.path.join(cfg.CHECKPOINT_DIR, "gin_pretrained_mixedwm38.pt")
    finetuned_ckpt = "gin_finetuned_mixedwm38.pt"

    if not args.skip_pretrain or not os.path.exists(pretrained_ckpt):
        print("\n=== Stage 1: Self-supervised pre-training on MixedWM38 ===")
        train_self_supervised(
            train_df=train_df, val_df=val_df,
            ckpt_name="gin_pretrained_mixedwm38.pt",
            epochs=args.pretrain_epochs, augment=True,
        )

    print("\n=== Stage 2: Supervised fine-tuning on MixedWM38 ===")
    finetuned_path = fine_tune_classifier(
        pretrained_ckpt=pretrained_ckpt,
        train_df=train_df, val_df=val_df,
        epochs=args.finetune_epochs, lr=1e-4,
        out_ckpt=finetuned_ckpt,
    )

    print("\n=== Stage 3: Evaluation ===")
    model = GINJumpingKnowledge().to(cfg.DEVICE)
    model.load_state_dict(torch.load(finetuned_path, map_location=cfg.DEVICE))
    metrics = _eval_test(model, test_df, class_names, four_mixed_idx)

    print(f"\nOverall accuracy:        {metrics['overall_accuracy']*100:.2f}%  "
          f"(Wang 2020 baseline: 93.20%)")
    print(f"Overall macro-F1:        {metrics['overall_macro_f1']*100:.2f}%")
    print(f"4-mixed accuracy:        {metrics['four_mixed_accuracy']*100:.2f}%  "
          f"(Wang 2020 baseline: 88.05%)")
    print(f"4-mixed macro-F1:        {metrics['four_mixed_macro_f1']*100:.2f}%")
    print(f"Clustering ARI:          {metrics['clustering_ari_finetuned']:.3f}")
    print(f"Clustering NMI:          {metrics['clustering_nmi_finetuned']:.3f}")

    out_path = os.path.join(cfg.RESULTS_DIR, "transferability_mixedwm38.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    run()
