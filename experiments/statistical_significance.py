"""Table 9: paired t-test over 5 seeds for key comparisons."""

import json
import os

import numpy as np
from scipy import stats
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.neural_network import MLPClassifier

from config import cfg
from data_loader import prepare_dataset
from dataset_pyg import WaferGraphDataset, collate_graphs
from evaluate import extract_embeddings
from model import GINJumpingKnowledge
from train import train_self_supervised


def _seeded_run(seed, train_df, val_df, test_df):
    torch.manual_seed(seed); np.random.seed(seed)
    cfg.SEED = seed
    ckpt = train_self_supervised(
        train_df=train_df, val_df=val_df,
        ckpt_name=f"sig_seed_{seed}.pt", epochs=50,
    )
    model = GINJumpingKnowledge().to(cfg.DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE))

    train_loader = DataLoader(WaferGraphDataset(train_df, precompute_descriptors=False), batch_size=cfg.BATCH_SIZE,
                               shuffle=False, collate_fn=collate_graphs)
    test_loader = DataLoader(WaferGraphDataset(test_df, precompute_descriptors=False), batch_size=cfg.BATCH_SIZE,
                              shuffle=False, collate_fn=collate_graphs)
    z_train, y_train = extract_embeddings(model, train_loader)
    z_test, y_test = extract_embeddings(model, test_loader)

    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                         random_state=seed, early_stopping=True).fit(z_train, y_train)
    acc = accuracy_score(y_test, clf.predict(z_test))
    ari = adjusted_rand_score(
        y_test, AgglomerativeClustering(n_clusters=cfg.NUM_CLASSES).fit_predict(z_test)
    )
    return acc, ari


def paired_t_test(a, b):
    diff = np.array(a) - np.array(b)
    t, p = stats.ttest_rel(a, b)
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    ci = (mean - 1.96 * se, mean + 1.96 * se)
    return {"delta_mean": mean, "p_value": float(p), "ci95": ci}


def run():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    train_df, val_df, test_df = prepare_dataset()

    runs = {"contrastive": [], "random_init": []}
    for seed in [0, 1, 2, 3, 4]:
        # Contrastive
        acc_c, ari_c = _seeded_run(seed, train_df, val_df, test_df)
        runs["contrastive"].append((acc_c, ari_c))

        # Random init: skip pre-training, evaluate fresh model
        torch.manual_seed(seed)
        model = GINJumpingKnowledge().to(cfg.DEVICE)
        train_loader = DataLoader(WaferGraphDataset(train_df, precompute_descriptors=False), batch_size=cfg.BATCH_SIZE,
                                   shuffle=False, collate_fn=collate_graphs)
        test_loader = DataLoader(WaferGraphDataset(test_df, precompute_descriptors=False), batch_size=cfg.BATCH_SIZE,
                                  shuffle=False, collate_fn=collate_graphs)
        z_tr, y_tr = extract_embeddings(model, train_loader)
        z_te, y_te = extract_embeddings(model, test_loader)
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                             random_state=seed, early_stopping=True).fit(z_tr, y_tr)
        acc_r = accuracy_score(y_te, clf.predict(z_te))
        ari_r = adjusted_rand_score(
            y_te, AgglomerativeClustering(n_clusters=cfg.NUM_CLASSES).fit_predict(z_te)
        )
        runs["random_init"].append((acc_r, ari_r))

    acc_c = [r[0] for r in runs["contrastive"]]
    acc_r = [r[0] for r in runs["random_init"]]
    results = {
        "contrastive_vs_random_init_accuracy": paired_t_test(acc_c, acc_r),
        "raw": runs,
    }
    with open(os.path.join(cfg.RESULTS_DIR, "table9_significance.json"), "w") as f:
        json.dump(results, f, indent=2, default=lambda o: list(o) if hasattr(o, "__iter__") else str(o))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    run()
