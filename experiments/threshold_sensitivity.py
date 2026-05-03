"""Table 10: similarity-threshold sensitivity sweep."""

import os
import json

from config import cfg
from data_loader import prepare_dataset
from train import train_self_supervised


GRID = [
    (0.5, 0.3), (0.6, 0.2), (0.6, 0.3), (0.6, 0.4), (0.7, 0.3),
]


def run():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    train_df, val_df, _ = prepare_dataset()
    results = {}
    for pos, neg in GRID:
        cfg.POS_THRESHOLD = pos
        cfg.NEG_THRESHOLD = neg
        ckpt = train_self_supervised(
            train_df=train_df, val_df=val_df,
            ckpt_name=f"thr_{pos:.1f}_{neg:.1f}.pt",
            epochs=50,
        )
        results[f"pos={pos}, neg={neg}"] = {"checkpoint": ckpt}

    with open(os.path.join(cfg.RESULTS_DIR, "table10_thresholds.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run()
