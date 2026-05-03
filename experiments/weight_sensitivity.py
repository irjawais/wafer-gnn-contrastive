"""Table 11: criteria-weight sensitivity sweep."""

import os
import json

from config import cfg
from data_loader import prepare_dataset
from train import train_self_supervised


WEIGHT_GRID = [
    {"sp": 0.4, "den": 0.3, "str": 0.2, "pr": 0.1},   # Ours
    {"sp": 0.25, "den": 0.25, "str": 0.25, "pr": 0.25},  # Equal
    {"sp": 0.5, "den": 0.2, "str": 0.2, "pr": 0.1},
    {"sp": 0.3, "den": 0.4, "str": 0.2, "pr": 0.1},
    {"sp": 0.4, "den": 0.3, "str": 0.3, "pr": 0.0},   # No production proximity
]


def run():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    train_df, val_df, _ = prepare_dataset()
    results = {}
    for i, w in enumerate(WEIGHT_GRID):
        cfg.SIM_WEIGHTS = w
        ckpt = train_self_supervised(
            train_df=train_df, val_df=val_df,
            ckpt_name=f"weights_{i}.pt", epochs=50,
        )
        results[str(w)] = {"checkpoint": ckpt}
    with open(os.path.join(cfg.RESULTS_DIR, "table11_weights.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run()
