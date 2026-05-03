"""MixedWM38 dataset adapter.

Source: Wang et al. (2020), IEEE TSM. https://github.com/Junliangwangdhu/WaferMap

Encoding (matches WM-811K convention used in this repository):
    0 = blank spot (outside wafer)
    1 = normal die
    2 = defective die

Labels are multi-hot 8-bit (one bit per single-type defect: C, D, EL, ER, L, NF,
R, S). The unique multi-hot patterns observed in the dataset form 38 classes
(1 normal + 8 single + 13 two-mixed + 12 three-mixed + 4 four-mixed).

This loader collapses the multi-hot representation into a single 38-way
categorical label so the existing CrossEntropyLoss head in `model.py` works
unchanged.
"""

import os

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import cfg


# Single-type defect bit positions (matches the 8 columns of arr_1)
SINGLE_TYPE_NAMES = ["C", "D", "EL", "ER", "L", "NF", "R", "S"]


def _multihot_key(vec: np.ndarray) -> str:
    """Stable string key for a multi-hot label, e.g. '00000000' or '10010001'."""
    return "".join(str(int(b)) for b in vec.tolist())


def load_mixedwm38(npz_path: str) -> pd.DataFrame:
    """Load the MixedWM38 .npz file into a DataFrame compatible with our pipeline.

    Returns a DataFrame with columns:
        wafer_map_resized: int64 array (TARGET x TARGET) per row
        label:             int (0..K-1) categorical label
        label_str:         string multi-hot key (e.g. '00100000' or '10010001')
        lotName:           None (MixedWM38 does not record lot IDs)
    """
    data = np.load(npz_path)
    wafers = data["arr_0"]            # (N, 52, 52)
    multihot = data["arr_1"]          # (N, 8)

    # Build categorical mapping from multi-hot patterns to integer labels
    keys = [_multihot_key(v) for v in multihot]
    unique_keys = sorted(set(keys))
    key_to_idx = {k: i for i, k in enumerate(unique_keys)}

    target = cfg.WAFER_SIZE
    rows = []
    for w, k in zip(wafers, keys):
        w_resized = cv2.resize(
            w.astype(np.uint8), (target, target), interpolation=cv2.INTER_NEAREST
        ).astype(np.int64)
        rows.append({
            "wafer_map_resized": w_resized,
            "label": key_to_idx[k],
            "label_str": k,
            "lotName": None,
        })

    df = pd.DataFrame(rows)
    df.attrs["key_to_idx"] = key_to_idx
    df.attrs["num_classes"] = len(unique_keys)
    df.attrs["class_names"] = _build_class_names(unique_keys)
    return df


def _build_class_names(keys):
    """Human-readable class names like 'C+L+EL+S' from multi-hot keys."""
    names = []
    for k in keys:
        bits = [SINGLE_TYPE_NAMES[i] for i, c in enumerate(k) if c == "1"]
        names.append("+".join(bits) if bits else "Normal")
    return names


def split_dataset(df: pd.DataFrame):
    """60/20/20 split stratified by collapsed label."""
    train_df, temp_df = train_test_split(
        df, test_size=cfg.VAL_SPLIT + cfg.TEST_SPLIT,
        stratify=df["label"], random_state=cfg.SEED,
    )
    rel = cfg.TEST_SPLIT / (cfg.VAL_SPLIT + cfg.TEST_SPLIT)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel, stratify=temp_df["label"], random_state=cfg.SEED,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def prepare_mixedwm38(npz_path: str = None):
    """End-to-end loader: returns (train_df, val_df, test_df, class_names)."""
    if npz_path is None:
        npz_path = os.environ.get(
            "MIXEDWM38_PATH",
            "../MixedWM38/Wafer_Map_Datasets.npz",
        )
    df = load_mixedwm38(npz_path)
    class_names = df.attrs["class_names"]
    num_classes = df.attrs["num_classes"]
    train_df, val_df, test_df = split_dataset(df)
    return train_df, val_df, test_df, class_names, num_classes


def four_mixed_class_indices(class_names):
    """Return the integer label indices for four-mixed defect classes."""
    return [i for i, name in enumerate(class_names) if name.count("+") == 3]
