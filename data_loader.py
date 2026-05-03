"""WM-811K dataset loading and preprocessing.

Encoding (per dataset documentation):
    0 = non-existent die area
    1 = normal die
    2 = defective die
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import cfg


# Maps the WM-811K failureType strings to integer labels.
# The paper text uses "Local"; the dataset stores "Loc".
LABEL_MAP = {
    "none": 0, "Center": 1, "Donut": 2, "Edge-Loc": 3,
    "Edge-Ring": 4, "Loc": 5, "Near-full": 6, "Random": 7, "Scratch": 8,
}


def load_wm811k(path: str = cfg.DATA_PATH) -> pd.DataFrame:
    """Load the WM-811K dataset (LSWMD pickle)."""
    df = pd.read_pickle(path)
    df = df[df["failureType"].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)]
    df["label_str"] = df["failureType"].apply(
        lambda x: x[0][0] if isinstance(x, np.ndarray) and x.size > 0 else "none"
    )
    df = df[df["label_str"].isin(LABEL_MAP.keys())]
    df["label"] = df["label_str"].map(LABEL_MAP)
    return df.reset_index(drop=True)


def resize_wafer(wafer: np.ndarray, target: int = cfg.WAFER_SIZE) -> np.ndarray:
    """Standardize wafer maps to a fixed grid via OpenCV nearest-neighbor."""
    return cv2.resize(
        wafer.astype(np.uint8), (target, target), interpolation=cv2.INTER_NEAREST
    ).astype(np.int64)


def augment_wafer(wafer: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random rotation (90/180/270) and H/V flips per Section 4.1."""
    k = int(rng.integers(0, 4))
    if k > 0:
        wafer = np.rot90(wafer, k=k)
    if rng.random() < 0.5:
        wafer = np.fliplr(wafer)
    if rng.random() < 0.5:
        wafer = np.flipud(wafer)
    return np.ascontiguousarray(wafer)


def stratified_sample(df: pd.DataFrame, frac: float = cfg.SAMPLE_FRAC) -> pd.DataFrame:
    """Stratified sampling preserving the defect-class distribution."""
    return df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=cfg.SEED)
    ).reset_index(drop=True)


def split_dataset(df: pd.DataFrame):
    """60/20/20 split stratified by label."""
    train_df, temp_df = train_test_split(
        df, test_size=cfg.VAL_SPLIT + cfg.TEST_SPLIT,
        stratify=df["label"], random_state=cfg.SEED,
    )
    rel = cfg.TEST_SPLIT / (cfg.VAL_SPLIT + cfg.TEST_SPLIT)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel, stratify=temp_df["label"], random_state=cfg.SEED,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_dataset():
    """End-to-end loading: load, resize, sample, split."""
    df = load_wm811k()
    df["wafer_map_resized"] = df["waferMap"].apply(resize_wafer)
    df = stratified_sample(df)
    return split_dataset(df)
