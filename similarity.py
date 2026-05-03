"""Multi-criteria similarity scoring for contrastive pair selection.

S_total = 0.4 S_sp + 0.3 S_den + 0.2 S_str + 0.1 S_pr

    S_sp:  Pearson correlation between flattened wafer maps
    S_den: exp(-20 * |density_i - density_j|)
    S_str: Cosine similarity of Hu moments + Fourier descriptors
    S_pr:  Same lot or close in time (default to 0 when unknown)
"""

import math
import numpy as np
import cv2

from config import cfg


def spatial_pattern_similarity(w1: np.ndarray, w2: np.ndarray) -> float:
    a = w1.flatten().astype(np.float32)
    b = w2.flatten().astype(np.float32)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def defect_density(wafer: np.ndarray) -> float:
    return float(np.mean(wafer == 2))


def density_proximity(w1: np.ndarray, w2: np.ndarray) -> float:
    return math.exp(-20.0 * abs(defect_density(w1) - defect_density(w2)))


def hu_moments(wafer: np.ndarray) -> np.ndarray:
    bin_img = (wafer == 2).astype(np.uint8) * 255
    moments = cv2.moments(bin_img)
    hu = cv2.HuMoments(moments).flatten()
    # Log-scale for numerical stability
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)


def fourier_descriptors(wafer: np.ndarray, n_descriptors: int = 10) -> np.ndarray:
    """Always returns a fixed-length vector of `n_descriptors` floats
    (zero-padded if the contour is shorter than n_descriptors)."""
    bin_img = (wafer == 2).astype(np.uint8)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(n_descriptors, dtype=np.float32)
    if not contours:
        return out
    contour = max(contours, key=cv2.contourArea).squeeze()
    if contour.ndim < 2 or len(contour) < 2:
        return out
    complex_pts = contour[:, 0] + 1j * contour[:, 1]
    fft = np.fft.fft(complex_pts)
    mag = np.abs(fft[:n_descriptors]).astype(np.float32)
    out[: len(mag)] = mag
    return out


def structural_similarity(w1: np.ndarray, w2: np.ndarray) -> float:
    f1 = np.concatenate([hu_moments(w1), fourier_descriptors(w1)])
    f2 = np.concatenate([hu_moments(w2), fourier_descriptors(w2)])
    n1, n2 = np.linalg.norm(f1), np.linalg.norm(f2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(f1, f2) / (n1 * n2))


def production_proximity(lot_i, lot_j, time_i=None, time_j=None) -> float:
    """1.0 if same lot or within 2-hour window, else 0.0."""
    if lot_i is not None and lot_j is not None and lot_i == lot_j:
        return 1.0
    if time_i is not None and time_j is not None:
        if abs(time_i - time_j) <= 2 * 3600:
            return 1.0
    return 0.0


def total_similarity(
    w1: np.ndarray, w2: np.ndarray,
    lot_i=None, lot_j=None, time_i=None, time_j=None,
    weights: dict = None,
) -> float:
    weights = weights or cfg.SIM_WEIGHTS
    s_sp = max(0.0, spatial_pattern_similarity(w1, w2))
    s_den = density_proximity(w1, w2)
    s_str = max(0.0, structural_similarity(w1, w2))
    s_pr = production_proximity(lot_i, lot_j, time_i, time_j)
    return (
        weights["sp"] * s_sp
        + weights["den"] * s_den
        + weights["str"] * s_str
        + weights["pr"] * s_pr
    )


def assign_pair_label(
    score: float,
    pos_thresh: float = cfg.POS_THRESHOLD,
    neg_thresh: float = cfg.NEG_THRESHOLD,
) -> int:
    """Return 1 (positive), 0 (negative), or -1 (excluded)."""
    if score > pos_thresh:
        return 1
    if score < neg_thresh:
        return 0
    return -1


def batched_total_similarity(
    wafers, lots, structurals, densities, weights=None,
) -> np.ndarray:
    """Vectorized N x N similarity matrix using pre-computed descriptors."""
    weights = weights or cfg.SIM_WEIGHTS
    n = len(wafers)

    # S_sp: Pearson correlation between flattened wafer maps
    flat = np.stack([w.flatten().astype(np.float32) for w in wafers])
    flat_centered = flat - flat.mean(axis=1, keepdims=True)
    flat_std = flat.std(axis=1, keepdims=True) + 1e-9
    flat_norm = flat_centered / flat_std
    s_sp = np.clip((flat_norm @ flat_norm.T) / flat.shape[1], 0.0, 1.0)

    # S_den: exp(-20 |dens_i - dens_j|)
    dens = np.array(densities, dtype=np.float32).reshape(-1, 1)
    s_den = np.exp(-20.0 * np.abs(dens - dens.T))

    # S_str: cosine similarity of pre-computed Hu + Fourier descriptors
    structurals = np.asarray(structurals, dtype=np.float32)
    norms = np.linalg.norm(structurals, axis=1, keepdims=True) + 1e-9
    str_norm = structurals / norms
    s_str = np.clip(str_norm @ str_norm.T, 0.0, 1.0)

    # S_pr: same lot indicator
    s_pr = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if lots[i] is not None and lots[j] is not None and lots[i] == lots[j]:
                s_pr[i, j] = 1.0

    return (
        weights["sp"] * s_sp
        + weights["den"] * s_den
        + weights["str"] * s_str
        + weights["pr"] * s_pr
    ).astype(np.float32)


def assign_pair_label_matrix(
    score_matrix: np.ndarray,
    pos_thresh: float = cfg.POS_THRESHOLD,
    neg_thresh: float = cfg.NEG_THRESHOLD,
) -> np.ndarray:
    """Vectorized version of assign_pair_label over an N x N score matrix."""
    out = np.full(score_matrix.shape, -1, dtype=np.int64)
    out[score_matrix > pos_thresh] = 1
    out[score_matrix < neg_thresh] = 0
    np.fill_diagonal(out, -1)
    return out
