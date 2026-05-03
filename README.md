# Graph-Based Contrastive Learning for Self-Supervised Wafer Defect Detection

Implementation of a self-supervised approach combining Graph Neural Networks with multi-criteria contrastive learning for semiconductor wafer defect detection on the WM-811K dataset.

---

## Overview

The pipeline transforms wafer maps into graph representations and learns embeddings without manual labels using domain-driven contrastive pairs. Training proceeds in two stages:

1. **Self-supervised pre-training** — The GIN encoder is trained with a margin-based contrastive loss whose positive/negative pairs are derived from a multi-criteria similarity score (spatial pattern, defect density, structural descriptors, production context).
2. **Supervised fine-tuning** — A classifier head is added on top of the pre-trained backbone and fine-tuned with cross-entropy on labeled data.

Two evaluation paths are supported:

- **Path A (clustering on frozen embeddings)** — measures the unsupervised quality of the learned representation (K-means / Agglomerative / DBSCAN with Silhouette / ARI / NMI).
- **Path B (in-model classifier on fine-tuned backbone)** — measures end-to-end classification accuracy after supervised fine-tuning.

```
WM-811K (LSWMD.pkl)
   |
   v
data_loader.py        --> resize, stratified sample, 60/20/20 split, augmentation
   |
   v
graph_construction.py --> 8-connectivity graph + 9-d node features
   |
   v
similarity.py         --> S_total = 0.4 S_sp + 0.3 S_den + 0.2 S_str + 0.1 S_pr
   |
   v
contrastive_loss.py   --> margin-based loss with class-balance weights
   |
   v
model.py              --> GIN (4 layers, 64-128-256-128) + Jumping Knowledge
   |
   v
train.py              --> SSL pre-training + supervised fine-tuning
   |
   v
evaluate.py           --> clustering metrics + classification metrics
```

---

## Repository Structure

```
.
├── config.py                 # Global hyperparameters
├── data_loader.py            # WM-811K loading, resize, augmentation, splits
├── graph_construction.py     # Wafer -> graph (8-connectivity + edge weights)
├── similarity.py             # Multi-criteria similarity (vectorized N x N)
├── contrastive_loss.py       # Margin-based contrastive loss
├── model.py                  # GIN + Jumping Knowledge backbone
├── dataset_pyg.py            # PyG Dataset wrapper (worker-safe RNG)
├── train.py                  # SSL pre-training + supervised fine-tuning
├── evaluate.py               # Clustering + classification evaluation
├── main.py                   # CLI entry point
├── requirements.txt          # Python dependencies
├── baselines/                # CNN, ResNet-50, Autoencoder baselines
└── experiments/              # One script per experiment
```

---

## Installation

### Prerequisites
- Python 3.8+
- (Recommended) NVIDIA GPU with CUDA 11.x for training

### Setup

```bash
git clone https://github.com/irjawais/wafer-gnn-contrastive.git
cd wafer-gnn-contrastive
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset

Download the WM-811K dataset (`LSWMD.pkl`) from the [Kaggle mirror](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) and place it at:

```
../MIR-WM811K/Python/LSWMD.pkl
```

Or update `cfg.DATA_PATH` in `config.py`.

---

## Usage

```bash
# Full pipeline: pre-train -> fine-tune -> evaluate (both paths)
python main.py --mode all

# Self-supervised pre-training only
python main.py --mode pretrain

# Supervised fine-tuning of pre-trained backbone
python main.py --mode finetune

# Path A: clustering + sklearn MLP on frozen pre-trained embeddings
python main.py --mode eval

# Path B: in-model classifier from fine-tuned checkpoint
python main.py --mode eval-finetuned

# Run every experiment in sequence
python -m experiments.run_all
```

Checkpoints are written to `checkpoints/` (e.g. `gin_pretrained.pt`, `gin_finetuned.pt`). Numerical results are written as JSON to `results/`. Figures are written as PNG to `results/`.

---

## Experiments

Each script under `experiments/` is self-contained and produces a JSON result file (and a PNG when applicable) in `results/`. Outputs are deterministic given the configured seed.

| Script | What it produces |
|--------|------------------|
| `experiments/pretrain_main.py` | Self-supervised pre-trained checkpoint |
| `experiments/feature_ablation.py` | Node-feature ablation (position-only / +spatial / +contextual) |
| `experiments/per_class_f1.py` | Per-class F1 with 5 random seeds (mean +/- std) |
| `experiments/run_baselines.py` | CNN / ResNet-50 / Autoencoder baselines under identical conditions |
| `experiments/sota_comparison.py` | Fine-tuned classification + clustering for SOTA comparison |
| `experiments/objective_and_graph_ablation.py` | Learning-objective and graph-construction ablations |
| `experiments/statistical_significance.py` | Paired t-test over 5 seeds with 95% CIs |
| `experiments/threshold_sensitivity.py` | Sweep over positive / negative pair thresholds |
| `experiments/weight_sensitivity.py` | Sweep over multi-criteria similarity weights |
| `experiments/computational_analysis.py` | Inference latency, parameter count, memory footprint |
| `experiments/low_label_experiment.py` | Accuracy when only 25% of labels are used |
| `experiments/robustness.py` | Accuracy under Gaussian feature noise |
| `experiments/ood_detection.py` | Out-of-distribution detection via held-out class |
| `experiments/embedding_visualization.py` | PCA + t-SNE figure |
| `experiments/confusion_matrix_plot.py` | Confusion matrix figure |

---

## Hyperparameters

All hyperparameters live in `config.py`.

| Parameter | Value |
|-----------|-------|
| Wafer size | 25 x 25 |
| Connectivity | 8 |
| Node feature dim | 9 |
| GIN hidden dims | 64 -> 128 -> 256 -> 128 |
| Projection dim | 128 |
| Dropout | 0.2 |
| Margin (contrastive) | 2.0 |
| Positive / negative threshold | 0.6 / 0.3 |
| Similarity weights (sp, den, str, pr) | (0.4, 0.3, 0.2, 0.1) |
| Optimizer | Adam (lr=1e-3, wd=1e-5) |
| Fine-tune learning rate | 1e-4 |
| Batch size | 64 |
| Epochs | 200 |
| Patience (early stopping) | 20 |
| Min-defective fraction per batch | 0.30 |
| Random seed | 42 |

---

## Implementation Notes

- **Stratified sampling.** `StratifiedDefectSampler` (in `train.py`) enforces that at least 30% of every batch is defective, addressing the 91.5% / 8.5% class imbalance in WM-811K.
- **Vectorized similarity.** Hu moments and Fourier descriptors are pre-computed once per wafer in `dataset_pyg.py`. The N x N similarity matrix used to construct contrastive pairs is built with NumPy operations rather than per-pair Python loops.
- **Worker-safe augmentation RNG.** `make_worker_init_fn` re-seeds each DataLoader worker independently so parallel workers do not produce duplicate augmentations.
- **Augmentation-aware descriptors.** When `augment=True`, structural descriptors and density are recomputed on the augmented wafer rather than reusing cached values, ensuring consistency between the graph fed to the model and the descriptors used in similarity scoring.
- **Two-stage training is exposed at the CLI.** `main.py` provides explicit `pretrain`, `finetune`, `eval`, and `eval-finetuned` modes, so reviewers can verify each stage independently.

---

## Dataset Encoding

WM-811K wafer maps use three integer values per die:

| Value | Meaning |
|-------|---------|
| 0 | Non-existent die area (outside the wafer) |
| 1 | Normal die |
| 2 | Defective die |

Defect classes (9): None, Center, Donut, Edge-Loc, Edge-Ring, Local, Near-full, Random, Scratch. The dataset stores `Loc` for the `Local` class; the codebase preserves this mapping in `data_loader.LABEL_MAP`.

---

## What's Included

- [x] Fixed random seed in `config.py` (default: 42)
- [x] Deterministic seeding propagated through `main.py`, `train.py`, and `evaluate.py`
- [x] Worker-safe RNG for DataLoader augmentation
- [x] Pre-trained checkpoint can be loaded independently of training code
- [x] Each table / figure is regeneratable from a single script
- [x] All baselines are implemented locally rather than relying on external repositories
- [x] Statistical significance computed via paired t-test over 5 seeds (`experiments/statistical_significance.py`)
