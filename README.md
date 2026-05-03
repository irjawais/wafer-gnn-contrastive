# Graph-Based Contrastive Learning for Self-Supervised Wafer Defect Detection

Implementation of a self-supervised approach combining Graph Neural Networks with multi-criteria contrastive learning for semiconductor wafer defect detection on the WM-811K dataset.

---

## Repository Structure

```
.
├── config.py                 # Global hyperparameters
├── data_loader.py            # WM-811K loading, resize, augmentation, splits
├── graph_construction.py     # Wafer → graph (8-connectivity + edge weights)
├── similarity.py             # Multi-criteria similarity (vectorized N×N)
├── contrastive_loss.py       # Margin-based contrastive loss
├── model.py                  # GIN + Jumping Knowledge backbone
├── dataset_pyg.py            # PyG Dataset wrapper
├── train.py                  # Self-supervised pre-training + fine-tuning
├── evaluate.py               # Clustering + classification evaluation
├── main.py                   # CLI entry point
├── baselines/                # CNN, ResNet-50, Autoencoder baselines
└── experiments/              # One script per paper table / figure
```

---

## Installation

```bash
git clone https://github.com/irjawais/wafer-gnn-contrastive.git
cd wafer-gnn-contrastive
pip install -r requirements.txt
```

### Dataset

Download the WM-811K dataset and place `LSWMD.pkl` at `../MIR-WM811K/Python/LSWMD.pkl`, or update `cfg.DATA_PATH` in `config.py`.

---

## Usage

```bash
# Full pipeline: pre-train → fine-tune → evaluate
python main.py --mode all

# Self-supervised pre-training only
python main.py --mode pretrain

# Supervised fine-tuning of pre-trained backbone
python main.py --mode finetune

# Evaluate frozen pre-trained embeddings (clustering)
python main.py --mode eval

# Evaluate fine-tuned in-model classifier
python main.py --mode eval-finetuned

# Regenerate every paper table and figure
python -m experiments.run_all
```

---

## Reproducing Paper Artifacts

| Paper artifact | Script |
|----------------|--------|
| Table 4 — Feature ablation | `experiments/feature_ablation.py` |
| Table 5 — Per-class F1 | `experiments/per_class_f1.py` |
| Table 6 — Computational comparison | `experiments/run_baselines.py` |
| Table 7 — SOTA comparison | `experiments/sota_comparison.py` |
| Table 8 — Objective + graph ablation | `experiments/objective_and_graph_ablation.py` |
| Table 9 — Statistical significance | `experiments/statistical_significance.py` |
| Table 10 — Threshold sensitivity | `experiments/threshold_sensitivity.py` |
| Table 11 — Criteria-weight sensitivity | `experiments/weight_sensitivity.py` |
| Table 12 — Inference / FLOPs / memory | `experiments/computational_analysis.py` |
| 25%-labeled experiment | `experiments/low_label_experiment.py` |
| Robustness (Gaussian noise) | `experiments/robustness.py` |
| OOD detection | `experiments/ood_detection.py` |
| Figs 15–16 — PCA / t-SNE | `experiments/embedding_visualization.py` |
| Fig 17 — Confusion matrix | `experiments/confusion_matrix_plot.py` |

---

## Hyperparameters

All hyperparameters live in `config.py`.

| Parameter | Value |
|-----------|-------|
| Wafer size | 25 × 25 |
| Connectivity | 8 |
| Node feature dim | 9 |
| GIN hidden dims | 64 → 128 → 256 → 128 |
| Projection dim | 128 |
| Dropout | 0.2 |
| Margin (contrastive) | 2.0 |
| Positive / negative threshold | 0.6 / 0.3 |
| Similarity weights (sp, den, str, pr) | (0.4, 0.3, 0.2, 0.1) |
| Optimizer | Adam (lr=1e-3, wd=1e-5) |
| Batch size | 64 |
| Epochs | 200 |
| Patience | 20 |
| Min-defective fraction per batch | 0.30 |
| Random seed | 42 |
