"""Self-supervised pre-training of GIN with multi-criteria contrastive loss."""

import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from config import cfg
from contrastive_loss import ContrastiveLoss, class_balance_weights
from data_loader import prepare_dataset
from dataset_pyg import WaferGraphDataset, collate_graphs, make_worker_init_fn
from model import GINJumpingKnowledge
from similarity import assign_pair_label_matrix, batched_total_similarity


class StratifiedDefectSampler(Sampler):
    """Each batch contains at least `min_defective_frac` defective samples.

    Note: __len__ returns num_batches * batch_size; with drop_last=True on the
    DataLoader this matches actual yielded count exactly.
    """

    def __init__(self, labels, batch_size, min_defective_frac, seed=0, num_batches=None):
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.min_def = max(1, int(round(batch_size * min_defective_frac)))
        self.normal_idx = np.where(self.labels == 0)[0]
        self.defective_idx = np.where(self.labels != 0)[0]
        if len(self.defective_idx) == 0 or len(self.normal_idx) == 0:
            raise ValueError(
                "StratifiedDefectSampler requires both normal (label==0) and "
                "defective (label!=0) samples in the training set."
            )
        self.rng = np.random.default_rng(seed)
        self.num_batches = num_batches or (len(labels) // batch_size)

    def __iter__(self):
        for _ in range(self.num_batches):
            n_def = self.min_def
            n_norm = self.batch_size - n_def
            def_pick = self.rng.choice(self.defective_idx, size=n_def, replace=True)
            norm_pick = self.rng.choice(self.normal_idx, size=n_norm, replace=True)
            batch = np.concatenate([def_pick, norm_pick])
            self.rng.shuffle(batch)
            for idx in batch:
                yield int(idx)

    def __len__(self):
        return self.num_batches * self.batch_size


def _train_one_epoch(model, loader, criterion, optimizer, class_freq):
    model.train()
    running = 0.0
    for graphs, labels, wafers, lots, structurals, densities in tqdm(loader, leave=False):
        graphs = graphs.to(cfg.DEVICE)
        labels = labels.to(cfg.DEVICE)
        score_matrix = batched_total_similarity(wafers, lots, structurals, densities)
        pair_labels = torch.from_numpy(assign_pair_label_matrix(score_matrix)).to(cfg.DEVICE)
        weights = class_balance_weights(class_freq, labels).to(cfg.DEVICE)

        embeddings = model.encode(graphs.x, graphs.edge_index, graphs.batch)
        loss = criterion(embeddings, pair_labels, weights=weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / max(1, len(loader))


@torch.no_grad()
def _validate(model, loader, criterion):
    model.eval()
    running = 0.0
    for graphs, labels, wafers, lots, structurals, densities in loader:
        graphs = graphs.to(cfg.DEVICE)
        score_matrix = batched_total_similarity(wafers, lots, structurals, densities)
        pair_labels = torch.from_numpy(assign_pair_label_matrix(score_matrix)).to(cfg.DEVICE)
        embeddings = model.encode(graphs.x, graphs.edge_index, graphs.batch)
        running += criterion(embeddings, pair_labels).item()
    return running / max(1, len(loader))


def train_self_supervised(
    train_df=None, val_df=None, ckpt_name="gin_pretrained.pt",
    epochs=None, augment=True,
):
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    if train_df is None or val_df is None:
        train_df, val_df, _ = prepare_dataset()
    print(f"Train/Val sizes: {len(train_df)}/{len(val_df)}")

    train_set = WaferGraphDataset(train_df, augment=augment, seed=cfg.SEED)
    val_set = WaferGraphDataset(val_df, augment=False, seed=cfg.SEED + 1)

    sampler = StratifiedDefectSampler(
        labels=train_df["label"].tolist(),
        batch_size=cfg.BATCH_SIZE,
        min_defective_frac=cfg.MIN_DEFECTIVE_FRAC,
        seed=cfg.SEED,
    )
    worker_init = make_worker_init_fn(cfg.SEED)
    train_loader = DataLoader(
        train_set, batch_size=cfg.BATCH_SIZE, sampler=sampler,
        collate_fn=collate_graphs, num_workers=0, drop_last=True,
        worker_init_fn=worker_init,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.BATCH_SIZE, shuffle=False,
        collate_fn=collate_graphs, num_workers=0,
        worker_init_fn=worker_init,
    )

    model = GINJumpingKnowledge().to(cfg.DEVICE)
    criterion = ContrastiveLoss(margin=cfg.MARGIN)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY,
    )

    class_freq = {k: v / len(train_df) for k, v in Counter(train_df["label"]).items()}

    best_val = float("inf")
    no_improve = 0
    n_epochs = epochs or cfg.EPOCHS
    for epoch in range(n_epochs):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, class_freq)
        val_loss = _validate(model, val_loader, criterion)
        print(f"[Pre-train] Epoch {epoch+1}/{n_epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, ckpt_name))
        else:
            no_improve += 1
            if no_improve >= cfg.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Self-supervised pre-training complete.")
    return os.path.join(cfg.CHECKPOINT_DIR, ckpt_name)


def fine_tune_classifier(
    pretrained_ckpt: str,
    train_df, val_df,
    epochs: int = 50,
    lr: float = 1e-4,  # 10x lower than pre-train LR to preserve learned features
    out_ckpt: str = "gin_finetuned.pt",
):
    """Supervised fine-tuning of the GIN backbone + classifier head.

    The fine-tune learning rate is set to 1e-4 (10x lower than the
    self-supervised pre-training lr) to avoid overwriting the contrastive
    representations learned in stage 1. This is a standard transfer-learning
    practice and is not specified in the paper.
    """
    torch.manual_seed(cfg.SEED)
    train_set = WaferGraphDataset(train_df, augment=True, seed=cfg.SEED)
    val_set = WaferGraphDataset(val_df, augment=False, seed=cfg.SEED + 1,
                                 precompute_descriptors=False)
    worker_init = make_worker_init_fn(cfg.SEED)
    train_loader = DataLoader(
        train_set, batch_size=cfg.BATCH_SIZE, shuffle=True,
        collate_fn=collate_graphs, num_workers=0, worker_init_fn=worker_init,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.BATCH_SIZE, shuffle=False,
        collate_fn=collate_graphs, num_workers=0, worker_init_fn=worker_init,
    )

    model = GINJumpingKnowledge().to(cfg.DEVICE)
    if pretrained_ckpt and os.path.exists(pretrained_ckpt):
        model.load_state_dict(torch.load(pretrained_ckpt, map_location=cfg.DEVICE))
        print(f"Loaded pre-trained weights from {pretrained_ckpt}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    ce = torch.nn.CrossEntropyLoss()

    best = 0.0
    out_path = os.path.join(cfg.CHECKPOINT_DIR, out_ckpt)
    for epoch in range(epochs):
        model.train()
        for graphs, labels, *_ in train_loader:
            graphs = graphs.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)
            _, logits = model(graphs.x, graphs.edge_index, graphs.batch)
            loss = ce(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for graphs, labels, *_ in val_loader:
                graphs = graphs.to(cfg.DEVICE)
                labels = labels.to(cfg.DEVICE)
                _, logits = model(graphs.x, graphs.edge_index, graphs.batch)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        acc = correct / max(1, total)
        print(f"[Fine-tune] Epoch {epoch+1}/{epochs}  val_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), out_path)
    print(f"Fine-tuning complete. Best val acc: {best:.4f}")
    return out_path


if __name__ == "__main__":
    train_self_supervised()
