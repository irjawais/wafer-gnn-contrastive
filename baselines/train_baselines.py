"""Train CNN, ResNet-50, and Autoencoder baselines under identical conditions."""

import os
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from config import cfg
from data_loader import prepare_dataset
from baselines.cnn_nakazawa import TraditionalCNN
from baselines.resnet50 import ResNet50Wafer
from baselines.autoencoder import WaferAutoencoder


class WaferImageDataset(Dataset):
    """Wafer maps as image tensors (1 x 25 x 25)."""

    def __init__(self, df):
        self.wafers = df["wafer_map_resized"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.wafers)

    def __getitem__(self, idx):
        x = torch.tensor(self.wafers[idx], dtype=torch.float32).unsqueeze(0) / 2.0
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def train_supervised(model, train_loader, val_loader, epochs=30, lr=1e-3):
    model = model.to(cfg.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            logits = model(x)
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        # validation
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for x, y in val_loader:
                preds.extend(model(x.to(cfg.DEVICE)).argmax(1).cpu().tolist())
                ys.extend(y.tolist())
        acc = accuracy_score(ys, preds)
        if acc > best: best = acc
    return best


def train_autoencoder(model, train_loader, epochs=30, lr=1e-3):
    model = model.to(cfg.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        for x, _ in train_loader:
            x = x.to(cfg.DEVICE)
            x_hat, _ = model(x)
            x_hat = nn.functional.interpolate(x_hat, size=x.shape[-2:], mode="nearest")
            loss = mse(x_hat, x)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def autoencoder_classify(model, train_loader, test_loader):
    model.eval()
    z_tr, y_tr, z_te, y_te = [], [], [], []
    with torch.no_grad():
        for x, y in train_loader:
            _, z = model(x.to(cfg.DEVICE))
            z_tr.append(z.cpu().numpy()); y_tr.extend(y.tolist())
        for x, y in test_loader:
            _, z = model(x.to(cfg.DEVICE))
            z_te.append(z.cpu().numpy()); y_te.extend(y.tolist())
    z_tr = np.concatenate(z_tr); z_te = np.concatenate(z_te)
    clf = LogisticRegression(max_iter=500).fit(z_tr, y_tr)
    return accuracy_score(y_te, clf.predict(z_te))


def measure_inference(model, loader):
    model.eval()
    times = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(cfg.DEVICE)
            t0 = time.perf_counter()
            _ = model(x)
            if cfg.DEVICE.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0 / x.size(0))
    return float(np.mean(times))


def run():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    train_df, val_df, test_df = prepare_dataset()
    train_loader = DataLoader(WaferImageDataset(train_df), batch_size=cfg.BATCH_SIZE,
                               shuffle=True, num_workers=2)
    val_loader = DataLoader(WaferImageDataset(val_df), batch_size=cfg.BATCH_SIZE,
                             shuffle=False, num_workers=2)
    test_loader = DataLoader(WaferImageDataset(test_df), batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=2)

    results = {}
    for name, ctor in [("CNN", TraditionalCNN), ("ResNet50", ResNet50Wafer)]:
        print(f"\n--- {name} ---")
        m = ctor()
        t0 = time.time()
        acc = train_supervised(m, train_loader, val_loader)
        train_h = (time.time() - t0) / 3600
        infer_ms = measure_inference(m, test_loader)
        results[name] = {"accuracy": acc, "train_hours": train_h, "infer_ms": infer_ms}

    print("\n--- Autoencoder ---")
    ae = WaferAutoencoder()
    t0 = time.time()
    ae = train_autoencoder(ae, train_loader)
    train_h = (time.time() - t0) / 3600
    acc = autoencoder_classify(ae, train_loader, test_loader)
    infer_ms = measure_inference(ae, test_loader)
    results["Autoencoder"] = {"accuracy": acc, "train_hours": train_h, "infer_ms": infer_ms}

    with open(os.path.join(cfg.RESULTS_DIR, "table12_baselines.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()
