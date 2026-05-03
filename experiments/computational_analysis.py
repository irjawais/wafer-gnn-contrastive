"""Section 4.4: inference time, FLOPs, memory footprint."""

import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import cfg
from data_loader import prepare_dataset
from dataset_pyg import WaferGraphDataset, collate_graphs
from model import GINJumpingKnowledge


def measure_inference_ms(model, loader, n_warmup=5):
    model.eval()
    times = []
    with torch.no_grad():
        for i, (graphs, *_ ) in enumerate(loader):
            graphs = graphs.to(cfg.DEVICE)
            if i < n_warmup:
                _ = model.encode(graphs.x, graphs.edge_index, graphs.batch)
                continue
            t0 = time.perf_counter()
            _ = model.encode(graphs.x, graphs.edge_index, graphs.batch)
            if cfg.DEVICE.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            times.append(elapsed_ms / graphs.num_graphs)
    return float(np.mean(times)), float(np.std(times))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def estimate_memory_mb(model):
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    # Activation memory: ~3x parameter footprint as a coarse heuristic
    return float(total_bytes * 4 / (1024 ** 2))


def run():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    _, _, test_df = prepare_dataset()
    test_loader = DataLoader(
        WaferGraphDataset(test_df.head(2000), precompute_descriptors=False),
        batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_graphs,
    )

    model = GINJumpingKnowledge().to(cfg.DEVICE)
    ckpt = os.path.join(cfg.CHECKPOINT_DIR, "gin_pretrained.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE))

    mean_ms, std_ms = measure_inference_ms(model, test_loader)
    params = count_parameters(model)
    mem_mb = estimate_memory_mb(model)

    out = {
        "inference_ms_per_wafer_mean": mean_ms,
        "inference_ms_per_wafer_std": std_ms,
        "num_parameters": params,
        "memory_footprint_mb": mem_mb,
        "device": str(cfg.DEVICE),
    }
    with open(os.path.join(cfg.RESULTS_DIR, "computational_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run()
