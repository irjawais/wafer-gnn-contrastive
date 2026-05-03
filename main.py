"""Entry point — orchestrate self-supervised pre-training, fine-tuning, and eval."""

import argparse
import os
import random

import numpy as np
import torch

from config import cfg
from data_loader import prepare_dataset
from evaluate import main as evaluate_main
from train import fine_tune_classifier, train_self_supervised


def _seed_all():
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["pretrain", "finetune", "eval", "eval-finetuned", "all"],
        default="all",
        help=(
            "pretrain: self-supervised pre-training only. "
            "finetune: supervised fine-tuning of pre-trained backbone. "
            "eval: clustering + sklearn MLP on frozen pre-trained embeddings. "
            "eval-finetuned: in-model logits from fine-tuned checkpoint. "
            "all: pretrain -> finetune -> eval (both paths)."
        ),
    )
    return p.parse_args()


def main():
    _seed_all()
    args = parse_args()

    if args.mode in ("pretrain", "all"):
        train_self_supervised()

    if args.mode in ("finetune", "all"):
        train_df, val_df, _ = prepare_dataset()
        pretrained = os.path.join(cfg.CHECKPOINT_DIR, "gin_pretrained.pt")
        fine_tune_classifier(
            pretrained_ckpt=pretrained,
            train_df=train_df, val_df=val_df,
            epochs=30, lr=1e-4, out_ckpt="gin_finetuned.pt",
        )

    if args.mode in ("eval", "all"):
        print("\n>>> Path 1: frozen pre-trained embeddings <<<")
        evaluate_main(use_finetuned=False)

    if args.mode in ("eval-finetuned", "all"):
        print("\n>>> Path 2: fine-tuned in-model classifier <<<")
        evaluate_main(use_finetuned=True)


if __name__ == "__main__":
    main()
