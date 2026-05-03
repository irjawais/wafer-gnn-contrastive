"""Run the main self-supervised pre-training and produce primary checkpoints."""

from data_loader import prepare_dataset
from train import train_self_supervised


def run():
    train_df, val_df, _ = prepare_dataset()
    train_self_supervised(
        train_df=train_df, val_df=val_df, ckpt_name="gin_pretrained.pt",
    )


if __name__ == "__main__":
    run()
