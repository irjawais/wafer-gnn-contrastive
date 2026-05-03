"""Global configuration for the wafer defect detection pipeline."""

import torch

class Config:
    # Dataset
    DATA_PATH = "../MIR-WM811K/Python/LSWMD.pkl"
    WAFER_SIZE = 25
    NUM_CLASSES = 9
    CLASS_NAMES = [
        "None", "Center", "Donut", "Edge-Loc", "Edge-Ring",
        "Local", "Near-full", "Random", "Scratch"
    ]
    SAMPLE_FRAC = 0.30
    TRAIN_SPLIT = 0.60
    VAL_SPLIT = 0.20
    TEST_SPLIT = 0.20

    # Graph construction
    CONNECTIVITY = 8
    ALPHA_SEM = 0.7
    NODE_FEATURE_DIM = 9

    # Model (GIN + Jumping Knowledge)
    GIN_HIDDEN_DIMS = [64, 128, 256, 128]
    NUM_GIN_LAYERS = 4
    DROPOUT = 0.2
    PROJECTION_DIM = 128

    # Contrastive learning
    MARGIN = 2.0
    POS_THRESHOLD = 0.6
    NEG_THRESHOLD = 0.3
    SIM_WEIGHTS = {"sp": 0.4, "den": 0.3, "str": 0.2, "pr": 0.1}

    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    EPOCHS = 200
    PATIENCE = 20
    MIN_DEFECTIVE_FRAC = 0.30
    SEED = 42

    # Device — prefer CUDA, then Apple MPS, else CPU
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    # Output
    CHECKPOINT_DIR = "./checkpoints"
    RESULTS_DIR = "./results"


cfg = Config()
