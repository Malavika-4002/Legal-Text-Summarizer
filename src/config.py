import os
import torch

# 1. Get the main folder path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Define where Data and Models live
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# 3. Specific paths for Phase 1 (SBD)
SBD_WEIGHTS_PATH = os.path.join(MODEL_DIR, "sbd_weights", "cnn_model.pth")
SBD_CRF_PATH = os.path.join(MODEL_DIR, "sbd_weights", "crf_hybrid_model.joblib")
VOCAB_PATH = os.path.join(MODEL_DIR, "sbd_weights", "char_vocab.json")

# 4. Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"