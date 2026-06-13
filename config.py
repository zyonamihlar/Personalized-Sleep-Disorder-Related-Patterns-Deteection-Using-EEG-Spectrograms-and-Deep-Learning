import os

# =============================
# Project paths
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =============================
# EEG settings
# =============================
EEG_CHANNEL = "EEG Fpz-Cz"
LOW_FREQ = 0.5
HIGH_FREQ = 30.0
EPOCH_SECONDS = 30
TARGET_SAMPLE_RATE = 100

# =============================
# Spectrogram settings
# =============================
N_PER_SEG = 128
N_OVERLAP = 64

# =============================
# Training settings
# =============================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 8
NUM_CLASSES = 5

CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]