import os
import torch

# === Path Setup ===
DATA_DIR = r"C:\Users\dilip\.cache\kagglehub\datasets\nibinv23\iam-handwriting-word-database\versions\2"

# === Image Settings ===
IMG_HEIGHT = 32
IMG_MAX_WIDTH = 256
CHANNELS = 1

# === DataLoader Settings ===
BATCH_SIZE = 8
NUM_WORKERS = 2

# === Character Set ===
CHARACTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-'"
BLANK_IDX = 0  # reserved for CTC blank
NUM_CLASSES = len(CHARACTERS) + 1

# === Model Hyperparameters ===
CNN_OUT_DIM = 256
RNN_HIDDEN_SIZE = 256
RNN_NUM_LAYERS = 2

# === Training Hyperparameters ===
EPOCHS = 50
LEARNING_RATE = 1e-3

# === Device ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
