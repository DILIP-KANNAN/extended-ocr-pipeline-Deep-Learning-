import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms

# -----------------------------
# Import config
# -----------------------------
from utils.config import BATCH_SIZE, NUM_WORKERS, IMG_HEIGHT, IMG_MAX_WIDTH, DATA_DIR

# -----------------------------
# Image transform
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# Character Mapper
# -----------------------------
class TextMapper:
    def __init__(self, characters="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-'"):
        self.chars = characters
        self.char_to_idx = {c: i+1 for i, c in enumerate(characters)}  # 0 is CTC blank
        self.idx_to_char = {i+1: c for i, c in enumerate(characters)}
        self.blank_idx = 0

    def encode(self, text):
        return [self.char_to_idx.get(c, self.blank_idx) for c in text]

# -----------------------------
# Dataset for IAM word-level
# -----------------------------
class IAMWordDataset(Dataset):
    def __init__(self, images_dir=None, label_file=None, mapper=None, max_width=IMG_MAX_WIDTH):
        self.images_dir = images_dir or os.path.join(DATA_DIR, "iam_words", "words")
        self.label_file = label_file or os.path.join(DATA_DIR, "iam_words", "words.txt")
        self.mapper = mapper or TextMapper()
        self.max_width = max_width

        # Load word labels
        self.samples = []
        with open(self.label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 9 and parts[1] == "ok":  # only correct segmentation
                    word_id = parts[0]
                    transcription = " ".join(parts[8:])  # ASCII word label
                    img_path = os.path.join(self.images_dir, f"{word_id}.png")
                    if os.path.isfile(img_path):
                        self.samples.append((img_path, transcription))
        print(f"[INFO] Loaded {len(self.samples)} labeled samples from {self.images_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")

        # maintain aspect ratio
        w, h = img.size
        new_h = IMG_HEIGHT
        new_w = int(w * (new_h / h))
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # pad / truncate width
        arr = np.array(img).astype(np.float32) / 255.0
        if new_w < self.max_width:
            pad = np.ones((new_h, self.max_width), dtype=np.float32)
            pad[:, :new_w] = arr
            arr = pad
        else:
            arr = arr[:, :self.max_width]

        img_t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # add channel dimension
        encoded_label = torch.tensor(self.mapper.encode(label), dtype=torch.long)
        return img_t, encoded_label, len(encoded_label)

# -----------------------------
# Collate function for CTC
# -----------------------------
def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.cat(labels)
    label_lengths = torch.tensor(lengths, dtype=torch.long)

    # approximate input lengths after CNN (downsample width by 4)
    B, C, H, W = images.shape
    input_lengths = torch.full((B,), W // 4, dtype=torch.long)
    return images, labels, input_lengths, label_lengths

# -----------------------------
# DataLoader
# -----------------------------
def get_loader(batch_size=BATCH_SIZE, shuffle=True):
    ds = IAMWordDataset()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=NUM_WORKERS, collate_fn=collate_fn)
    return loader
