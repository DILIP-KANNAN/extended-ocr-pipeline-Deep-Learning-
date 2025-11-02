
# Simple inference example: load checkpoint and run greedy decode on a single image
import torch
from models.crnn_model import CRNN
from utils.config import MODEL_DIR, DEVICE
from utils.data_loader import IAMLineDataset
from utils.text_utils import TextMapper

def infer_one(image_path, ckpt=None):
    device = torch.device(DEVICE)
    model = CRNN().to(device)
    if ckpt is None:
        ckpt = os.path.join(MODEL_DIR, sorted([p for p in os.listdir(MODEL_DIR) if p.endswith('.pth')])[-1])
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    ds = IAMLineDataset()
    # find sample by name
    for p, label in ds.samples:
        if p.endswith(image_path):
            img, _, _ = ds[0]
            break
    # This is a placeholder for proper usage - adapt as required.
