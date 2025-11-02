import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from difflib import SequenceMatcher
from model import CRNN  # import your trained model class

# ==============================
# 🧩 Preprocessing - EXACT same as training
# ==============================
def preprocess_input_image(img_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 512)),  # match dataset preprocessing
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(img_path).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension [1, 1, 128, 512]
    return image


# ==============================
# 🔠 Decoding logic
# ==============================
def decode_prediction(preds, idx_to_char):
    pred_texts = []
    for pred in preds:
        pred_indices = pred.argmax(1)
        text = ""
        last_char = ""
        for idx in pred_indices:
            char = idx_to_char.get(idx.item(), "")
            if char != last_char:
                if char != "-" and char != "":
                    text += char
            last_char = char
        pred_texts.append(text)
    return pred_texts


# ==============================
# 📊 Accuracy metrics
# ==============================
def word_accuracy(gt_list, pred_list):
    correct = sum(1 for gt, pr in zip(gt_list, pred_list) if gt.lower() == pr.lower())
    return correct / len(gt_list) * 100

def char_accuracy(gt_list, pred_list):
    total, correct = 0, 0
    for gt, pr in zip(gt_list, pred_list):
        matcher = SequenceMatcher(None, gt.lower(), pr.lower())
        correct += matcher.ratio() * len(gt)
        total += len(gt)
    return (correct / total) * 100 if total > 0 else 0


# ==============================
# 🚀 Evaluation Script
# ==============================
if __name__ == "__main__":
    # Paths
    model_path = "crnn_epoch_20.pth"
    test_images_dir = "test_samples"  # folder with test images
    char_set = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    # Character mapping
    idx_to_char = {i + 1: c for i, c in enumerate(char_set)}
    idx_to_char[0] = "-"  # blank for CTC

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = CRNN(imgH=128, nc=1, nclass=len(idx_to_char), nh=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Evaluate images
    gt_texts, pred_texts = [], []

    for img_file in os.listdir(test_images_dir):
        if img_file.endswith(".png") or img_file.endswith(".jpg"):
            img_path = os.path.join(test_images_dir, img_file)
            gt = os.path.splitext(img_file)[0]  # Ground truth from filename
            image = preprocess_input_image(img_path).to(device)

            with torch.no_grad():
                preds = model(image)
                pred_text = decode_prediction(preds, idx_to_char)[0]

            print(f"GT: {gt} → Pred: {pred_text}")
            gt_texts.append(gt)
            pred_texts.append(pred_text)

    # Calculate metrics
    w_acc = word_accuracy(gt_texts, pred_texts)
    c_acc = char_accuracy(gt_texts, pred_texts)

    print("\n=====================")
    print(f"✅ Word Accuracy:  {w_acc:.2f}%")
    print(f"✅ Char Accuracy:  {c_acc:.2f}%")
    print("=====================")
