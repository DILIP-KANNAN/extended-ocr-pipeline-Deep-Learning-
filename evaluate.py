import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
from jiwer import wer, cer

# ======================
# ✅ Model Definition
# ======================
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(128 * 8, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.permute(1, 0, 2)

# ======================
# ✅ Charset
# ======================
def create_charset():
    charset = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'-.,")
    return {c: i + 1 for i, c in enumerate(charset)}, {i + 1: c for i, c in enumerate(charset)}

char_to_idx, idx_to_char = create_charset()

# ======================
# ✅ Decode Predictions
# ======================
def decode_predictions(preds):
    preds = torch.argmax(preds, dim=2).permute(1, 0)
    decoded_texts = []
    for seq in preds:
        text = "".join([idx_to_char.get(idx.item(), "") for idx in seq])
        text = "".join(ch for i, ch in enumerate(text) if i == 0 or ch != text[i - 1])
        decoded_texts.append(text.strip())
    return decoded_texts

# ======================
# ✅ Preprocess Image
# ======================
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 31, 15)
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]
    img = 255 - img
    desired_h = 32
    scale = desired_h / img.shape[0]
    new_w = int(img.shape[1] * scale)
    img = cv2.resize(img, (min(new_w, 128), desired_h))
    if new_w < 128:
        pad_width = 128 - new_w
        img = np.pad(img, ((0, 0), (0, pad_width)), constant_values=255)
    img = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(img).unsqueeze(0)

# ======================
# ✅ Evaluate Function
# ======================
def evaluate_model(model, test_dir):
    image_files = [f for f in os.listdir(test_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    true_texts = []
    pred_texts = []

    for img_file in image_files:
        gt_text = os.path.splitext(img_file)[0].replace("_", " ")
        img_path = os.path.join(test_dir, img_file)
        tensor = preprocess_image(img_path).to(device)
        with torch.no_grad():
            preds = model(tensor)
            pred_text = decode_predictions(preds)[0]
        true_texts.append(gt_text.strip())
        pred_texts.append(pred_text.strip())
        print(f"GT: {gt_text} → Pred: {pred_text}")

    word_acc = 1 - wer(true_texts, pred_texts)
    char_acc = 1 - cer(true_texts, pred_texts)

    print("\n=====================")
    print(f"✅ Word Accuracy:  {word_acc * 100:.2f}%")
    print(f"✅ Char Accuracy:  {char_acc * 100:.2f}%")
    print("=====================")

# ======================
# ✅ Run Evaluation
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(char_to_idx) + 1
model = CRNN(num_classes).to(device)
model.load_state_dict(torch.load("crnn_epoch_20.pth", map_location=device))
model.eval()

test_dir = "test_samples"  # 🔹 Folder with test images named as <text>.png
evaluate_model(model, test_dir)
