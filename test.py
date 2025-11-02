import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# =====================================================
# ✅ Config
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "crnn_epoch_20.pth"
print(f"[INFO] Device: {device}")

# =====================================================
# ✅ Charset (same as training)
# =====================================================
def create_charset():
    charset = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'-.,")
    return {c: i + 1 for i, c in enumerate(charset)}, {i + 1: c for i, c in enumerate(charset)}

char_to_idx, idx_to_char = create_charset()

# =====================================================
# ✅ Model Definition
# =====================================================
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

# =====================================================
# ✅ Decode Predictions
# =====================================================
def decode_predictions(preds):
    preds = torch.argmax(preds, dim=2).permute(1, 0)
    decoded_texts = []
    for seq in preds:
        text = "".join([idx_to_char.get(idx.item(), "") for idx in seq])
        text = "".join(ch for i, ch in enumerate(text) if i == 0 or ch != text[i - 1])
        decoded_texts.append(text.strip())
    return decoded_texts

# =====================================================
# ✅ Preprocessing (robust to real handwriting)
# =====================================================
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Apply adaptive threshold to clean noise
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 31, 15)

    # Find contours and crop tight bounding box
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]

    # Invert back to white background
    img = 255 - img

    # Resize with preserved aspect ratio
    desired_h = 32
    scale = desired_h / img.shape[0]
    new_w = int(img.shape[1] * scale)
    img = cv2.resize(img, (new_w, desired_h))

    # Pad width to 128 max
    if new_w < 128:
        pad_width = 128 - new_w
        img = np.pad(img, ((0, 0), (0, pad_width)), constant_values=255)
    else:
        img = cv2.resize(img, (128, 32))

    # Normalize
    img = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(img).unsqueeze(0)


# =====================================================
# ✅ Predict Custom Image
# =====================================================
def predict(model, image_path):
    tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        preds = model(tensor)
        text = decode_predictions(preds)[0]
    return text


# =====================================================
# ✅ Load and Run
# =====================================================
if __name__ == "__main__":
    num_classes = len(char_to_idx) + 1
    model = CRNN(num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    image_path = "image.png"  # 👉 replace with your own image
    pred = predict(model, image_path)

    print(f"\n🧾 Predicted Text: {pred}")
    plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap="gray")
    plt.title(f"Prediction: {pred}")
    plt.axis("off")
    plt.show()
