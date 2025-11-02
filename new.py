import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from jiwer import wer, cer  # pip install jiwer

# =====================================================
# ✅ Configurations
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

root_dir = os.path.expanduser(
    "~/.cache/kagglehub/datasets/nibinv23/iam-handwriting-word-database/versions/2/iam_words"
)
words_file = os.path.join(root_dir, "words.txt")
images_dir = os.path.join(root_dir, "words")

# =====================================================
# ✅ Charset
# =====================================================
def create_charset():
    charset = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'-.,")
    return {c: i + 1 for i, c in enumerate(charset)}, {i + 1: c for i, c in enumerate(charset)}

char_to_idx, idx_to_char = create_charset()


# =====================================================
# ✅ Model Definition (same as training)
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
# ✅ Preprocessing Transform
# =====================================================
transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# =====================================================
# ✅ Load Model
# =====================================================
def load_model(model_path):
    num_classes = len(char_to_idx) + 1
    model = CRNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[INFO] Model loaded from {model_path}")
    return model


# =====================================================
# ✅ Predict Function
# =====================================================
def predict_image(model, image_path):
    img = Image.open(image_path).convert("L")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(img_tensor)
        pred_text = decode_predictions(preds)[0]

    plt.imshow(np.array(img), cmap="gray")
    plt.title(f"Prediction: {pred_text}")
    plt.axis("off")
    plt.show()

    return pred_text


# =====================================================
# ✅ Evaluate on Sample Folder
# =====================================================
def load_label_mapping(words_file):
    mapping = {}
    with open(words_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(" ")
            if len(parts) >= 9:
                image_id = parts[0]
                word = parts[-1]
                mapping[image_id] = word
    return mapping


def evaluate_model(model, num_samples=50):
    mapping = load_label_mapping(words_file)
    image_paths, gt_texts, pred_texts = [], [], []

    count = 0
    for root, _, files in os.walk(images_dir):
        for file in files:
            if not file.endswith(".png"):
                continue
            img_id = os.path.splitext(file)[0]
            if img_id not in mapping:
                continue

            img_path = os.path.join(root, file)
            gt_text = mapping[img_id]
            pred = predict_image(model, img_path)

            image_paths.append(img_path)
            gt_texts.append(gt_text)
            pred_texts.append(pred)
            count += 1

            if count >= num_samples:
                break

    # -------------------------------------------------
    # ✅ Compute Evaluation Metrics
    # -------------------------------------------------
    total_chars = sum(len(gt) for gt in gt_texts)
    correct_chars = sum(sum(a == b for a, b in zip(gt, pr)) for gt, pr in zip(gt_texts, pred_texts))
    char_acc = correct_chars / total_chars if total_chars > 0 else 0

    cer_score = cer(gt_texts, pred_texts)
    wer_score = wer(gt_texts, pred_texts)

    print("\n================ Evaluation Report ================")
    print(f"Samples Tested: {len(gt_texts)}")
    print(f"Character Accuracy: {char_acc*100:.2f}%")
    print(f"CER (Character Error Rate): {cer_score:.4f}")
    print(f"WER (Word Error Rate): {wer_score:.4f}")
    print("===================================================\n")

    return gt_texts, pred_texts


# =====================================================
# ✅ Run Evaluation / Custom Prediction
# =====================================================
if __name__ == "__main__":
    model_path = "crnn_epoch_20.pth"  # Change this to your trained model
    model = load_model(model_path)

    # 🔹 For Custom Image Prediction:
    # image_path = "custom_test.png"
    # predict_image(model, image_path)

    # 🔹 For Evaluation on IAM Samples:
    evaluate_model(model, num_samples=10)
