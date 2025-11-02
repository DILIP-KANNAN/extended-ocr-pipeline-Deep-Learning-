import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# ✅ Configurations
# =====================================================
root_dir = os.path.expanduser(
    "~/.cache/kagglehub/datasets/nibinv23/iam-handwriting-word-database/versions/2/iam_words"
)
words_file = os.path.join(root_dir, "words.txt")
images_dir = os.path.join(root_dir, "words")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

# =====================================================
# ✅ Helper: Load Label Mapping from words.txt
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
    print(f"[INFO] Loaded {len(mapping)} word labels from words.txt")
    return mapping


label_mapping = load_label_mapping(words_file)

# =====================================================
# ✅ Dataset
# =====================================================
class IAMWordDataset(Dataset):
    def __init__(self, images_dir, label_mapping, transform=None):
        self.images_dir = images_dir
        self.label_mapping = label_mapping
        self.transform = transform
        self.samples = []

        for root, _, files in os.walk(images_dir):
            for file in files:
                if file.endswith(".png"):
                    img_id = os.path.splitext(file)[0]
                    if img_id in label_mapping:
                        path = os.path.join(root, file)
                        self.samples.append((path, label_mapping[img_id]))

        if not self.samples:
            raise RuntimeError(f"No valid labeled images found in {images_dir}")
        print(f"[INFO] Loaded {len(self.samples)} labeled images from {images_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("L")
        except (UnidentifiedImageError, OSError):
            return self.__getitem__((idx + 1) % len(self.samples))

        if self.transform:
            img = self.transform(img)
        return img, label


# =====================================================
# ✅ Character Mapping
# =====================================================
def create_charset():
    charset = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'-.,")
    return {c: i + 1 for i, c in enumerate(charset)}, {i + 1: c for i, c in enumerate(charset)}


char_to_idx, idx_to_char = create_charset()


# =====================================================
# ✅ Collate Function
# =====================================================
def collate_fn(batch):
    images, texts = zip(*batch)
    images = torch.stack(images)

    labels = []
    label_lengths = []
    for text in texts:
        encoded = [char_to_idx.get(c, 0) for c in text]
        labels.extend(encoded)
        label_lengths.append(len(encoded))

    labels = torch.tensor(labels, dtype=torch.long)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    input_lengths = torch.full(size=(images.size(0),), fill_value=images.size(3) // 4, dtype=torch.long)

    return images, labels, input_lengths, label_lengths


# =====================================================
# ✅ Transformations
# =====================================================
transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# =====================================================
# ✅ Model Definition (CNN + RNN + CTC)
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
        return x.permute(1, 0, 2)  # (T, N, C)


# =====================================================
# ✅ Utility: Decode Predictions
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
# ✅ Get DataLoader
# =====================================================
def get_loader(batch_size=16):
    ds = IAMWordDataset(images_dir=images_dir, label_mapping=label_mapping, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# =====================================================
# ✅ Training
# =====================================================
def train():
    train_loader = get_loader()
    num_classes = len(char_to_idx) + 1
    model = CRNN(num_classes).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, labels, input_lengths, label_lengths) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                print(f"[Epoch {epoch+1}] Step {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}] Avg Loss: {total_loss / len(train_loader):.4f}")

        # -------------------------------------------------
        # 🧾 Display Sample Prediction Each Epoch
        # -------------------------------------------------
        model.eval()
        with torch.no_grad():
            sample_img, gt_text = random.choice(train_loader.dataset.samples)
            img = Image.open(sample_img).convert("L")
            img_tensor = transform(img).unsqueeze(0).to(device)

            output = model(img_tensor)
            pred_text = decode_predictions(output)[0]

            plt.imshow(np.array(img), cmap="gray")
            plt.title(f"GT: {gt_text} | Pred: {pred_text}")
            plt.axis("off")
            plt.show()

        torch.save(model.state_dict(), f"crnn_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    train()
