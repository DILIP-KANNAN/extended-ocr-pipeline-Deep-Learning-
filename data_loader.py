import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class HandwritingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
        self.transform = transform if transform else transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 512)),  # uniform height and fixed width
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")
        image = self.transform(image)
        label = os.path.basename(img_path).split(".")[0]  # filename as pseudo-label
        return image, label


def get_dataloader(root_dir, batch_size=16, shuffle=True):
    dataset = HandwritingDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    # Load dataset
    dataset_dir = open(r"C:\Users\dilip\OneDrive\Desktop\handwriting_to_text\handwriting_to_text\dataset_path.txt").read().strip()
    data_path = os.path.join(dataset_dir, "data")
    
    dataloader = get_dataloader(data_path, batch_size=8)
    print(f"✅ Loaded {len(dataloader.dataset)} images.")

    # Test one batch
    for imgs, labels in dataloader:
        print("Batch shape:", imgs.shape)
        print("Sample labels:", labels[:4])
        break
