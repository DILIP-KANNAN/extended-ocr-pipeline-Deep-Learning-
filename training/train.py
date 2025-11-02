
import os, argparse, time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from models.crnn_model import CRNN
from utils.config import DEVICE, MODEL_DIR, EPOCHS, LEARNING_RATE
from utils.data_loader import get_loader
from utils.text_utils import TextMapper
from training.evaluate import greedy_decode, compute_cer

def train(args):
    os.makedirs(MODEL_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join('outputs','logs'))
    device = torch.device(DEVICE)
    model = CRNN().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    mapper = TextMapper()
    loader = get_loader(batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for i, (images, labels, input_lengths, label_lengths) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)
            optimizer.zero_grad()
            logits = model(images)  # T x B x C
            log_probs = nn.functional.log_softmax(logits, dim=2)
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i+1) % 10 == 0:
                print(f'Epoch {epoch+1}/{args.epochs} Step {i+1} Loss {loss.item():.4f}')
        avg = total_loss / (i+1)
        print(f'Epoch {epoch+1} AvgLoss {avg:.4f}')
        writer.add_scalar('train/loss', avg, epoch+1)
        # quick validation run (single batch) to show progress
        model.eval()
        with torch.no_grad():
            loader_val = get_loader(batch_size=4, shuffle=False)
            for images, labels, input_lengths, label_lengths in loader_val:
                images = images.to(device)
                logits = model(images)
                preds = greedy_decode(logits, mapper)
                print('Sample Predictions:')
                for p in preds[:4]:
                    print(' >', p)
                break
        # save checkpoint
        ckpt = os.path.join(MODEL_DIR, f'crnn_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), ckpt)
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
