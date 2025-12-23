import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import FingerprintPairDataset
from model import SiameseCNN


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean(
            label * distance.pow(2) +
            (1 - label) * torch.clamp(self.margin - distance, min=0).pow(2)
        )
        return loss


def train():
    device = torch.device("cpu")

    # DATASET (fail-safe)
    dataset = FingerprintPairDataset(
        data_dir="../data/train",   # SADECE TRAIN
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset boş. Train klasörünü kontrol et.")

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        drop_last=True   # tek kalan batch sorununu engeller
    )

    model = SiameseCNN().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 8

    print("\n>>> EĞİTİM BAŞLADI <<<\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        valid_batches = 0

        for img1, img2, label in loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)

            loss.backward()
            optimizer.step() #Embedding uzayı güncellenir

            total_loss += loss.item()
            valid_batches += 1

        avg_loss = total_loss / max(valid_batches, 1)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    # MODEL KAYIT
    torch.save(model.state_dict(), "siamese_fingerprint.pth")
    print("\nModel kaydedildi: siamese_fingerprint.pth\n")


if __name__ == "__main__":
    train()
