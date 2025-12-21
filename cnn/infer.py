import os
import cv2
import torch
import torch.nn.functional as F

from .model import SiameseCNN


def load_image(path, size=224):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Görüntü yüklenemedi: {path}")

    img = cv2.resize(img, (size, size))
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32)
    img = img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return img


def similarity_score(img_path1, img_path2):
    device = torch.device("cpu")

    # 🔹 MODEL YOLU (GARANTİLİ)
    BASE_DIR = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(BASE_DIR, "siamese_fingerprint.pth")

    # 🔹 MODELİ OLUŞTUR
    model = SiameseCNN().to(device)

    # 🔹 AĞIRLIKLARI YÜKLE
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    img1 = load_image(img_path1).to(device)
    img2 = load_image(img_path2).to(device)

    with torch.no_grad():
        emb1, emb2 = model(img1, img2)
        distance = F.pairwise_distance(emb1, emb2)
        score = 1 / (1 + distance.item())  # normalize skor

    return score
