import cv2
import numpy as np

def liveness_score(gray_img):
    if gray_img is None:
        return 0.0

    # 1️⃣ Kontrast
    contrast = gray_img.std() / 255.0

    # 2️⃣ Kenar yoğunluğu
    edges = cv2.Canny(gray_img, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # 3️⃣ Gradient varyansı
    gx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_var = np.var(gx) + np.var(gy)

    # Normalize
    gradient_score = min(gradient_var / 1000, 1.0)

    # 🔥 Birleşik skor
    score = (
        0.4 * contrast +
        0.3 * edge_density +
        0.3 * gradient_score
    )

    return round(score, 3)
