import cv2
import numpy as np
from skimage.morphology import skeletonize


def skeletonize_image(binary_img):
    """
    Binary görüntüyü skeleton haline getirir
    """
    skeleton = skeletonize(binary_img > 0)
    return skeleton.astype(np.uint8)


def compute_orientation(skeleton, x, y):
    """
    Minutiae noktasındaki yönü (orientation) hesaplar
    """
    gx = cv2.Sobel(skeleton, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(skeleton, cv2.CV_64F, 0, 1, ksize=3)

    angle = np.arctan2(gy[y, x], gx[y, x])
    return angle

def compute_density(binary_img, x, y, window_size=15):
    """
    Minutiae noktasının etrafındaki ridge (çizgi) yoğunluğunu hesaplar
    """
    half = window_size // 2

    y1 = max(0, y - half)
    y2 = min(binary_img.shape[0], y + half)
    x1 = max(0, x - half)
    x2 = min(binary_img.shape[1], x + half)

    window = binary_img[y1:y2, x1:x2]
    density = np.sum(window) / window.size

    return density

def detect_minutiae(skeleton, binary_img):
    """
    Skeleton üzerinde minutiae noktalarını bulur
    Çıkış: (x, y, angle, density)
    """
    endings = []
    bifurcations = []

    h, w = skeleton.shape

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 1:
                neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - 1

                angle = compute_orientation(skeleton, x, y)
                density = compute_density(binary_img, x, y)

                if neighbors == 1:
                    endings.append((x, y, angle, density))
                elif neighbors == 3:
                    bifurcations.append((x, y, angle, density))

    return endings, bifurcations

