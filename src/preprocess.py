import cv2
import numpy as np

def preprocess_image(img):
    """
    Parmak izi görüntüsünü ikili (binary) hale getirir
    """
    # Gürültü azaltma
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Adaptif threshold
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    return binary
