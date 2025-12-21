import cv2
import numpy as np

def get_fingerprint_roi(binary_img):
    """
    Parmak izi bölgesini maske olarak döndürür
    """
    kernel = np.ones((15, 15), np.uint8)

    closed = cv2.morphologyEx(
        binary_img,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    mask = np.zeros(binary_img.shape, dtype=np.uint8)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)

    return mask
