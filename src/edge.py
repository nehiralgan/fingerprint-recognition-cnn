import cv2

def extract_edges(img):
    """
    Canny algoritması ile parmak izi kenarlarını çıkarır
    """
    edges = cv2.Canny(
        img,
        threshold1=50,
        threshold2=150
    )
    return edges
