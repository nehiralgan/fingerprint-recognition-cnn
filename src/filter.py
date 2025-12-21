def filter_minutiae(minutiae, image_shape, border=10):
    """
    Görüntü kenarına çok yakın minutiae noktalarını temizler
    """
    filtered = []
    h, w = image_shape

    for x, y, t in minutiae:
        if border < x < w - border and border < y < h - border:
            filtered.append((x, y, t))

    return filtered
