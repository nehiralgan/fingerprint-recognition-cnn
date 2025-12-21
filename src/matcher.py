import math

def match_minutiae(test_pts, ref_pts, dist_thresh=15, angle_thresh=20):
    """
    Çıkış:
    score (0-1),
    matched_pairs: [((x1,y1),(x2,y2)), ...]
    """

    matched_pairs = []
    used_ref = set()

    for x1, y1, a1, d1 in test_pts:
        best_match = None
        best_dist = float("inf")

        for idx, (x2, y2, a2, d2) in enumerate(ref_pts):
            if idx in used_ref:
                continue

            dist = math.hypot(x1 - x2, y1 - y2)
            angle_diff = abs(a1 - a2)

            if dist < dist_thresh and angle_diff < angle_thresh:
                if dist < best_dist:
                    best_dist = dist
                    best_match = (idx, (x2, y2))

        if best_match:
            used_ref.add(best_match[0])
            matched_pairs.append(((x1, y1), best_match[1]))

    score = len(matched_pairs) / max(len(test_pts), len(ref_pts))
    return score, matched_pairs
