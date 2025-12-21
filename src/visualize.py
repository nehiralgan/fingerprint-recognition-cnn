import cv2


def visualize_matches(test_img_path, ref_img_path, matched_pairs, top_k=20):
    test_img = cv2.imread(test_img_path)
    ref_img = cv2.imread(ref_img_path)

    if test_img is None or ref_img is None:
        print("Görseller yüklenemedi")
        return

    test_vis = test_img.copy()
    ref_vis = ref_img.copy()

    # 🔹 SADECE EN GÜÇLÜ 20 EŞLEŞME
    top_matches = matched_pairs[:top_k]

    for i, pair in enumerate(top_matches):
        (x1, y1), (x2, y2) = pair

        # Test → yeşil
        cv2.circle(test_vis, (int(x1), int(y1)), 4, (0, 255, 0), -1)

        # Referans → kırmızı
        cv2.circle(ref_vis, (int(x2), int(y2)), 4, (0, 0, 255), -1)

        # 🔹 Numara ekle (sunum için çok iyi)
        cv2.putText(
            test_vis, str(i + 1),
            (int(x1) + 5, int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
        )

        cv2.putText(
            ref_vis, str(i + 1),
            (int(x2) + 5, int(y2) - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
        )

    cv2.imshow("TEST PARMAK IZI (Top 20 Eslesme)", test_vis)
    cv2.imshow("REFERANS PARMAK IZI (Top 20 Eslesme)", ref_vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
