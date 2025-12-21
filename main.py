import os
import cv2

from cnn.infer import similarity_score
from src.preprocess import preprocess_image
from src.minutiae import skeletonize_image, detect_minutiae
from src.matcher import match_minutiae
from src.liveness import liveness_score
from src.visualize import visualize_matches

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"


def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    binary = preprocess_image(img)
    skeleton = skeletonize_image(binary)
    endings, bifurcations = detect_minutiae(skeleton, binary)
    return endings + bifurcations


def main():

    THRESHOLD = 0.50
    LIVENESS_THRESHOLD = 0.25
    AMBIGUITY_MARGIN = 0.02
    MAX_REF = 5

    test_files = ["test_fp8.png"]

    for test_file in test_files:
        test_image = os.path.join(TEST_DIR, test_file)
        print(f"\n=== TEST: {test_file} ===")

        test_minutiae = extract_features(test_image)
        test_img_gray = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)

        live_score = liveness_score(test_img_gray)
        print(f"Liveness skoru: {live_score:.3f}")

        if live_score < LIVENESS_THRESHOLD:
            print("SAHTE – RED")
            continue

        person_results = []

        persons = sorted(
            os.listdir(TRAIN_DIR),
            key=lambda x: int(x.replace("person", ""))
        )

        for person in persons:
            person_path = os.path.join(TRAIN_DIR, person)
            if not os.path.isdir(person_path):
                continue

            best_score = 0
            best_ref = None
            best_pairs = None

            for file in os.listdir(person_path)[:MAX_REF]:
                if not file.endswith(".png"):
                    continue

                ref_image = os.path.join(person_path, file)
                ref_minutiae = extract_features(ref_image)

                minutiae_score, pairs = match_minutiae(
                    test_minutiae, ref_minutiae
                )

                cnn_score = similarity_score(test_image, ref_image)
                final_score = 0.6 * cnn_score + 0.4 * minutiae_score

                if final_score > best_score:
                    best_score = final_score
                    best_ref = ref_image
                    best_pairs = pairs

            print(f"{person} → En iyi skor: {best_score:.3f}")
            person_results.append((person, best_score, best_ref, best_pairs))

        person_results.sort(key=lambda x: x[1], reverse=True)

        best_person, best_score, best_ref, best_pairs = person_results[0]
        second_score = person_results[1][1]

        print("\n--- KARAR ---")
        print(f"1️⃣ {best_person}: {best_score:.3f}")
        print(f"2️⃣ skor: {second_score:.3f}")

        if best_score < THRESHOLD:
            print("SONUÇ: Eşleşme yok")

        elif best_score - second_score < AMBIGUITY_MARGIN:
            print("SONUÇ: BELİRSİZ")

        else:
            print(f"SONUÇ: KİMLİK DOĞRULANDI → {best_person}")
            visualize_matches(test_image, best_ref, best_pairs)


if __name__ == "__main__":
    main()
