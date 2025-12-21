import os
import random
import cv2
import torch
from torch.utils.data import Dataset


class FingerprintPairDataset(Dataset):
    def __init__(self, data_dir, image_size=224):
        self.data_dir = data_dir
        self.image_size = image_size
        self.pairs = []

        self._prepare_pairs()

    def _prepare_pairs(self):
        persons = os.listdir(self.data_dir)

        images_by_person = {}

        for person in persons:
            person_path = os.path.join(self.data_dir, person)
            if not os.path.isdir(person_path):
                continue

            images = [
                os.path.join(person_path, f)
                for f in os.listdir(person_path)
                if f.endswith(".png")
            ]

            images_by_person[person] = images

        # Pozitif çiftler (aynı kişi)
        for person, images in images_by_person.items():
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    self.pairs.append((images[i], images[j], 1))

        # Negatif çiftler (farklı kişiler)
        persons_list = list(images_by_person.keys())
        for _ in range(len(self.pairs)):
            p1, p2 = random.sample(persons_list, 2)
            img1 = random.choice(images_by_person[p1])
            img2 = random.choice(images_by_person[p2])
            self.pairs.append((img1, img2, 0))

        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        return img

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]

        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)

        return img1, img2, torch.tensor(label, dtype=torch.float32)
