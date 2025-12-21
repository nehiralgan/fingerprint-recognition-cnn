Fingerprint Recognition System
Hybrid Fingerprint Matching with CNN & Classical Vision

Live Demo / Example Output

If you want to see example outputs or visual results, check the screenshots/ directory or generate via running the system.

Project Overview

This repository implements a hybrid fingerprint recognition system that combines:
Convolutional Neural Network (Siamese CNN) for learned similarity
Minutiae extraction & matching for structural fingerprint features
Liveness detection to reject fake fingerprints
Visualization tools for matched minutiae and decision explanation

The system processes fingerprint images, computes similarity scores, and produces human-interpretable match visualizations.

What This Project Does

Fingerprint recognition is essential in biometric authentication. This system:
Preprocesses fingerprint images (binarization, skeletonization)
Extracts minutiae points (ridge endings & bifurcations)
Computes structural similarity via point matching
Computes embedding similarity via a Siamese CNN
Fuses scores for robust identity decision
Detects liveness (points to potential spoof fingerprints)
Visualizes matched features (top strongest matches)
This hybrid approach improves accuracy and interpretability compared to single-method systems.

Features
Feature Extraction

Skeletonization of fingerprint patterns
Local orientation & density scoring
Ending and bifurcation detection

Siamese CNN

Lightweight CNN trained with contrastive loss
Learns fingerprint embeddings for similarity

Liveness Detection

Rejects fakes based on texture & frequency analysis

Score Fusion and Decision Logic

Weighted fusion:
final_score = 0.4 × CNN_score + 0.6 × Minutiae_score
Ambiguity margin controls uncertain decisions
Thresholding for acceptance / rejection

Visualization

Two separate windows showing matched minutiae
Top-20 strongest matches numbered and color-coded

Repository Structure
fingerprint_project/
├── cnn/                      # CNN training & inference
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── infer.py
│
├── src/                      # Classical fingerprint analysis
│   ├── preprocess.py
│   ├── minutiae.py
│   ├── matcher.py
│   ├── liveness.py
│   └── visualize.py
│
├── data/                     # Train/test fingerprint images (not tracked)
│   ├── train/
│   └── test/
│
├── main.py                   # Runs full system on test set
├── requirements.txt
├── .gitignore
└── README.md

Installation

Ensure you have Python 3.10+, then create a virtual environment and install dependencies:

python -m venv venv
venv\Scripts\activate         # Windows
pip install -r requirements.txt


Dependencies include:

OpenCV
PyTorch
scikit-image
SciPy
(full list in requirements.txt)

Training the CNN

To train the fingerprint similarity model:

cd cnn
python train.py

This will produce a model file (e.g., siamese_fingerprint.pth).
Note: Model weights are not included in the repository.

Running Recognition

To run the full recognition pipeline:

python main.py

Output will include:
Liveness score
Scores for each enrolled person
Final decision (Accepted / Ambiguous / Rejected)
Visualization of matched minutiae points

How It Works (Technical Summary)
Minutiae Matching

Minutiae points are extracted and filtered. Matched pairs are found between test and reference prints. Top matched pairs show structural similarity.

Siamese CNN

Pairs of fingerprint images are embedded into a learned space. Similarity is computed as:

score = 1 / (1 + euclidean_distance)

Score Fusion & Decision

Final system decision is based on:
Weighted combination of CNN and structural scores
Threshold for valid identity
Gap margin to avoid ambiguous decisions
This design balances learned patterns and structural features.

Use Cases

Biometric authentication research
Academic demonstration of hybrid matching
Fingerprint liveness evaluation
Visual demonstration of matching

Limitations

Dataset is small — model accuracy is limited
CPU-only inference — slower than GPU
Not for production security systems
This project is for learning, experimentation, and prototyping.

Citation & Attribution

This project draws inspiration from hybrid approaches in biometrics and interactive CNN explainer models like CNN Explainer: an interactive CNN visualization tool. 

Contributing

Feel free to open issues or pull requests.
For major changes, please discuss before submitting.

Contact

If you have questions about this project, feel free to open an issue or contact the author.

License

Specify your license (e.g. MIT License) if you choose.


🇹🇷 
Proje Özeti

Bu proje, parmak izi tanıma için geliştirilmiş hibrit bir sistemdir.
CNN tabanlı öğrenme ile klasik görüntü işleme (minutiae) birlikte kullanılır.

Kurulum
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Eğitim
cd cnn
python train.py

Çalıştırma
python main.py

Özellikler

Minutiae çıkarımı
Siamese CNN benzerlik skorlaması
Canlılık testi
Eşleşen noktaların görsel gösterimi

Kısıtlamalar
Küçük veri seti
CPU ile çalışır.
Üretim için hazır değildir.
