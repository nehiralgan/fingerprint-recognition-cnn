**Fingerprint Recognition System
Hybrid Fingerprint Matching with CNN & Classical Vision**

This repository implements a hybrid fingerprint recognition system that combines:
convolutional Neural Network (Siamese CNN) for learned similarity,
minutiae extraction & matching for structural fingerprint features,
liveness detection to reject fake fingerprints,
visualization tools for matched minutiae and decision explanation.

The system processes fingerprint images, computes similarity scores, and produces human-interpretable match visualizations.

**Note:** If you want to see example outputs or visual results, check the screenshots/ directory or generate via running the system.

**General Information**

Fingerprint recognition is essential in biometric authentication. This system:
Preprocesses fingerprint images (binarization, skeletonization),
extracts minutiae points (ridge endings & bifurcations),
computes structural similarity via point matching,
computes embedding similarity via a Siamese CNN,
fuses scores for robust identity decision,
detects liveness (points to potential spoof fingerprints),
visualizes matched features (top strongest matches).
This hybrid approach improves accuracy and interpretability compared to single-method systems.

**Features
Feature Extraction**

Skeletonization of fingerprint patterns
Local orientation & density scoring
Ending and bifurcation detection

**Liveness Detection**

Rejects fakes based on texture & frequency analysis

**Score Fusion and Decision Logic**

Weighted fusion:
final_score = 0.4 × CNN_score + 0.6 × Minutiae_score,
Ambiguity margin controls uncertain decisions,
Thresholding for acceptance / rejection.

**Visualization**

Two separate windows showing matched minutiae,
Top-20 strongest matches numbered and color-coded.

**Requirements**

Ensure you have Python 3.10+, then create a virtual environment and install dependencies:

`python -m venv venv                                                                                                              
venv\Scripts\activate       # Windows                                                                                              
pip install -r requirements.txt`

Dependencies include:

OpenCV                                                                                   
PyTorch                                                                               
scikit-image                                                                       
SciPy

**Training the CNN**

To train the fingerprint similarity model:

`cd cnn
python train.py`

This will produce a model file (e.g., siamese_fingerprint.pth).

**Note:** Model weights are not included in the repository.

**Running Recognition**

To run the full recognition pipeline:

`python main.py`

Output will include:                                                           
Liveness score                                                             
Scores for each enrolled person                                            
Final decision (Accepted / Ambiguous / Rejected)                                 
Visualization of matched minutiae points

**How It Works (Technical Summary)                                                                                                         
Minutiae Matching**

Minutiae points are extracted and filtered. Matched pairs are found between test and reference prints. Top matched pairs show structural similarity.

**Siamese CNN**

Pairs of fingerprint images are embedded into a learned space. 
Similarity is computed as:

`score = 1 / (1 + euclidean_distance)`

**Score Fusion & Decision**

Final system decision is based on:
Weighted combination of CNN and structural scores
Threshold for valid identity
Gap margin to avoid ambiguous decisions
This design balances learned patterns and structural features.

**Use Cases**

Biometric authentication research
Academic demonstration of hybrid matching
Fingerprint liveness evaluation
Visual demonstration of matching

**Limitations**

Dataset is small — model accuracy is limited
CPU-only inference — slower than GPU
Not for production security systems
This project is for learning, experimentation, and prototyping.

**Citation & Attribution**

This project draws inspiration from hybrid approaches in biometrics and interactive CNN explainer models like CNN Explainer: an interactive CNN visualization tool. 

**Contributing**

Feel free to open issues or pull requests.
For major changes, please discuss before submitting.

**Contact**

If you have questions about this project, feel free to open an issue or contact the author.



**🇹🇷** 

**Parmak İzi Tanıma Sistemi**

**CNN ve Klasik Görüntü İşleme ile Hibrit Parmak İzi Eşleştirme**

Bu depo, aşağıdaki yöntemleri birleştiren hibrit bir parmak izi tanıma sistemi sunmaktadır:
Öğrenilmiş benzerlik için Evrişimli Sinir Ağı (Siamese CNN)
Yapısal parmak izi özellikleri için minütia çıkarımı ve eşleştirme
Sahte parmak izlerini elemek için canlılık (liveness) tespiti
Eşleşen minütiaları ve karar sürecini açıklamak için görselleştirme araçları
Sistem, parmak izi görüntülerini işler, benzerlik skorlarını hesaplar ve insan tarafından yorumlanabilir eşleşme görselleri üretir.

**Not:** Örnek çıktı veya görsel sonuçları görmek için screenshots/ dizinine bakabilir ya da sistemi çalıştırarak çıktıları kendiniz üretebilirsiniz.

**Genel Bilgiler**

Parmak izi tanıma, biyometrik kimlik doğrulama sistemlerinde kritik bir rol oynar. Bu sistem:
Parmak izi görüntülerini ön işler (ikili hale getirme, iskelet çıkarımı),
minutiae noktalarını çıkarır (ridge bitişleri ve çatallanma noktaları),
nokta eşleştirme ile yapısal benzerlik hesaplar,
siamese CNN ile öznitelik (embedding) benzerliği hesaplar,
daha güvenilir bir kimlik kararı için skorları birleştirir,
sahte parmak izlerini tespit etmek için canlılık analizi yapar,
eşleşen özellikleri görselleştirir (en güçlü eşleşmeler).
Bu hibrit yaklaşım, tek bir yönteme dayalı sistemlere kıyasla daha yüksek doğruluk ve daha iyi yorumlanabilirlik sağlar.

**Özellikler
Özellik Çıkarımı**

Parmak izi desenlerinin iskeletleştirilmesi,
yerel yönelim ve yoğunluk skorlama,
ridge bitişi ve çatallanma tespiti,
canlılık (Liveness) tespiti,
doku ve frekans analizi kullanarak sahte parmak izlerini reddeder.

**Skor Birleştirme ve Karar Mantığı**

Ağırlıklı skor birleşimi:

final_score = 0.4 × CNN_skoru + 0.6 × Minütia_skoru

Belirsiz kararları kontrol etmek için belirsizlik marjı,
Kabul / ret için eşik tabanlı karar mekanizması

**Görselleştirme**

Eşleşen minütiaları iki ayrı pencerede gösterir.
En güçlü 20 eşleşme numaralandırılmış ve renklendirilmiş şekilde çizilir.

**Gereksinimler**

Python 3.10 veya üzeri bir sürümün yüklü olduğundan emin olun. Ardından bir sanal ortam oluşturup bağımlılıkları yükleyin:

`python -m venv venv                                                                                                                  
venv\Scripts\activate       # Windows                                                                                                
pip install -r requirements.txt`

Kullanılan temel bağımlılıklar:

OpenCV                                                                          
PyTorch                                                                                  
scikit-image                                                                            
SciPy

**CNN Modelinin Eğitilmesi**

Parmak izi benzerlik modelini eğitmek için:

`cd cnn
python train.py`

Bu işlem sonunda bir model dosyası üretilir (örneğin siamese_fingerprint.pth).


**Not:** Model ağırlıkları depoya dahil edilmemiştir.

**Tanıma Sisteminin Çalıştırılması**

Tüm tanıma hattını çalıştırmak için:

`python main.py`

Çıktı olarak şunlar üretilir:
Canlılık skoru,
kayıtlı her kişi için benzerlik skorları,
nihai karar (Kabul / Belirsiz / Reddedildi),
eşleşen minutiae noktalarının görselleştirilmesi.

**Nasıl Çalışır? (Teknik Özet)                                                                                                           
Minutiae Eşleştirme**

Minutiae noktaları çıkarılır ve filtrelenir. Test ve referans parmak izleri arasında eşleşen nokta çiftleri bulunur. En güçlü eşleşmeler yapısal benzerliği gösterir.

**Siamese CNN**

Parmak izi görüntü çiftleri öğrenilmiş bir uzaya gömülür (embedding).
Benzerlik şu şekilde hesaplanır:

`score = 1 / (1 + euclidean_distance)`

**Skor Birleştirme ve Karar**

Nihai karar şu unsurlara dayanır:
CNN ve yapısal skorların ağırlıklı birleşimi,
geçerli kimlik için eşik değeri,
belirsiz kararları önlemek için skor farkı marjı.
Bu tasarım, öğrenilmiş örüntüler ile yapısal özellikler arasında denge kurar.

**Kullanım Alanları**

Biyometrik kimlik doğrulama araştırmaları,
hibrit eşleştirme sistemleri için akademik demonstrasyon,
parmak izi canlılık analizi,
eşleşme süreçlerinin görsel anlatımı.

**Sınırlamalar**

Veri kümesi küçük olduğu için model doğruluğu sınırlıdır.
Yalnızca CPU üzerinde çalışır — GPU’ya göre daha yavaştır.
Üretim ortamlarında kullanılmak üzere tasarlanmamıştır.
Bu proje, öğrenme, deney yapma ve prototipleme amaçlıdır.

**Atıf ve Kaynaklar**

Bu proje, biyometrik sistemlerde kullanılan hibrit yaklaşımlardan ve CNN Explainer gibi etkileşimli CNN görselleştirme araçlarından ilham almıştır.

**Katkı**

Katkıda bulunmak isterseniz issue açabilir veya pull request gönderebilirsiniz.
Büyük değişiklikler için lütfen önce tartışma başlatın.

**İletişim**

Bu proje hakkında sorularınız varsa issue açabilir veya proje sahibiyle iletişime geçebilirsiniz.
CPU ile çalışır.
Üretim için hazır değildir.
