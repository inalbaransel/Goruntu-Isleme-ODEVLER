# Görüntü İşleme Ders Ödevleri

Bu depo, Görüntü İşleme dersi kapsamında geliştirilen haftalık uygulamaları ve ödevleri içermektedir.

---

## Hafta İçerikleri

### 1. Hafta: Çözünürlük ve Boyutlandırma
* Görüntü matrisleri ve çözünürlük kavramları.
* Ekran çözünürlüğü ve piksel matrisi manipülasyonları.

### 2. Hafta: Parlaklık (Gri Seviye) Çözünürlüğü ve Histogram
* Görüntü nicemleme (quantization - 128, 64, 32, 16, 8, 4, 2 seviye).
* Maksimum parlaklık seviyeleri ve dinamik aralık.
* 4 parçalı görüntü analizi ve parça bazlı histogram hesaplama.
* Satır satır vs sütun sütun bellek erişim performansı analizi.

### 3. Hafta: Kontrast Germe ve Histogram Eşitleme
* **Görev 1: Kontrast Germe (Doğrusal Min-Max Ölçekleme):**
  * Düşük kontrastlı gri seviye bir görüntünün minimum ve maksimum değerleri bulunur.
  * Hazır fonksiyon kullanılmadan $I_{yeni} = \frac{I - I_{min}}{I_{max} - I_{min}} \times 255$ formülüyle $[0, 255]$ aralığına doğrusal ölçeklenir.
* **Görev 2: Histogram Eşitleme (Histogram Equalization):**
  * Hazır fonksiyon kullanılmadan 256-değerli parlaklık histogramı ($h(r)$) çıkarılır.
  * Kümülatif Dağılım Fonksiyonu (CDF - Cumulative Distribution Function) hesaplanır.
  * CDF tabanlı dönüşüm formülü uygulanarak histogram dengelenir ve çıktı kaydedilir.
* **Görev 3: CLAHE (Contrast Limited Adaptive Histogram Equalization):**
  * OpenCV hazır `cv::createCLAHE` fonksiyonu ile görüntü iyileştirilir.
  * Sıfırdan C++ ile 8x8 karo (tile) tabanlı, kontrast sınırlandırmalı (clipping) ve çift doğrusal enterpolasyonlu (bilinear interpolation) CLAHE algoritması kodlanır ve karşılaştırılır.

---

## Çalıştırma

Gereksinimler:
```bash
pip install opencv-python numpy matplotlib
```

Örnek çalıştırma:
```bash
# 3. Hafta - Görev 1
python3 "3.HAFTA/1.GOREV/KontrastGerme.py"

# 3. Hafta - Görev 2
python3 "3.HAFTA/2.GOREV/HistogramEsitleme.py"

# 3. Hafta - Görev 3 (C++)
cd "3.HAFTA/3.GOREV"
cmake . && make
./clahe
```
