"""
Görüntü İşleme - 3. Hafta - Görev 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
-------------------------------------------------------------------------------------------
C++ versiyonu: clahe.cpp
Bu Python scripti de aynı algoritmanın Python karşılığıdır.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

KLASOR = os.path.dirname(os.path.abspath(__file__))
RESIM_YOLU = os.path.join(KLASOR, "dusuk_kontrast.jpg")

def ozel_clahe(src, grid_x=8, grid_y=8, clip_limit=2.0):
    yukseklik, genislik = src.shape
    karo_w = genislik // grid_x
    karo_h = yukseklik // grid_y

    # 1. Her karo için Histogram, Kırpma (Clipping) ve LUT oluştur
    lut = np.zeros((grid_y, grid_x, 256), dtype=np.uint8)

    for ty in range(grid_y):
        for tx in range(grid_x):
            x1 = tx * karo_w
            y1 = ty * karo_h
            x2 = genislik if tx == grid_x - 1 else (tx + 1) * karo_w
            y2 = yukseklik if ty == grid_y - 1 else (ty + 1) * karo_h

            karo = src[y1:y2, x1:x2]
            karo_piksel_sayisi = karo.size

            # Histogram
            hist = np.zeros(256, dtype=int)
            for p in karo.ravel():
                hist[p] += 1

            # Kontrast Kırpma (Clipping)
            clip_val = max(1, int(clip_limit * (karo_piksel_sayisi / 256.0)))
            tasan = 0
            for i in range(256):
                if hist[i] > clip_val:
                    tasan += (hist[i] - clip_val)
                    hist[i] = clip_val

            # Kırpılan pikselleri eşit dağıt
            hist += tasan // 256
            hist[:tasan % 256] += 1

            # CDF
            cdf = np.cumsum(hist)
            cdf_min = cdf[cdf > 0][0]
            toplam = cdf[-1]

            if toplam == cdf_min:
                lut[ty, tx] = np.arange(256, dtype=np.uint8)
            else:
                degerler = np.round(((cdf - cdf_min) / (toplam - cdf_min)) * 255.0)
                lut[ty, tx] = np.clip(degerler, 0, 255).astype(np.uint8)

    # 2. Çift Doğrusal Enterpolasyon (Bilinear Interpolation)
    dst = np.zeros_like(src)

    for y in range(yukseklik):
        norm_y = (y - karo_h / 2.0) / karo_h
        y1 = int(np.clip(np.floor(norm_y), 0, grid_y - 1))
        y2 = int(np.clip(y1 + 1, 0, grid_y - 1))
        wy = norm_y - np.floor(norm_y)
        if norm_y < 0.0 or norm_y >= grid_y - 1:
            wy = 0.0

        for x in range(genislik):
            piksel = src[y, x]

            norm_x = (x - karo_w / 2.0) / karo_w
            x1 = int(np.clip(np.floor(norm_x), 0, grid_x - 1))
            x2 = int(np.clip(x1 + 1, 0, grid_x - 1))
            wx = norm_x - np.floor(norm_x)
            if norm_x < 0.0 or norm_x >= grid_x - 1:
                wx = 0.0

            v11 = lut[y1, x1, piksel]
            v12 = lut[y1, x2, piksel]
            v21 = lut[y2, x1, piksel]
            v22 = lut[y2, x2, piksel]

            ust = (1.0 - wx) * v11 + wx * v12
            alt = (1.0 - wx) * v21 + wx * v22
            dst[y, x] = int(np.clip(np.round((1.0 - wy) * ust + wy * alt), 0, 255))

    return dst

def main():
    print("==================================================")
    print("GÖREV 3: CLAHE (Python & C++ Karşılaştırma)")
    print("==================================================")
    
    src = cv2.imread(RESIM_YOLU, cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise FileNotFoundError(f"Resim bulunamadı: {RESIM_YOLU}")

    # 1. OpenCV Hazır CLAHE
    clahe_opencv = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    sonuc_opencv = clahe_opencv.apply(src)
    cv2.imwrite(os.path.join(KLASOR, "clahe_opencv_hazir.jpg"), sonuc_opencv)

    # 2. Özel CLAHE
    sonuc_ozel = ozel_clahe(src, grid_x=8, grid_y=8, clip_limit=2.0)
    cv2.imwrite(os.path.join(KLASOR, "clahe_ozel_kod.jpg"), sonuc_ozel)

    # 3. Görselleştirme
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(src, cmap='gray', vmin=0, vmax=255)
    plt.title("Orijinal Düşük Kontrast")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(sonuc_opencv, cmap='gray', vmin=0, vmax=255)
    plt.title("OpenCV Hazır CLAHE")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(sonuc_ozel, cmap='gray', vmin=0, vmax=255)
    plt.title("Özel Yazılan CLAHE")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(KLASOR, "clahe_karsilastirma.png"), dpi=150)
    print("Sonuçlar başarıyla kaydedildi!")
    plt.show()

if __name__ == "__main__":
    main()
