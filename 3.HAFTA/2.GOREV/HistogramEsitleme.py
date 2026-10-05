"""
Görüntü İşleme - 3. Hafta - Görev 2: Histogram Eşitleme (Histogram Equalization)
-----------------------------------------------------------------------------
Bu program; tek kanallı ve düşük kontrastlı bir görüntüyü okur,
HİÇBİR HAZIR FONKSİYON KULLANMADAN:
  1) 256-değerli parlaklık histogramını çıkarır.
  2) Kümülatif Dağılım Fonksiyonunu (CDF - Cumulative Distribution Function) hesaplar.
  3) Histogram eşitleme dönüşümünü uygulayarak kontrastı dinamik aralığa yayar.
  4) Eşitlenmiş çıktı resmini diske kaydeder.
  5) Karşılaştırma grafiklerini (Görüntüler, Histogramlar ve CDF) çizer.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Dosya Yolları
# ---------------------------------------------------------------------------
KLASOR = os.path.dirname(os.path.abspath(__file__))
GIRDI_RESIM = os.path.join(KLASOR, "dusuk_kontrast.jpg")
CIKTI_RESIM = os.path.join(KLASOR, "esitlenmis_resim.jpg")
CIKTI_GRAFIK = os.path.join(KLASOR, "histogram_esitleme_karsilastirma.png")


# ---------------------------------------------------------------------------
# 1) Histogram Çıkarma (Hazır Fonksiyon Olmadan)
# ---------------------------------------------------------------------------
def histogram_hesapla(resim):
    """
    Resimdeki her bir gri tonun (0-255) kaç defa geçtiğini sayar.
    Hazır fonksiyon (np.histogram veya cv2.calcHist) KULLANILMAZ.
    """
    hist = [0] * 256
    yukseklik, genislik = resim.shape
    
    for y in range(yukseklik):
        for x in range(genislik):
            piksel_degeri = resim[y, x]
            hist[piksel_degeri] += 1
            
    return hist


# ---------------------------------------------------------------------------
# 2) Kümülatif Dağılım Fonksiyonu (CDF) Hesabı (Hazır Fonksiyon Olmadan)
# ---------------------------------------------------------------------------
def cdf_hesapla(hist):
    """
    Histogram değerlerini kümülatif (birikimli) olarak toplar.
    CDF[k] = 0'dan k'ya kadar olan piksellerin toplam sayısıdır.
    Hazır fonksiyon (np.cumsum) KULLANILMAZ.
    """
    cdf = [0] * 256
    toplam = 0
    for i in range(256):
        toplam += hist[i]
        cdf[i] = toplam
    return cdf


# ---------------------------------------------------------------------------
# 3) Histogram Eşitleme (Hazır cv2.equalizeHist KULLANMADAN)
# ---------------------------------------------------------------------------
def histogram_esitle(resim):
    """
    CDF değerlerini kullanarak pikselleri [0, 255] aralığına dengeli yayar.
    
    Formül:
        s_k = round( ((CDF[k] - CDF_min) / (Toplam_Piksel - CDF_min)) * 255 )
    """
    yukseklik, genislik = resim.shape
    toplam_piksel = yukseklik * genislik
    
    # 1. Histogramı hesapla
    hist = histogram_hesapla(resim)
    
    # 2. CDF'i hesapla
    cdf = cdf_hesapla(hist)
    
    # Sıfır olmayan ilk kümülatif değeri bul (en koyu pikselin başladığı yer)
    cdf_min = None
    for val in cdf:
        if val > 0:
            cdf_min = val
            break
            
    # Eğer resim tamamen tek bir renkten oluşuyorsa bölme hatası olmasın
    if cdf_min is None or cdf_min == toplam_piksel:
        return resim.copy(), hist, cdf, hist, cdf

    # 3. Dönüşüm Tablosu (Lookup Table - LUT) Hazırlama
    # 0'dan 255'e kadar her gri tonun yeni değerini önceden hesaplıyoruz
    lut = [0] * 256
    for i in range(256):
        if cdf[i] < cdf_min:
            lut[i] = 0
        else:
            yeni_deger = round(((cdf[i] - cdf_min) / (toplam_piksel - cdf_min)) * 255)
            lut[i] = min(max(yeni_deger, 0), 255)  # [0, 255] sınırla

    # 4. Resmi dönüştürme (Piksel piksel yeni değerleri atama)
    esitlenmis_resim = np.zeros((yukseklik, genislik), dtype=np.uint8)
    for y in range(yukseklik):
        for x in range(genislik):
            esitlenmis_resim[y, x] = lut[resim[y, x]]
            
    # Sonucun histogram ve CDF'ini de analiz için çıkaralım
    yeni_hist = histogram_hesapla(esitlenmis_resim)
    yeni_cdf = cdf_hesapla(yeni_hist)
    
    return esitlenmis_resim, hist, cdf, yeni_hist, yeni_cdf


# ---------------------------------------------------------------------------
# Ana Akış
# ---------------------------------------------------------------------------
def main():
    print("==================================================")
    print("GÖREV 2: Histogram Eşitleme (Histogram Equalization)")
    print("==================================================")

    # 1. Görüntüyü OpenCV ile Gri Seviye (Tek Kanallı) Aç
    if not os.path.exists(GIRDI_RESIM):
        raise FileNotFoundError(f"Girdi resmi bulunamadı: {GIRDI_RESIM}")

    resim = cv2.imread(GIRDI_RESIM, cv2.IMREAD_GRAYSCALE)
    print(f"Girdi Resmi Yüklendi: {GIRDI_RESIM}")
    print(f"Çözünürlük: {resim.shape[1]}x{resim.shape[0]} piksel | Min: {resim.min()}, Max: {resim.max()}")

    # 2. Histogram Eşitlemeyi Uygula
    esitlenmis, hist_eski, cdf_eski, hist_yeni, cdf_yeni = histogram_esitle(resim)
    print(f"Histogram Eşitleme Tamamlandı -> Yeni Min: {esitlenmis.min()}, Yeni Max: {esitlenmis.max()}")

    # 3. Çıktı Resmini Kaydet
    cv2.imwrite(CIKTI_RESIM, esitlenmis)
    print(f"Çıktı resmi kaydedildi: {CIKTI_RESIM}")

    # 4. Sonuçları Görselleştir ve Karşılaştır
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # Orijinal Resim
    axes[0, 0].imshow(resim, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title(f"Orijinal Düşük Kontrast (Min: {resim.min()}, Max: {resim.max()})")
    axes[0, 0].axis('off')

    # Eşitlenmiş Resim
    axes[0, 1].imshow(esitlenmis, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title(f"Histogram Eşitlenmiş Resim (Min: {esitlenmis.min()}, Max: {esitlenmis.max()})")
    axes[0, 1].axis('off')

    # Orijinal Histogram
    axes[1, 0].bar(range(256), hist_eski, color='#555555', width=1.0)
    axes[1, 0].set_title("Orijinal Histogram (Dar ve Yığılmış)")
    axes[1, 0].set_xlim([0, 255])
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    # Eşitlenmiş Histogram
    axes[1, 1].bar(range(256), hist_yeni, color='#1f77b4', width=1.0)
    axes[1, 1].set_title("Eşitlenmiş Histogram (Düzgün Yayılmış)")
    axes[1, 1].set_xlim([0, 255])
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    # Orijinal CDF
    toplam_piksel = resim.shape[0] * resim.shape[1]
    norm_cdf_eski = [c / toplam_piksel for c in cdf_eski]
    axes[2, 0].plot(range(256), norm_cdf_eski, color='red', linewidth=2)
    axes[2, 0].set_title("Orijinal Kümülatif Dağılım (CDF) - Dik ve Dar")
    axes[2, 0].set_xlim([0, 255])
    axes[2, 0].set_ylim([0, 1.05])
    axes[2, 0].grid(True, linestyle='--', alpha=0.5)

    # Eşitlenmiş CDF
    norm_cdf_yeni = [c / toplam_piksel for c in cdf_yeni]
    axes[2, 1].plot(range(256), norm_cdf_yeni, color='green', linewidth=2)
    axes[2, 1].set_title("Eşitlenmiş Kümülatif Dağılım (CDF) - Lineer / Doğrusal")
    axes[2, 1].set_xlim([0, 255])
    axes[2, 1].set_ylim([0, 1.05])
    axes[2, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(CIKTI_GRAFIK, dpi=150)
    print(f"Karşılaştırma grafiği kaydedildi: {CIKTI_GRAFIK}")
    print("Grafik penceresi açılıyor (kapatınca program sonlanır)...")
    plt.show()


if __name__ == "__main__":
    main()
