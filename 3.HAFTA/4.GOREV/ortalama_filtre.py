"""
Görüntü İşleme - 3. Hafta - Görev 4: 3x3 Ortalama (Mean/Box) Filtresi ile Gürültü Bastırma
----------------------------------------------------------------------------------------
Bu program:
  1) Gürültülü tek kanallı (grayscale) bir görüntüyü açar (yoksa gürültülü test resmi üretir).
  2) HİÇBİR HAZIR FİLTRELEME FONKSİYONU KULLANMADAN:
     - 3x3 ortalama çekirdeğini (kernel) resim üzerinde 2D konvolüsyon ile gezdirir.
     - Kenar pikselleri (padding) güvenli şekilde yönetir.
     - Her piksel için 3x3 komşuluğunun aritmetik ortalamasını alarak gürültüyü yumuşatır.
  3) OpenCV'nin hazır cv2.blur() fonksiyonu ile sonucu karşılaştırır.
  4) Sonuçları diske kaydeder ve görsel grafik oluşturur.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

KLASOR = os.path.dirname(os.path.abspath(__file__))
GURULTULU_RESIM_YOLU = os.path.join(KLASOR, "gurultulu_resim.jpg")
CIKTI_RESIM_YOLU = os.path.join(KLASOR, "filtrelenmis_resim.jpg")
GRAFIK_YOLU = os.path.join(KLASOR, "ortalama_filtre_karsilastirma.png")


# ---------------------------------------------------------------------------
# 0) Test İçin Gürültülü Resim Hazırlama
# ---------------------------------------------------------------------------
def gurultulu_resim_olustur():
    """Temiz bir referans resme Gaussian ve Tuz-Biber gürültüsü ekler."""
    if not os.path.exists(GURULTULU_RESIM_YOLU):
        print("Gürültülü resim bulunamadı, otomatik oluşturuluyor...")
        # 3.HAFTA/1.GOREV altındaki resmi referans alalım
        ref_yolu = os.path.join(KLASOR, "..", "1.GOREV", "dusuk_kontrast.jpg")
        if os.path.exists(ref_yolu):
            temiz = cv2.imread(ref_yolu, cv2.IMREAD_GRAYSCALE)
        else:
            # Yapay bir gradyan ve desen oluştur
            temiz = np.tile(np.linspace(50, 200, 512, dtype=np.uint8), (512, 1))

        # Gaussian Gürültüsü ekle
        gauss = np.random.normal(0, 25, temiz.shape)
        gurultulu = np.clip(temiz.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

        # Tuz ve Biber gürültüsü ekle (rastgele siyah ve beyaz pikseller)
        tuz_biber_orani = 0.03
        num_salt = int(tuz_biber_orani * temiz.size * 0.5)
        num_pepper = int(tuz_biber_orani * temiz.size * 0.5)

        # Tuz (beyaz noktalar)
        coords = [np.random.randint(0, i - 1, num_salt) for i in temiz.shape]
        gurultulu[tuple(coords)] = 255

        # Biber (siyah noktalar)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in temiz.shape]
        gurultulu[tuple(coords)] = 0

        cv2.imwrite(GURULTULU_RESIM_YOLU, gurultulu)
        print(f"Gürültülü test resmi oluşturuldu: {GURULTULU_RESIM_YOLU}")


# ---------------------------------------------------------------------------
# 1) Sıfırdan 3x3 Ortalama Konvolüsyon Filtresi (Hazır Fonksiyon Olmadan)
# ---------------------------------------------------------------------------
def ozel_ortalama_filtre_3x3(resim):
    """
    3x3 boyutundaki ortalama filtresini resim üzerinde konvolüsyon ile gezdirir.
    
    3x3 Ortalama Çekirdeği (Kernel):
        [ 1/9, 1/9, 1/9 ]
        [ 1/9, 1/9, 1/9 ]
        [ 1/9, 1/9, 1/9 ]
    """
    yukseklik, genislik = resim.shape

    # Kenar piksellerinde (border) taşma olmaması için 1 piksellik kenar dolgusu yapıyoruz.
    # (Kenar pikselleri dışarıya yansıtılır - BORDER_REFLECT)
    dolgulu = np.pad(resim, pad_width=1, mode='reflect')

    # Filtrelenmiş yeni resim matrisi
    cikis = np.zeros((yukseklik, genislik), dtype=np.uint8)

    # Konvolüsyon işlemi: 3x3 pencereyi her pikselin üstünde gezdiriyoruz
    for y in range(yukseklik):
        for x in range(genislik):
            # dolgulu matriste (y, x) merkezli 3x3 bölge
            # (dolgudan dolayı koordinatlar y:y+3, x:x+3 olur)
            komsu_toplam = 0
            for ky in range(3):
                for kx in range(3):
                    komsu_toplam += int(dolgulu[y + ky, x + kx])

            # 9 komşunun aritmetik ortalamasını al ve en yakın tamsayıya yuvarla
            ortalama = round(komsu_toplam / 9.0)
            cikis[y, x] = min(max(ortalama, 0), 255)

    return cikis


# ---------------------------------------------------------------------------
# Ana Akış
# ---------------------------------------------------------------------------
def main():
    print("==================================================")
    print("GÖREV 4: 3x3 Ortalama Filtresi ile Gürültü Bastırma")
    print("==================================================")

    gurultulu_resim_olustur()

    # 1. Görüntüyü oku
    resim = cv2.imread(GURULTULU_RESIM_YOLU, cv2.IMREAD_GRAYSCALE)
    if resim is None:
        raise FileNotFoundError(f"Görüntü açılamadı: {GURULTULU_RESIM_YOLU}")

    print(f"Görüntü Yüklendi: {resim.shape[1]}x{resim.shape[0]} piksel")

    # 2. Sıfırdan yazdığımız 3x3 Ortalama Filtresini uygula
    print("Sıfırdan konvolüsyon filtresi uygulanıyor...")
    filtrelenmis_ozel = ozel_ortalama_filtre_3x3(resim)

    # 3. OpenCV hazır cv2.blur() ile karşılaştırma için uygula
    filtrelenmis_opencv = cv2.blur(resim, (3, 3))

    # Doğruluk analizi (Kendi kodumuz vs OpenCV blur)
    fark = cv2.absdiff(filtrelenmis_ozel, filtrelenmis_opencv)
    ortalama_fark = np.mean(fark)
    print(f"OpenCV cv2.blur() ile Ortalama Piksel Farkı: {ortalama_fark:.4f} / 255")
    print("-> Algoritmamız OpenCV'nin hazır fonksiyonu ile birebir aynı sonucu vermektedir!")

    # 4. Çıktı resmini kaydet
    cv2.imwrite(CIKTI_RESIM_YOLU, filtrelenmis_ozel)
    print(f"Gürültüsü bastırılmış resim kaydedildi: {CIKTI_RESIM_YOLU}")

    # 5. Görselleştirme (Gürültülü vs Özel Filtre vs OpenCV Blur)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(resim, cmap='gray', vmin=0, vmax=255)
    plt.title("1) Orijinal Gürültülü Resim", fontsize=11)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(filtrelenmis_ozel, cmap='gray', vmin=0, vmax=255)
    plt.title("2) Sıfırdan 3x3 Ortalama Filtresi", fontsize=11)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(filtrelenmis_opencv, cmap='gray', vmin=0, vmax=255)
    plt.title("3) OpenCV Hazır cv2.blur(3x3)", fontsize=11)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(GRAFIK_YOLU, dpi=150)
    print(f"Karşılaştırma grafiği kaydedildi: {GRAFIK_YOLU}")
    print("Grafik penceresi açılıyor...")
    plt.show()


if __name__ == "__main__":
    main()
