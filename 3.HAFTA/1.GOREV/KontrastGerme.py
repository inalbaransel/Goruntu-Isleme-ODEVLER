import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Dosya yollarını script'in bulunduğu klasöre göre ayarlayalım
KLASOR = os.path.dirname(os.path.abspath(__file__))
RESIM_YOLU = os.path.join(KLASOR, "dusuk_kontrast.jpg")
CIKTI_GRAFIK = os.path.join(KLASOR, "kontrast_sonuc.png")

# 0. Test için düşük kontrastlı resim yoksa otomatik oluştur/indir
def ornek_resim_hazirla():
    if not os.path.exists(RESIM_YOLU):
        print("Düşük kontrastlı test resmi bulunamadı, otomatik hazırlanıyor...")
        url = "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/camera.png"
        istek = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(istek, timeout=15) as cevap:
            veri = cevap.read()
        orijinal = cv2.imdecode(np.frombuffer(veri, np.uint8), cv2.IMREAD_GRAYSCALE)
        # Pikselleri [70, 150] aralığına sıkıştırarak düşük kontrast elde ediyoruz
        dusuk_kontrast = (orijinal.astype(np.float32) / 255.0 * (150 - 70) + 70).astype(np.uint8)
        cv2.imwrite(RESIM_YOLU, dusuk_kontrast)
        print(f"Örnek resim oluşturuldu: {RESIM_YOLU}")

ornek_resim_hazirla()

# 1. Düşük kontrastlı görüntüyü tek kanallı (GRAYSCALE) olarak oku
resim = cv2.imread(RESIM_YOLU, cv2.IMREAD_GRAYSCALE)

if resim is None:
    raise FileNotFoundError(f"Resim açılamadı: {RESIM_YOLU}")

# 2. Minimum ve maksimum değerleri bul (Döngü ile)
min_val = 255
max_val = 0
yukseklik, genislik = resim.shape

for y in range(yukseklik):
    for x in range(genislik):
        piksel = resim[y, x]
        if piksel < min_val:
            min_val = piksel
        if piksel > max_val:
            max_val = piksel

print("==================================================")
print(f"Orijinal Görüntü -> Min: {min_val}, Max: {max_val}")

# 3. Doğrusal Ölçekleme (Hazır cv2.normalize KULLANMADAN)
# Taşmaları (overflow/underflow) önlemek için float32'ye çeviriyoruz
resim_float = resim.astype(np.float32)

if max_val > min_val:  # Sıfıra bölme hatasını önlemek için kontrol
    # Formül: ((I - min) / (max - min)) * 255
    olcekli_float = ((resim_float - min_val) / (max_val - min_val)) * 255.0
    # En yakın tamsayıya yuvarlayıp tekrar [0, 255] uint8 formatına dönüştürüyoruz
    resim_olcekli = np.clip(np.round(olcekli_float), 0, 255).astype(np.uint8)
else:
    resim_olcekli = resim.copy()

print(f"Ölçeklenmiş Görüntü -> Min: {resim_olcekli.min()}, Max: {resim_olcekli.max()}")
print("==================================================")

# 4. Sonuçları Karşılaştır (Görüntüler ve Histogramlar)
plt.figure(figsize=(11, 7))

# Sol Üst: Orijinal Düşük Kontrastlı Resim
plt.subplot(2, 2, 1)
plt.imshow(resim, cmap='gray', vmin=0, vmax=255)
plt.title(f"Düşük Kontrast (Min: {min_val}, Max: {max_val})", fontsize=11)
plt.axis('off')

# Sağ Üst: Orijinal Histogram (Dar aralık)
plt.subplot(2, 2, 2)
plt.hist(resim.ravel(), bins=256, range=[0, 256], color='#555555')
plt.title("Orijinal Histogram (Dar Aralık)", fontsize=11)
plt.xlim([0, 256])
plt.grid(True, linestyle='--', alpha=0.5)

# Sol Alt: Ölçeklenmiş (Kontrastı Gerilmiş) Resim
plt.subplot(2, 2, 3)
plt.imshow(resim_olcekli, cmap='gray', vmin=0, vmax=255)
plt.title(f"Ölçeklenmiş Görüntü (Min: {resim_olcekli.min()}, Max: {resim_olcekli.max()})", fontsize=11)
plt.axis('off')

# Sağ Alt: Genişletilmiş Histogram [0, 255]
plt.subplot(2, 2, 4)
plt.hist(resim_olcekli.ravel(), bins=256, range=[0, 256], color='#1f77b4')
plt.title("Ölçeklenmiş Histogram [0, 255] (Tüm Aralığa Yayılmış)", fontsize=11)
plt.xlim([0, 256])
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

# Sonucu görsel dosya olarak kaydet
plt.savefig(CIKTI_GRAFIK, dpi=150)
print(f"Grafik karşılaştırması kaydedildi: {CIKTI_GRAFIK}")

# Ekranda pencere olarak aç
plt.show()
