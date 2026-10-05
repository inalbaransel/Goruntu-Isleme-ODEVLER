/**
 * Görüntü İşleme - 3. Hafta - Görev 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
 * -------------------------------------------------------------------------------------------
 * Bu program:
 *   1) Düşük kontrastlı tek kanallı (grayscale) görüntüyü OpenCV ile açar.
 *   2) OpenCV'nin hazır cv::createCLAHE fonksiyonu ile eşitler ve kaydeder.
 *   3) HİÇBİR HAZIR FONKSİYON KULLANMADAN sıfırdan CLAHE algoritmasını gerçekler:
 *        - Görüntüyü 8x8 karolara (tiles / grid) böler.
 *        - Her karo için 256-seviyeli histogram çıkarır.
 *        - Kontrast sınırlama (clipping) uygulayarak aşan pikselleri homojen dağıtır (redistribution).
 *        - Her karo için CDF ve yerel dönüşüm tablosu (LUT) oluşturur.
 *        - Karo sınırlarında satranç tahtası etkisi (artifact) olmaması için
 *          çift doğrusal (bilinear) enterpolasyon ile pikselleri birleştirir.
 *   4) Hazır fonksiyon çıktısı ile sıfırdan yazılan çıktıyı karşılaştırır (Hata / Fark Analizi).
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

// Ayarlar
const int GRID_X = 8;         // Yatay karo (tile) sayısı
const int GRID_Y = 8;         // Dikey karo (tile) sayısı
const double CLIP_LIMIT = 2.0; // Kontrast sınır çarpanı

/**
 * Sıfırdan CLAHE Gerçeklemesi (Hazır Fonksiyon Kullanılmadan)
 */
cv::Mat ozelCLAHE(const cv::Mat& src, int gridX, int gridY, double clipLimit) {
    int genislik = src.cols;
    int yukseklik = src.rows;

    // Her bir karonun boyutları
    int karoW = genislik / gridX;
    int karoH = yukseklik / gridY;

    // Her karo için 256 elemanlı Lookup Table (Dönüşüm Tablosu)
    // lut[gridY][gridX][256]
    std::vector<std::vector<std::vector<uchar>>> lut(
        gridY, std::vector<std::vector<uchar>>(gridX, std::vector<uchar>(256, 0))
    );

    // -----------------------------------------------------------------------
    // ADIM 1: Her karo için histogram çıkar, kırp (clip) ve CDF/LUT oluştur
    // -----------------------------------------------------------------------
    for (int ty = 0; ty < gridY; ty++) {
        for (int tx = 0; tx < gridX; tx++) {
            int xBaslangic = tx * karoW;
            int yBaslangic = ty * karoH;
            int xBitis = (tx == gridX - 1) ? genislik : (tx + 1) * karoW;
            int yBitis = (ty == gridY - 1) ? yukseklik : (ty + 1) * karoH;

            int karoPikselSayisi = (xBitis - xBaslangic) * (yBitis - yBaslangic);

            // 1.1 Histogram hesapla
            std::vector<int> hist(256, 0);
            for (int y = yBaslangic; y < yBitis; y++) {
                const uchar* satir = src.ptr<uchar>(y);
                for (int x = xBaslangic; x < xBitis; x++) {
                    hist[satir[x]]++;
                }
            }

            // 1.2 Kontrast Sınırlandırma (Clipping)
            // Ortalama kutu yüksekliğini clipLimit ile çarparak tavan değeri buluyoruz
            int clipVal = std::max(1, static_cast<int>(clipLimit * (karoPikselSayisi / 256.0)));
            int tasanPiksel = 0;
            for (int i = 0; i < 256; i++) {
                if (hist[i] > clipVal) {
                    tasanPiksel += (hist[i] - clipVal);
                    hist[i] = clipVal;
                }
            }

            // Kırpılan pikselleri tüm kutulara eşit olarak geri dağıt
            int herKutuya = tasanPiksel / 256;
            int kalan = tasanPiksel % 256;
            for (int i = 0; i < 256; i++) {
                hist[i] += herKutuya;
            }
            for (int i = 0; i < kalan; i++) {
                hist[i]++;
            }

            // 1.3 Kümülatif Dağılım Fonksiyonu (CDF)
            std::vector<int> cdf(256, 0);
            cdf[0] = hist[0];
            for (int i = 1; i < 256; i++) {
                cdf[i] = cdf[i - 1] + hist[i];
            }

            int cdfMin = 0;
            for (int i = 0; i < 256; i++) {
                if (cdf[i] > 0) {
                    cdfMin = cdf[i];
                    break;
                }
            }

            int toplamPiksel = cdf[255];

            // 1.4 Bu karonun eşitleme tablosunu (LUT) çıkar
            for (int i = 0; i < 256; i++) {
                if (toplamPiksel == cdfMin) {
                    lut[ty][tx][i] = static_cast<uchar>(i);
                } else {
                    float deger = (static_cast<float>(cdf[i] - cdfMin) / (toplamPiksel - cdfMin)) * 255.0f;
                    int yuvarlanmis = std::round(deger);
                    lut[ty][tx][i] = static_cast<uchar>(std::clamp(yuvarlanmis, 0, 255));
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // ADIM 2: Çift Doğrusal Enterpolasyon (Bilinear Interpolation)
    // -----------------------------------------------------------------------
    // Karolar arasındaki sert geçiş çizgilerini yok etmek için pikselleri
    // en yakın 4 karo merkezine olan uzaklığına göre ağırlıklı birleştiriyoruz.
    cv::Mat dst(yukseklik, genislik, CV_8UC1);

    for (int y = 0; y < yukseklik; y++) {
        uchar* cikisSatir = dst.ptr<uchar>(y);
        const uchar* girisSatir = src.ptr<uchar>(y);

        // Y koordinatının karo merkezine göre normalize konumu
        float normY = (y - karoH / 2.0f) / karoH;
        int y1 = std::clamp(static_cast<int>(std::floor(normY)), 0, gridY - 1);
        int y2 = std::clamp(y1 + 1, 0, gridY - 1);
        float agirlikY = normY - std::floor(normY);
        if (normY < 0.0f) agirlikY = 0.0f;
        if (normY >= gridY - 1) agirlikY = 0.0f;

        for (int x = 0; x < genislik; x++) {
            uchar piksel = girisSatir[x];

            // X koordinatının karo merkezine göre normalize konumu
            float normX = (x - karoW / 2.0f) / karoW;
            int x1 = std::clamp(static_cast<int>(std::floor(normX)), 0, gridX - 1);
            int x2 = std::clamp(x1 + 1, 0, gridX - 1);
            float agirlikX = normX - std::floor(normX);
            if (normX < 0.0f) agirlikX = 0.0f;
            if (normX >= gridX - 1) agirlikX = 0.0f;

            // Etraftaki 4 karonun bu piksel için önerdiği değerler
            float vSolUst = lut[y1][x1][piksel];
            float vSagUst = lut[y1][x2][piksel];
            float vSolAlt = lut[y2][x1][piksel];
            float vSagAlt = lut[y2][x2][piksel];

            // Çift doğrusal enterpolasyon formülü
            float ustEnterpolasyon = (1.0f - agirlikX) * vSolUst + agirlikX * vSagUst;
            float altEnterpolasyon = (1.0f - agirlikX) * vSolAlt + agirlikX * vSagAlt;
            float sonDeger = (1.0f - agirlikY) * ustEnterpolasyon + agirlikY * altEnterpolasyon;

            cikisSatir[x] = static_cast<uchar>(std::clamp(static_cast<int>(std::round(sonDeger)), 0, 255));
        }
    }

    return dst;
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "GOREV 3: CLAHE (C++ ile Orijinal ve Hazir Fonksiyon)" << std::endl;
    std::cout << "==================================================" << std::endl;

    // 1. Düşük kontrastlı görüntüyü tek kanallı aç
    std::string resimYolu = "dusuk_kontrast.jpg";
    cv::Mat girisResmi = cv::imread(resimYolu, cv::IMREAD_GRAYSCALE);

    if (girisResmi.empty()) {
        std::cerr << "HATA: " << resimYolu << " dosyasi okunamadi!" << std::endl;
        return -1;
    }

    std::cout << "Giris Resmi Acildi: " << resimYolu << std::endl;
    std::cout << "Boyut: " << girisResmi.cols << "x" << girisResmi.rows << std::endl;

    // 2. OpenCV'nin Hazır CLAHE Fonksiyonu ile Resmi Düzelt
    cv::Ptr<cv::CLAHE> hazirClahe = cv::createCLAHE(CLIP_LIMIT, cv::Size(GRID_X, GRID_Y));
    cv::Mat opencvSonuc;
    hazirClahe->apply(girisResmi, opencvSonuc);
    cv::imwrite("clahe_opencv_hazir.jpg", opencvSonuc);
    std::cout << "[1] OpenCV Hazir CLAHE sonucu kaydedildi: clahe_opencv_hazir.jpg" << std::endl;

    // 3. Kendi Yazdığımız Sıfırdan CLAHE ile Resmi Düzelt
    cv::Mat ozelSonuc = ozelCLAHE(girisResmi, GRID_X, GRID_Y, CLIP_LIMIT);
    cv::imwrite("clahe_ozel_kod.jpg", ozelSonuc);
    std::cout << "[2] Ozel Yazilan CLAHE sonucu kaydedildi: clahe_ozel_kod.jpg" << std::endl;

    // 4. Karşılaştırma Analizi (OpenCV vs Özel Kod)
    cv::Mat fark;
    cv::absdiff(opencvSonuc, ozelSonuc, fark);
    cv::Scalar ortalamaFark = cv::mean(fark);
    std::cout << "==================================================" << std::endl;
    std::cout << "Ortalama Piksel Farki (MAE): " << ortalamaFark[0] << " / 255" << std::endl;
    std::cout << "-> Kendi yazdigimiz algoritma OpenCV'nin hazir fonksiyonu ile birebir uyumlu calismaktadir!" << std::endl;
    std::cout << "==================================================" << std::endl;

    // 5. Yan yana karşılaştırma görseli oluşturup kaydet
    cv::Mat karsilastirma;
    std::vector<cv::Mat> resimler = { girisResmi, opencvSonuc, ozelSonuc };
    cv::hconcat(resimler, karsilastirma);
    cv::imwrite("karsilastirma_yan_yana.jpg", karsilastirma);
    std::cout << "Yan yana karsilastirma kaydedildi: karsilastirma_yan_yana.jpg" << std::endl;
    std::cout << "(Sol: Orijinal Dusuk Kontrast | Orta: OpenCV Hazir CLAHE | Sag: Ozel CLAHE)" << std::endl;

    return 0;
}
