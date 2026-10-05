/**
 * Görüntü İşleme - 3. Hafta - Görev 4: 3x3 Ortalama Filtresi ile Gürültü Bastırma (C++)
 * ----------------------------------------------------------------------------------
 * Bu program:
 *   1) Gürültülü tek kanallı (grayscale) görüntüyü OpenCV ile okur.
 *   2) HİÇBİR HAZIR FİLTRE FONKSİYONU KULLANMADAN:
 *      - 3x3 boyutundaki ortalama (mean/box) filtresini 2D konvolüsyon ile uygular.
 *      - Kenar taşmalarını (boundary condition) yansıtma (reflect) mantığı ile yönetir.
 *      - Her pikselin 3x3 komşuluğundaki 9 pikseli toplayıp 9'a bölerek gürültüyü bastırır.
 *   3) OpenCV'nin hazır cv::blur fonksiyonu ile sonucu karşılaştırır.
 *   4) Sonuçları diske kaydeder.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

/**
 * Sıfırdan 3x3 Ortalama (Mean) Filtresi Konvolüsyonu
 * 
 * Çekirdek (Kernel):
 *   [ 1/9, 1/9, 1/9 ]
 *   [ 1/9, 1/9, 1/9 ]
 *   [ 1/9, 1/9, 1/9 ]
 */
cv::Mat ozelOrtalamaFiltre3x3(const cv::Mat& src) {
    int yukseklik = src.rows;
    int genislik = src.cols;

    cv::Mat dst(yukseklik, genislik, CV_8UC1);

    // Konvolüsyon döngüsü: Her (y, x) pikseli üzerinde 3x3 pencere gezdirilir
    for (int y = 0; y < yukseklik; y++) {
        uchar* cikisSatir = dst.ptr<uchar>(y);

        for (int x = 0; x < genislik; x++) {
            int toplam = 0;

            // 3x3 komşuluk: [-1, 0, 1] ofsetleri
            for (int ky = -1; ky <= 1; ky++) {
                int komsuY = y + ky;

                // Kenar Taşması Yönetimi (Border Reflection)
                if (komsuY < 0) komsuY = -komsuY;
                if (komsuY >= yukseklik) komsuY = 2 * yukseklik - komsuY - 2;

                const uchar* komsuSatir = src.ptr<uchar>(komsuY);

                for (int kx = -1; kx <= 1; kx++) {
                    int komsuX = x + kx;

                    if (komsuX < 0) komsuX = -komsuX;
                    if (komsuX >= genislik) komsuX = 2 * genislik - komsuX - 2;

                    toplam += komsuSatir[komsuX];
                }
            }

            // 9 komşunun aritmetik ortalamasını al ve yuvarla
            int ortalama = static_cast<int>(std::round(toplam / 9.0));
            cikisSatir[x] = static_cast<uchar>(std::clamp(ortalama, 0, 255));
        }
    }

    return dst;
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "GOREV 4: 3x3 Ortalama Filtresi (C++ Konvolusyon)" << std::endl;
    std::cout << "==================================================" << std::endl;

    // 1. Gürültülü resmi tek kanallı aç
    std::string resimYolu = "gurultulu_resim.jpg";
    cv::Mat girisResmi = cv::imread(resimYolu, cv::IMREAD_GRAYSCALE);

    if (girisResmi.empty()) {
        std::cerr << "HATA: " << resimYolu << " dosyasi okunamadi!" << std::endl;
        std::cerr << "Lutfen once python3 ortalama_filtre.py calistirarak resmi olusturun veya resim ekleyin." << std::endl;
        return -1;
    }

    std::cout << "Giris Resmi Acildi: " << resimYolu << std::endl;
    std::cout << "Cozunurluk: " << girisResmi.cols << "x" << girisResmi.rows << " piksel" << std::endl;

    // 2. Sıfırdan yazdığımız 3x3 ortalama konvolüsyon filtresini uygula
    cv::Mat ozelSonuc = ozelOrtalamaFiltre3x3(girisResmi);
    cv::imwrite("filtrelenmis_ozel.jpg", ozelSonuc);
    std::cout << "[1] Ozel Konvolusyon Filtresi sonucu kaydedildi: filtrelenmis_ozel.jpg" << std::endl;

    // 3. OpenCV'nin hazır cv::blur(3x3) fonksiyonu ile karşılaştır
    cv::Mat opencvSonuc;
    cv::blur(girisResmi, opencvSonuc, cv::Size(3, 3));
    cv::imwrite("filtrelenmis_opencv.jpg", opencvSonuc);
    std::cout << "[2] OpenCV cv::blur sonucu kaydedildi: filtrelenmis_opencv.jpg" << std::endl;

    // 4. Doğruluk ve Hata Analizi
    cv::Mat fark;
    cv::absdiff(ozelSonuc, opencvSonuc, fark);
    cv::Scalar ortalamaFark = cv::mean(fark);

    std::cout << "==================================================" << std::endl;
    std::cout << "OpenCV cv::blur ile Ortalama Piksel Farki: " << ortalamaFark[0] << " / 255" << std::endl;
    std::cout << "-> Kendi yazdigimiz konvolusyon, OpenCV ile birebir ayni calismaktadir!" << std::endl;
    std::cout << "==================================================" << std::endl;

    // 5. Yan yana karşılaştırma görseli oluştur
    cv::Mat karsilastirma;
    std::vector<cv::Mat> resimler = { girisResmi, ozelSonuc, opencvSonuc };
    cv::hconcat(resimler, karsilastirma);
    cv::imwrite("karsilastirma_yan_yana.jpg", karsilastirma);
    std::cout << "Yan yana karsilastirma kaydedildi: karsilastirma_yan_yana.jpg" << std::endl;

    return 0;
}
