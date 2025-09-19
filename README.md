# Bodt-Fat
Yağ Oranı Hesaplama Yöntemleri

Proje Amacı:
Bu proje, bireylerin yağ oranını tahmin etmek için farklı istatistiksel ve makine öğrenmesi yöntemlerini karşılaştırır.
Amaç, sporcular ve sağlıklı yaşam hedefleyen bireyler için hızlı ve güvenilir bir yağ oranı tahmin sistemi geliştirmektir.

Kullanılan Yöntemler:
- Lineer Regresyon (en iyi performansı veren model)
- Ridge, Lasso, Elastic Net
- KNN ve diğer karşılaştırma modelleri

Veri ve Özellikler:
Kullanıcıdan alınan veriler:
- Boy
- Kilo
- Yaş
- Cinsiyet
- Bel / kalça ölçüleri (opsiyonel)
- Çıktı: Tahmini yağ oranı (%)

Özellikler:
Kullanıcı girdilerine göre otomatik tahmin
Sonuçlar:
- En iyi model: Lineer Regresyon
- Başarı metriği (R²): ~0.85
- RMSE: düşük hata oranı
- Modellerin performans karşılaştırmaları

* En uygun modelin seçilmesi (Lineer Regresyon)
