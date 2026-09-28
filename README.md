# DisasterVision AI: Yapay Zeka Destekli Otonom Hasar ve İnsan Tespit Sistemi

Bu proje, afet bölgelerinde (deprem vb.) havadan (İHA/drone) çekilen görüntüler üzerinden bina hasar durumunu tespit etmek ve enkaz altındaki/çevresindeki insan varlığını algılamak amacıyla geliştirilmiş bir bilgisayarla görü (computer vision) projesidir.

## 📌 Proje Durumu ve Notlar
⚠️ **Geliştirme Aşamasında / Durduruldu:** Proje şu an için aktif geliştirme aşamasında dondurulmuştur. Çok kaynaklı veri setlerinin birleştirilmesi ve etiket uyarlamaları başarıyla tamamlanmış olsa da, **mevcut veri setlerinin çeşitlilik ve miktar olarak yetersiz kalması**, modelin karmaşık ve uzak açılı gerçek dünya afet sahnelerinde yüksek doğrulukla (özellikle insan sınıfında) çalışmasını kısıtlamıştır. İlerleyen süreçte daha zengin ve geniş veri setleriyle projenin yeniden ele alınması planlanmaktadır.

## 🎯 Projenin Amacı ve Kapsamı
* **Çok Sınıflı Hasar Tespiti:** Binaların durumunu 4 farklı kategoride sınıflandırmak (`destroyed`, `major-damage`, `minor-damage`, `no-damage`).
* **İnsan Tespiti (`person`):** Afet sahasındaki arama-kurtarma faaliyetlerine destek olmak amacıyla insan siluetlerinin tespiti.
* **Veri Mühendisliği:** Farklı kaynaklardan gelen veri setlerinin ortak bir şemada (`T3_Merged_Dataset`) birleştirilmesi ve sınıf ID'lerinin otomatik olarak yeniden haritalandırılması (`remap`).

## 🛠️ Kullanılan Teknolojiler
* Python[cite: 1]
* Ultralytics YOLO (v8)[cite: 1]
* Google Colab (Tesla T4 GPU)[cite: 3]
* Roboflow[cite: 4] ve PyYAML

## 📊 Mimari ve Sınıf Dağılımı
Model, toplamda 5 sınıflı bir hiyerarşi ile eğitilmiştir:
1. `person` (İnsan)
2. `destroyed` (Yıkık Bina)
3. `major-damage` (Ağır Hasarlı)
4. `minor-damage` (Hafif Hasarlı)
5. `no-damage` (Hasarsız)

## 🚀 Çalıştırma Rehberi

Modeli test etmek ve tahmin yürütmek için şu adımları izleyebilirsiniz:
1. Colab veya Jupyter ortamına test etmek istediğiniz afete ait **fotoğrafı yükleyin**.
2. Yüklediğiniz fotoğrafın **dosya yolunu kopyalayın** (Örn: `/content/fotograf_adi.png`).
3. Aşağıdaki Python kod bloğunda yer alan `source` parametresine bu dosya yolunu yapıştırarak çalıştırın:

```python
from ultralytics import YOLO

# Eğitilmiş model ağırlıklarını yüklüyoruz (veya varsayılan model)
model = YOLO('yolov8s.pt')

# Test edilecek görselin yolunu source kısmına yapıştırın
results = model.predict(source="/content/fotograf_adi.png", save=True, imgsz=640, conf=0.25)
