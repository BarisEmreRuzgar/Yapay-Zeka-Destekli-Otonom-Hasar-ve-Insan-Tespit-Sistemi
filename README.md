<div align="center">

# 🌍 DisasterVision AI: Yapay Zeka Destekli Otonom Hasar ve İnsan Tespit Sistemi

<p align="center">
  <img src="https://img.shields.io/badge/PYTHON-3.13-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv8-ULTRALYTICS-green?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/GOOGLE_COLAB-TESLA_T4-orange?style=for-the-badge&logo=googlecolab&logoColor=white" />
  <img src="https://img.shields.io/badge/STATUS-PAUSED-red?style=for-the-badge" />
</p>

Afet bölgelerinde havadan (İHA/drone) çekilen görüntüler üzerinden bina hasar durumunu tespit eden ve insan varlığını algılayan bilgisayarla görü projesi.

</div>

---

## 📌 Proje Durumu ve Notlar

⚠️ **Geliştirme Aşamasında / Durduruldu:** Proje şu an için aktif geliştirme aşamasında dondurulmuştur. Çok kaynaklı veri setlerinin birleştirilmesi ve etiket uyarlamaları başarıyla tamamlanmış olsa da, **mevcut veri setlerinin çeşitlilik ve miktar olarak yetersiz kalması**, modelin karmaşık ve uzak açılı gerçek dünya afet sahnelerinde yüksek doğrulukla (özellikle insan sınıfında) çalışmasını kısıtlamıştır. İlerleyen süreçte daha zengin ve geniş veri setleriyle projenin yeniden ele alınması planlanmaktadır.

---

## 🎯 Projenin Amacı ve Kapsamı

| Özellik | Açıklama |
| :--- | :--- |
| 🏢 **Çok Sınıflı Hasar** | Binaların durumunu 4 kategoride sınıflandırma (`destroyed`, `major-damage`, `minor-damage`, `no-damage`). |
| 👤 **İnsan Tespiti** | Afet sahasındaki arama-kurtarma faaliyetlerine destek olmak amacıyla insan siluetlerinin tespiti (`person`). |
| 🔄 **Veri Mühendisliği** | Farklı kaynaklardan gelen veri setlerinin ortak şemada birleştirilmesi ve sınıf ID'lerinin yeniden haritalandırılması. |

---

## 🛠️ Teknik Özellikler

* **Yapay Zeka Altyapısı:** Ultralytics YOLOv8 nesne tespiti modeli.
* **Eğitim Ortamı:** Google Colab (Tesla T4 GPU gücüyle hızlandırılmış süreç).
* **Veri Yönetimi:** Roboflow entegrasyonu, çoklu veri seti birleştirme ve YAML konfigürasyonları.

---

## 📊 Sınıf Hiyerarşisi

Model, toplamda 5 sınıflı bir mimari ile tasarlanmıştır:
1. `person` (İnsan)
2. `destroyed` (Yıkık Bina)
3. `major-damage` (Ağır Hasarlı)
4. `minor-damage` (Hafif Hasarlı)
5. `no-damage` (Hasarsız)

---

## 🚀 Çalıştırma Rehberi

Modeli test etmek ve tahmin yürütmek için şu adımları izleyebilirsiniz:

1. Colab veya Jupyter ortamına test etmek istediğiniz afete ait **fotoğrafı yükleyin**.
2. Yüklediğiniz fotoğrafın **dosya yolunu kopyalayın** (Örn: `/content/fotograf_adi.png`).
3. Aşağıdaki Python kod bloğunda yer alan `source` parametresine bu dosya yolunu yapıştırarak çalıştırın:

```python
from ultralytics import YOLO

# Eğitilmiş model ağırlıklarını yüklüyoruz
model = YOLO('yolov8s.pt')

# Test edilecek görselin yolunu source kısmına yapıştırın
results = model.predict(source="/content/fotograf_adi.png", save=True, imgsz=640, conf=0.25)
