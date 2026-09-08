# 🚀 Local RAG Asistanı

Bu proje, **Microsoft Yaz Okulu** kapsamında geliştirilmiş, tamamen yerel (offline) çalışan bir Retrieval-Augmented Generation (RAG) uygulamasıdır.

🎥 **[Proje Sunum Videosu İçin Tıklayın](#) *(Video linki eklenecek)***

## 📌 Proje Hakkında
Dışarıdan hiçbir internet bağlantısına veya dış API'ye ihtiyaç duymadan çalışabilen bu asistan, sisteme yüklenen ders notlarını (PDF ve TXT) okur, anlar ve sorulara sadece bu notlara dayanarak cevap verir. Projenin ana odak noktası **Sorumlu Yapay Zeka (Responsible AI)** prensiplerini eksiksiz uygulamaktır.

## 🎯 Temel Özellikler ve Mühendislik Çözümleri
* **Tamamen Yerel Çalışma (Offline Execution):** Veritabanı ve LLM modelleri yerel işlemcide çalışır, dışarıya hiçbir veri sızmaz. Bu sayede tam bir veri gizliliği (Data Privacy) sağlanır.
* **Halüsinasyon Koruması (0.55 Barajı):** Dil modellerinin alakasız sorularda uydurma (Knowledge Bleed / Halüsinasyon) problemini çözmek için arama algoritmasına matematiksel bir **Kosinüs Benzerliği (Cosine Similarity) barajı** eklenmiştir. Kullanıcının sorusu, veritabanındaki notlarla en az `%55 (0.55)` oranında eşleşmiyorsa, soru yapay zekaya hiç gönderilmez ve sistem doğrudan "Üzgünüm, bu bilgi ders notlarımda yer almıyor." yanıtını verir.
* **Kullanıcı Dostu Arayüz:** Terminalden bağımsız, Streamlit ile tasarlanmış web arayüzü sayesinde yerel ağ (Network URL) üzerinden mobil cihazlarla da kullanılabilir.

## 🛠️ Kullanılan Teknolojiler
* **Dil:** Python
* **Modeller:** Phi-4-mini (Sohbet) ve Qwen3 (Embedding) - *Foundry Local SDK altyapısı ile*
* **Veritabanı:** SQLite (Vektör depolama için)
* **Arayüz:** Streamlit
* **Veri İşleme:** PyPDF2

## 🚀 Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda ayağa kaldırmak için aşağıdaki adımları izleyebilirsiniz:

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt

2. Uygulamayı başlatın: streamlit run app.py
3. Tarayıcınızda açılan http://localhost:8501 adresinden veya aynı ağdaki cihazınızdan terminalde verilen Network URL üzerinden asistanı kullanmaya başlayabilirsiniz.   
