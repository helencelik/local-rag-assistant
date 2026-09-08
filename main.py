import os
import sqlite3
import json
import numpy as np
import openai
import PyPDF2
from foundry_local_sdk import Configuration, FoundryLocalManager

print("Sistem başlatılıyor, yapay zeka modelleri hazırlanıyor (Bu biraz sürebilir)...")

# 1. MODEL VE SUNUCU AYARLARI
config = Configuration(app_name="rag_projesi")
FoundryLocalManager.initialize(config)
mgr = FoundryLocalManager.instance

EMB_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL = "phi-4-mini"

# Yerel modellerin indirilip yüklenmesi
emb_m = mgr.catalog.get_model(EMB_MODEL)
emb_m.download()
emb_m.load()

chat_m = mgr.catalog.get_model(CHAT_MODEL)
chat_m.download()
chat_m.load()

try:
    mgr.start_web_service()
except Exception:
    pass

client = openai.OpenAI(
    base_url=f"{mgr.urls[0]}/v1",
    api_key="none"
)

# 2. DOKÜMAN OKUMA VE PARÇALAMA
def get_docs(folder="data"):
    """Data klasöründeki TXT ve PDF dosyalarını okuyup paragraflara böler."""
    chunks = []
    for f_name in os.listdir(folder):
        path = os.path.join(folder, f_name)
        icerik = ""
        
        # Dosya TXT ise düz metin olarak oku
        if f_name.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                icerik = f.read()
                
        # Dosya PDF ise PyPDF2 ile sayfa sayfa oku
        elif f_name.endswith(".pdf"):
            with open(path, "rb") as f:
                okuyucu = PyPDF2.PdfReader(f)
                for sayfa in okuyucu.pages:
                    sayfa_metni = sayfa.extract_text()
                    if sayfa_metni:
                        icerik += sayfa_metni + "\n\n"
        else:
            continue # Desteklenmeyen formatları atla
            
        # Metni paragraflara böl ve listeye ekle
        paragraflar = icerik.split("\n\n")
        for p in paragraflar:
            p = p.strip()
            if len(p) > 0:
                chunks.append({"file": f_name, "text": p})
    return chunks

# 3. VERİTABANI İŞLEMLERİ (SQLITE)
def init_db(db="knowledge_base.db"):
    """Vektörlerin saklanacağı SQLite tablosunu oluşturur."""
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file TEXT,
            text TEXT,
            emb TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_embs(chunks, db="knowledge_base.db"):
    """Metin parçalarını vektörlere çevirip veritabanına kaydeder."""
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute("DELETE FROM docs") # Eski verileri temizle
    
    print("\nMetinler yapay zeka ile gerçek vektörlere çevriliyor...")
    for c_item in chunks:
        # Metni Qwen3 modeli ile vektöre (embedding) dönüştür
        resp = client.embeddings.create(model=EMB_MODEL, input=c_item["text"])
        gercek_emb = resp.data[0].embedding
        
        emb_str = json.dumps(gercek_emb)
        c.execute(
            "INSERT INTO docs (file, text, emb) VALUES (?, ?, ?)",
            (c_item["file"], c_item["text"], emb_str)
        )
    conn.commit()
    conn.close()
    print("Vektörler SQLite'a başarıyla kaydedildi!")

# 4. RAG ARAMA MANTIĞI
def calc_sim(v1, v2):
    """İki vektör arasındaki Cosine Similarity (Kosinüs Benzerliği) oranını hesaplar."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def search_chunk(query, db="knowledge_base.db"):
    """Kullanıcının sorusuna en çok benzeyen metin parçasını veritabanında bulur."""
    resp = client.embeddings.create(model=EMB_MODEL, input=query)
    q_emb = resp.data[0].embedding
    
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute("SELECT file, text, emb FROM docs")
    rows = c.fetchall()
    
    max_score = -1
    best = None
    
    for row in rows:
        d_emb = json.loads(row[2])
        skor = calc_sim(q_emb, d_emb)
        
        if skor > max_score:
            max_score = skor
            best = {"file": row[0], "text": row[1]}
            
    conn.close()
    
    if max_score < 0.55:
        return None
        
    return best

# 5. YAPAY ZEKA CEVAP ÜRETİMİ
def ask_llm(soru, baglam):
    """Bulunan metni Phi-4 modeline vererek halüsinasyonsuz cevap üretir."""
    
    # Terminalde harika çalışan o orijinal, sade komutumuz
    sys_msg = "Sen yardımcı bir asistansın. Sorulara SADECE aşağıdaki metne dayanarak cevap ver. Eğer sorunun cevabı sana verilen metinde yoksa, KESİNLİKLE kendi bilgilerini kullanma ve sadece 'Üzgünüm, bu bilgi ders notlarımda yer almıyor.' de."
    user_msg = f"Metin: {baglam}\n\nSoru: {soru}"
    
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]
    )
    return resp.choices[0].message.content

# ANA PROGRAM
if __name__ == "__main__":
    init_db()
    veri = get_docs()
    if len(veri) > 0:
        save_embs(veri)
    
    print("\n--- RAG Asistanına Hoşgeldiniz ---")
    
    while True:
        soru = input("\nSorunuz (Çıkmak için 'q'): ")
        if soru.lower() in ["q", "cikis", "exit"]:
            print("Görüşmek üzere!")
            break
            
        en_iyi_metin = search_chunk(soru)
        
        if en_iyi_metin:
            print("Cevap üretiliyor, lütfen bekleyin...")
            sonuc = ask_llm(soru, en_iyi_metin["text"])
            
            print("\n>> YAPAY ZEKA CEVABI:")
            print(sonuc)
            
            # Halüsinasyon engelleme: Eğer sistem üzgünüm dediyse kaynağı gösterme
            if "üzgünüm" not in sonuc.lower():
                print(f"(Kaynak: {en_iyi_metin['file']})")
        else:
                print("Veritabanında uygun sonuç bulunamadı.")