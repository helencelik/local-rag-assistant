import streamlit as st
from main import search_chunk, ask_llm

# Sayfa ayarları
st.set_page_config(page_title="RAG Ders Asistanı", page_icon="📚")
st.title("📚 Yapay Zeka Destekli Ders Asistanı")
st.markdown("Ders notlarınıza (PDF/TXT) dair sorularınızı aşağıdan sorabilirsiniz.")

# Sohbet geçmişini tutma
if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = []

# Eski mesajları ekrana yazdırma
for mesaj in st.session_state.mesajlar:
    with st.chat_message(mesaj["role"]):
        st.markdown(mesaj["content"])

# Kullanıcıdan soru alma
soru = st.chat_input("Ders notlarınızdan ne öğrenmek istersiniz?")

if soru:
    # Kullanıcının sorusunu ekrana bas
    st.session_state.mesajlar.append({"role": "user", "content": soru})
    with st.chat_message("user"):
        st.markdown(soru)

    # Yapay zekanın cevabını üretme ve ekrana basma
    with st.chat_message("assistant"):
        with st.spinner("Notlar taranıyor ve cevap üretiliyor..."):
            en_iyi_metin = search_chunk(soru)
            
            if en_iyi_metin:
                cevap = ask_llm(soru, en_iyi_metin["text"])
                
                # Halüsinasyon yoksa kaynağı ekle
                if "üzgünüm" not in cevap.lower():
                    cevap += f"\n\n*(Kaynak: {en_iyi_metin['file']})*"
                    
                st.markdown(cevap)
                st.session_state.mesajlar.append({"role": "assistant", "content": cevap})
            else:
                hata_mesaji = "Üzgünüm, bu bilgi ders notlarımda yer almıyor."
                st.markdown(hata_mesaji)
                st.session_state.mesajlar.append({"role": "assistant", "content": hata_mesaji})