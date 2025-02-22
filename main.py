import streamlit as st
from vector_store_manager import VectorStoreManager
from rag_chain import RAGChain
from databasemanager import DatabaseManager


def main():
    st.title("Retrieval-Augmented Generation (RAG) Projesi")
    st.write("Bu uygulama, FAISS index'inden sorgu sonuçlarını alıp, yerel LLM ile entegre şekilde cevap üretir.")

    # Veritabanı yöneticisini başlat
    db_manager = DatabaseManager()

    # Vektör deposunu yükle (önceden oluşturulmuş FAISS index dosyasından)
    vs_manager = VectorStoreManager()
    try:
        vectorstore = vs_manager.load_vectorstore()
        st.success("Vektör deposu başarıyla yüklendi.")
    except Exception as e:
        st.error(f"Vektör deposu yüklenirken hata oluştu: {e}")
        return

    # Sohbet geçmişini tutmak için session state oluşturun
    if "chat_history" not in st.session_state:
        # Her bir eleman: ("User", message) veya ("Assistant", answer, question)
        st.session_state.chat_history = []

    # Kullanıcıdan sorgu girişi al
    query = st.text_input("Sorgunuzu girin:")

    if st.button("Gönder"):
        if query:
            # Kullanıcı sorgusunu geçmişe ekle
            st.session_state.chat_history.append(("User", query))

            # Retrieval işlemi: Sorguya göre en yakın 3 sonucu al
            results = vs_manager.similarity_search(query, k=3)

            # Retrieval sonuçlarını kullanarak prompt oluşturma
            rag_chain = RAGChain()
            formatted_prompt = rag_chain.format_prompt(context=results, question=query)

            # LLM'den cevabı al
            response = rag_chain.query_llm(formatted_prompt)
            # Cevap ile birlikte orijinal sorguyu da saklıyoruz
            st.session_state.chat_history.append(("Assistant", response, query))
        else:
            st.warning("Lütfen sorgunuzu girin.")

    # Sohbet geçmişini ekranda göster
    st.markdown("### Sohbet Geçmişi")
    for i, entry in enumerate(st.session_state.chat_history):
        if entry[0] == "User":
            st.markdown(f"**Siz:** {entry[1]}")
        else:
            # entry[0]=="Assistant", entry[1] = cevap, entry[2] = sorgu
            st.markdown(f"**Asistan:** {entry[1]}")
            # Her asistan cevabı için like/dislike butonları oluştur
            col1, col2 = st.columns(2)
            if col1.button("Like", key=f"like_{i}"):
                db_manager.insert_response(entry[2], entry[1], "like")
                st.success("Cevap beğenildi ve kaydedildi!")
            if col2.button("Dislike", key=f"dislike_{i}"):
                db_manager.insert_response(entry[2], entry[1], "dislike")
                st.info("Cevap beğenilmedi ve kaydedildi!")
        st.markdown("---")


if __name__ == "__main__":
    main()