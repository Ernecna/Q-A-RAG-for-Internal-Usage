from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH

class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vectorstore = None

    def create_vectorstore(self, documents):
        """Dokümanlardan FAISS vektör deposunu oluşturur."""
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        return self.vectorstore

    def save_vectorstore(self):
        """Oluşturulan vektör deposunu yerel olarak kaydeder."""
        if self.vectorstore:
            self.vectorstore.save_local(FAISS_INDEX_PATH)

    def load_vectorstore(self):
        """Daha önce kaydedilmiş vektör deposunu yükler."""
        self.vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True  # Bu parametreyi ekleyerek pickle deserialization'ına izin veriyoruz.
        )
        return self.vectorstore

    def similarity_search(self, query, k=3):
        """Verilen sorguya göre en yakın k belgeyi ve skorlarını getirir."""
        if self.vectorstore is None:
            raise ValueError("Vectorstore has not been initialized.")
        return self.vectorstore.similarity_search_with_score(query, k=k)