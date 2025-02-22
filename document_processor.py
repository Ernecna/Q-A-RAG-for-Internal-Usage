import os
import csv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CURRENT_DIRECTORY, CSV_FILE_NAME, CHUNK_SIZE, CHUNK_OVERLAP

class DocumentProcessor:
    def __init__(self, file_path=None):
        if file_path is None:
            self.file_path = os.path.join(CURRENT_DIRECTORY, CSV_FILE_NAME)
        else:
            self.file_path = file_path

    def read_documents(self):
        """CSV dosyasından 'Sorun Açıklaması' ve 'Cevap' sütunlarını okuyarak metinleri birleştirir."""
        documents = []
        with open(self.file_path, "r", encoding="utf-8", errors="replace") as file:
            reader = csv.DictReader(file)
            for row in reader:
                combined_text = f"{row['Sorun Açıklaması']}\n{row['Cevap']}"
                documents.append(combined_text)
        return documents

    def split_documents(self, documents):
        """Okunan metinleri belirlenen chunk boyutu ve overlap ile parçalara ayırır."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        texts = text_splitter.create_documents(documents)
        return texts