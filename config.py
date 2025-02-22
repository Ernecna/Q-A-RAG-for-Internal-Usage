import os

# Genel ayarlar
CURRENT_DIRECTORY = os.getcwd()
CSV_FILE_NAME = "combined_output.csv"
FAISS_INDEX_PATH = "faiss_index"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"
