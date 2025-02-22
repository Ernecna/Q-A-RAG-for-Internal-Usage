import sqlite3
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_name="responses.db"):
        # check_same_thread=False; Streamlit yeniden çalıştığında çoklu thread'ler ile uyumlu çalışır.
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.create_table()

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            rating TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
        """
        self.conn.execute(query)
        self.conn.commit()

    def insert_response(self, question, answer, rating):
        timestamp = datetime.now().isoformat()
        query = "INSERT INTO responses (question, answer, rating, timestamp) VALUES (?, ?, ?, ?)"
        self.conn.execute(query, (question, answer, rating, timestamp))
        self.conn.commit()