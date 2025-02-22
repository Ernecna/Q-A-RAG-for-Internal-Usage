# Retrieval-Augmented Generation (RAG) Project

This project implements a Retrieval-Augmented Generation (RAG) system that integrates retrieval results from a FAISS vector store with a local LLM (e.g., a Llama-based model) to generate answers. The project processes documents from a CSV file, splits them into manageable chunks, indexes them using FAISS, and offers an interactive Streamlit interface where users can ask questions and provide feedback (like/dislike) on the generated responses. The feedback is saved into a SQLite database for later analysis.

## Features

- **Document Processing:**  
  Reads the "Sorun Açıklaması" (Problem Description) and "Cevap" (Answer) columns from a CSV file, concatenates them, and splits the text into chunks using a recursive text splitter.

- **Vector Store & Retrieval:**  
  Uses LangChain and FAISS (with the updated `langchain_community` imports) to create document embeddings and perform similarity search. It also supports loading an existing FAISS index from disk.

- **Retrieval-Augmented Generation (RAG):**  
  Combines retrieval results with a prompt template and sends the prompt to a local LLM model (via ollama, for example) to generate an answer.

- **Interactive User Interface:**  
  Built with Streamlit, the interface allows users to input queries, view retrieval results and LLM responses, and maintain a conversation history.

- **Feedback & Database Integration:**  
  Each assistant answer includes "Like" and "Dislike" buttons. The selected feedback, along with the question, answer, and timestamp, is stored in a SQLite database for further use.

## Project Structure
rag_project/   
├── config.py # Global project settings     
├── database_manager.py # Handles SQLite database operations   
├── document_processor.py # Reads and splits documents from the CSV file  
├── rag_chain.py # Manages retrieval and LLM integration (prompt chain)  
├── vector_store_manager.py # Manages FAISS vector store creation, saving, and loading  
├── app.py # Streamlit user interface for the RAG system └── main.py # Command-line execution script (optional)

## Installation

1. **Python Environment:**  
   Ensure you are using Python 3.7 or above.

2. **Install Dependencies:**  
   Create a `requirements.txt` file with the following (adjust versions as needed):

   ```txt
   streamlit
   langchain_community
   langchain_core
   pandas
   sqlite3
   transformers
   ollama
Install the dependencies with:

bash
Copy
pip install -r requirements.txt

FAISS Index:
If you already have a pre-built FAISS index (saved in the faiss_index directory), ensure it is located in the project root. Otherwise, run main.py to create a new index.
Usage
Running the App with the User Interface
To launch the interactive Streamlit interface, run the following command in your terminal:

**bash streamlit run app.py**
The interface will allow you to:

Enter your query.
View retrieval results with similarity scores.
See the generated prompt and LLM response.
Provide feedback on each assistant response using "Like" or "Dislike" buttons.
Running the Command-Line Script
Alternatively, you can run the main.py file to execute the full workflow (document processing, vector store creation, and LLM querying) from the command line:

**bash 
python main.py**  
**Future Developments**  
Enhanced User Interface:  
Improve the UI design for better aesthetics and usability.

Advanced Database Integration:
Expand support to other databases (e.g., PostgreSQL) for more robust data storage and analysis.

Improved Prompt Chains:
Develop more sophisticated retrieval-augmented generation chains and context management strategies to enhance answer quality.

**Security Notes**  
The FAISS index file is serialized using pickle. When loading the index, the allow_dangerous_deserialization=True parameter is set. Only use this option if you trust the source of the file.
License
This project is licensed under the MIT License. (Include your license file or adjust as needed.)
