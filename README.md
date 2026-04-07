# 📚 Intelligent Document Q&A System (RAG-Based)

This project implements an **Intelligent Document Question Answering System** using a **Retrieval-Augmented Generation (RAG)** approach.

It allows users to:
- Upload documents (PDF, DOCX, HTML, Markdown, TXT)
- Ask questions based on those documents
- Get context-aware answers
- Store memory and feedback

---

## 🧠 Overview

The system combines:
- **Semantic Search (Embeddings)**
- **Vector Database (ChromaDB)**
- **Language Model (GPT-2)**
- **Memory System**
- **Interactive UI (Colab Widgets)**

---

## ⚙️ How it Works
User Uploads Document
↓
Text Extraction + Chunking
↓
Embeddings (SentenceTransformer)
↓
Store in Vector DB (ChromaDB)
↓
User Question
↓
Query Expansion (Memory)
↓
Similarity Search (Top Chunks)
↓
GPT-2 (Generate Answer)

---

## 🚀 Features

- 📄 Multi-format document support (PDF, DOCX, HTML, MD, TXT)
- 🔍 Semantic search using embeddings
- 🧠 Short-term & long-term memory
- 💬 Context-aware Q&A system
- ⭐ Feedback-based learning
- 📊 Memory statistics tracking
- 🖥 Interactive UI using ipywidgets (Colab)

---

## 🛠 Tech Stack

- **Language Model**: GPT-2 (HuggingFace Transformers)
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector Database**: ChromaDB
- **Backend**: Python
- **UI**: ipywidgets (Google Colab)

### 📂 Document Processing
- pypdf (PDF)
- python-docx (DOCX)
- BeautifulSoup (HTML)
- markdown (MD)

---

## 🤖 Models Used

### 🔹 LLM
- **GPT-2**
- Used for generating answers
- Runs locally (no API required)

### 🔹 Embedding Model
- **all-MiniLM-L6-v2**
- Converts text into vector embeddings for semantic search

---

## 💻 How to Run (Google Colab)

### Step 1: Install dependencies
```bash
pip install transformers sentence-transformers torch chromadb pypdf python-docx beautifulsoup4 markdown ipywidgets
### Step 2: Run the code
Open Google Colab
Paste your code
Run all cells
Upload document
Ask questions

### 🖥 User Interface

The system uses ipywidgets UI in Colab:

📤 Upload button (for documents)
📝 Question input box
🔍 Ask button
⭐ Feedback system (rating)
🧠 Memory stats viewer

📂 Project Structure
rag_document_qa_system/
│
├── app.py              # Main application
├── requirements.txt    # Dependencies
└── README.md           # Documentation

⚠️ Limitations
GPT-2 is a small model → answers may be less accurate
No reranking implemented
Works best with clean documents
UI works mainly in Google Colab
