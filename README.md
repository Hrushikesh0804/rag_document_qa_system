# 📚 Intelligent Document Q&A System (RAG-Based)

An end-to-end Retrieval-Augmented Generation (RAG) system that allows
users to upload documents and ask context-aware questions.

------------------------------------------------------------------------

## 🚀 Features

-   Upload PDF, DOCX, HTML, Markdown, TXT
-   Semantic search using embeddings
-   Context-aware answers using GPT-2
-   Memory system (short-term + long-term)
-   Feedback system with ratings
-   Vector database (ChromaDB)
-   Interactive UI (Google Colab)

------------------------------------------------------------------------

## 🧠 How It Works

1.  Upload document\
2.  Extract & chunk text\
3.  Convert chunks → embeddings\
4.  Store in ChromaDB\
5.  User asks question\
6.  Retrieve relevant chunks\
7.  GPT-2 generates answer\
8.  Store in memory

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python
-   Transformers (GPT-2)
-   SentenceTransformers (all-MiniLM-L6-v2)
-   ChromaDB
-   ipywidgets (UI)

------------------------------------------------------------------------

## 💻 Run (Colab)

``` bash
pip install transformers sentence-transformers torch chromadb pypdf python-docx beautifulsoup4 markdown ipywidgets
```

Run all cells → Upload document → Ask questions

------------------------------------------------------------------------

## 📂 Structure

-   app.py
-   requirements.txt
-   README.md

------------------------------------------------------------------------

## ⚠️ Limitations

-   GPT-2 is small → limited accuracy
-   No reranking
-   UI works best in Colab

------------------------------------------------------------------------

