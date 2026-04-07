# 📚 Intelligent Document Q&A System (RAG-Based)

An end-to-end **Retrieval-Augmented Generation (RAG)** system that
allows users to upload documents and ask context-aware questions.

------------------------------------------------------------------------

## 🚀 Project Overview

This project builds a complete pipeline that:

-   📄 Accepts documents (PDF, DOCX, HTML, Markdown, TXT)
-   🔍 Converts them into embeddings
-   🧠 Stores them in a vector database (ChromaDB)
-   💬 Answers user questions using GPT-2
-   ⭐ Learns from feedback using memory

------------------------------------------------------------------------

## ⚙️ How It Works

Upload Document → Text Extraction → Chunking → Embeddings → Store in
ChromaDB → Query → Retrieval → GPT-2 Answer → Memory

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Language Model: GPT-2 (HuggingFace Transformers)
-   Embeddings: all-MiniLM-L6-v2
-   Vector DB: ChromaDB
-   Backend: Python
-   UI: ipywidgets (Colab)

------------------------------------------------------------------------

## 🤖 Models Used

-   GPT-2 → Answer generation
-   all-MiniLM-L6-v2 → Embeddings

------------------------------------------------------------------------

## 💻 How to Run

### ✅ Google Colab (Recommended)

⚠️ This project is designed to run mainly in Google Colab.

    pip install transformers sentence-transformers torch chromadb pypdf python-docx beautifulsoup4 markdown ipywidgets

Run → Upload → Ask

------------------------------------------------------------------------

### ⚠️ Local Machine

-   Works but slower
-   UI may not behave properly
-   Recommended: 8GB RAM+

------------------------------------------------------------------------

## 🌐 Streamlit

This project currently uses **Colab UI (ipywidgets)**\
Streamlit is not implemented but can be added in future.

------------------------------------------------------------------------

## 📂 Project Structure

-   app.py
-   requirements.txt
-   README.md

------------------------------------------------------------------------

## ⚠️ Limitations

-   GPT-2 is small → lower accuracy
-   No reranking
-   Works best with clean data

------------------------------------------------------------------------

## 🚀 Future Improvements

-   Better models (LLaMA / Mistral)
-   Streamlit UI
-   Better retrieval


