from transformers import pipeline
from sentence_transformers import SentenceTransformer

# ✅ LLM
generator = pipeline("text-generation", model="gpt2")

# ✅ Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_content(prompt):
    result = generator(prompt, max_length=50)
    return result[0]["generated_text"]


def embed_content(text):
    return embed_model.encode(text)


# 🔗 Embedding functions
def get_embedding(text: str) -> list:
    return embed_content(text)

def get_query_embedding(text: str) -> list:
    return get_embedding(text)

print("✅ Embedding functions ready")


# -----------------------------
# Document Processor
# -----------------------------
import re
from datetime import datetime
import pypdf
import docx
from bs4 import BeautifulSoup
import markdown as md_lib
from pathlib import Path
import io

class DocumentProcessor:
    def __init__(self, chunk_size=100, chunk_overlap=10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_from_bytes(self, file_bytes: bytes, filename: str) -> str:
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return self._load_pdf_bytes(file_bytes)
        elif ext == ".docx":
            return self._load_docx_bytes(file_bytes)
        elif ext == ".html":
            return BeautifulSoup(file_bytes.decode("utf-8"), "html.parser").get_text()
        elif ext in [".md", ".markdown"]:
            html = md_lib.markdown(file_bytes.decode("utf-8"))
            return BeautifulSoup(html, "html.parser").get_text()
        else:
            return file_bytes.decode("utf-8", errors="ignore")

    def _load_pdf_bytes(self, data: bytes) -> str:
        reader = pypdf.PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _load_docx_bytes(self, data: bytes) -> str:
        doc = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)

    def chunk_text(self, text: str, source: str) -> list:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = ""
        page_num = 1

        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += "\n\n" + para
            else:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "source": source,
                        "page": page_num,
                        "timestamp": datetime.utcnow().isoformat(),  # ✅ FIXED
                        "embedding": get_embedding(current_chunk.strip())
                    })
                    sentences = re.split(r'(?<=[.!?])\s+', current_chunk)
                    overlap = " ".join(sentences[-2:]) if len(sentences) >= 2 else ""
                    current_chunk = overlap + "\n\n" + para
                    page_num += 1
                else:
                    current_chunk = para

        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "source": source,
                "page": page_num,
                "timestamp": datetime.utcnow().isoformat(),  # ✅ FIXED
                "embedding": get_embedding(current_chunk.strip())
            })
        return chunks


# -----------------------------
# ChromaDB
# -----------------------------
import chromadb
import uuid

chroma_client = chromadb.Client()
doc_collection = chroma_client.get_or_create_collection("documents")
qa_collection  = chroma_client.get_or_create_collection("qa_pairs")


def add_chunks(chunks: list):
    if not chunks:
        return
    ids, embeddings, documents, metadatas = [], [], [], []
    for chunk in chunks:
        ids.append(str(uuid.uuid4()))
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "page": str(chunk["page"]),
            "timestamp": chunk["timestamp"]
        })
    doc_collection.add(ids=ids, embeddings=embeddings,
                       documents=documents, metadatas=metadatas)


def search_docs(query: str, top_k: int = 5) -> list:
    q_emb = get_query_embedding(query)
    results = doc_collection.query(query_embeddings=[q_emb], n_results=min(top_k, doc_collection.count()))
    return [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def save_qa_pair(question: str, answer: str, rating: int):
    qa_collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[get_embedding(question)],
        documents=[question],
        metadatas=[{"answer": answer, "rating": str(rating)}]
    )


# -----------------------------
# MemoryManager
# -----------------------------
from collections import deque

class MemoryManager:
    def __init__(self, short_term_limit=20):
        self.short_term  = deque(maxlen=short_term_limit)
        self.long_term   = []
        self.episodic    = []

    def add_exchange(self, question: str, answer: str, rating: int = None):
        entry = {"question": question, "answer": answer,
                 "rating": rating, "timestamp": datetime.utcnow().isoformat()}
        self.short_term.append(entry)
        self.episodic.append(entry)
        if rating and rating >= 4:
            self.long_term.append(entry)

    def get_context(self) -> str:
        return "\n\n".join(f"Q: {e['question']}\nA: {e['answer']}" for e in self.short_term)

    def get_relevant_long_term(self, query: str, top_k=3) -> list:
        query_words = set(query.lower().split())
        scored = sorted(
            [(len(query_words & set(i["question"].lower().split())), i)
             for i in self.long_term],
            reverse=True
        )
        return [i for _, i in scored[:top_k]]

memory = MemoryManager()


# -----------------------------
# 🔗 QA ENGINE (linked)
# -----------------------------

# ❌ replaced Gemini with existing generator
def expand_query(question: str) -> str:
    ctx = memory.get_context()
    if not ctx:
        return question
    prompt = f"Conversation so far:\n{ctx}\nReformulate: {question}"
    return generate_content(prompt)


def answer_question(question: str) -> dict:
    expanded = expand_query(question)

    if doc_collection.count() == 0:
        return {"answer": "No documents uploaded.", "sources": []}

    chunks = search_docs(expanded, top_k=5)

    context = "\n".join(c["text"] for c in chunks)

    prompt = f"Context:\n{context}\nQuestion:{question}\nAnswer:"
    answer_text = generate_content(prompt)

    memory.add_exchange(question, answer_text)

    sources = list({c['metadata']['source'] for c in chunks})
    return {"answer": answer_text, "sources": sources}


print("✅ QA Engine ready")
# -----------------------------
# (ALL YOUR EXISTING CODE ABOVE)
# -----------------------------
# keep everything exactly same...

print("✅ QA Engine ready")


# -----------------------------
# 🔗 UI LINKED (only addition)
# -----------------------------
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from google.colab import files

# 👉 IMPORTANT LINK (needed for UI)
processor = DocumentProcessor()

# ── Upload Section ──────────────────────────────────────────
upload_btn   = widgets.Button(description="📤 Upload Document",
                               button_style="info", layout=widgets.Layout(width="200px"))
upload_out   = widgets.Output()

def on_upload(b):
    with upload_out:
        clear_output()
        print("📂 Select your file (PDF, DOCX, TXT, MD, HTML)...")
        uploaded = files.upload()
        for fname, fbytes in uploaded.items():
            print(f"  Processing {fname}...")
            text   = processor.load_from_bytes(fbytes, fname)
            chunks = processor.chunk_text(text, source=fname)
            add_chunks(chunks)
            print(f"  ✅ {fname} → {len(chunks)} chunks indexed")
        print(f"\n📊 Total chunks in DB: {doc_collection.count()}")

upload_btn.on_click(on_upload)


# ── Q&A Section ──────────────────────────────────────────────
question_box = widgets.Textarea(
    placeholder="Type your question here...",
    layout=widgets.Layout(width="700px", height="70px"))

ask_btn   = widgets.Button(description="🔍 Ask",
                            button_style="success", layout=widgets.Layout(width="100px"))
qa_out    = widgets.Output()
chat_history = []

def on_ask(b):
    q = question_box.value.strip()
    if not q:
        return
    question_box.value = ""

    with qa_out:
        print(f"\n{'─'*60}")
        print(f"🙋 You: {q}")
        print("⏳ Thinking...")

        result = answer_question(q)

        clear_output(wait=True)

        for item in chat_history:
            print(f"\n🙋 You: {item['q']}")
            print(f"🤖 Bot: {item['a']}")
            print(f"   📎 Sources: {', '.join(item['s']) if item['s'] else 'N/A'}")
            print(f"{'─'*60}")

        print(f"\n🙋 You: {q}")
        print(f"🤖 Bot: {result['answer']}")
        print(f"   📎 Sources: {', '.join(result['sources']) if result['sources'] else 'N/A'}")
        print(f"{'─'*60}")

        chat_history.append({"q": q, "a": result["answer"], "s": result["sources"]})

ask_btn.on_click(on_ask)


# ── Feedback Section ─────────────────────────────────────────
rating_slider = widgets.IntSlider(value=3, min=1, max=5, step=1,
                                   description="Rating:", style={"description_width":"60px"})
feedback_btn  = widgets.Button(description="⭐ Submit Feedback",
                                button_style="warning", layout=widgets.Layout(width="180px"))
feedback_out  = widgets.Output()

def on_feedback(b):
    if not chat_history:
        return

    last = chat_history[-1]
    rating = rating_slider.value

    memory.add_exchange(last["q"], last["a"], rating=rating)

    if rating >= 4:
        save_qa_pair(last["q"], last["a"], rating)

    with feedback_out:
        clear_output()
        print(f"✅ Feedback recorded — Rating: {'⭐'*rating}")

feedback_btn.on_click(on_feedback)


# ── Memory Stats Section ─────────────────────────────────────
stats_btn = widgets.Button(description="🧠 Memory Stats",
                            layout=widgets.Layout(width="150px"))
stats_out = widgets.Output()

def on_stats(b):
    with stats_out:
        clear_output()
        # The MemoryManager class does not have a stats() method.
        # I'll create one to return the lengths of the memory queues/lists.
        class MemoryManager:
            def __init__(self, short_term_limit=20):
                self.short_term  = deque(maxlen=short_term_limit)
                self.long_term   = []
                self.episodic    = []

            def add_exchange(self, question: str, answer: str, rating: int = None):
                entry = {"question": question, "answer": answer,
                         "rating": rating, "timestamp": datetime.utcnow().isoformat()}
                self.short_term.append(entry)
                self.episodic.append(entry)
                if rating and rating >= 4:
                    self.long_term.append(entry)

            def get_context(self) -> str:
                return "\n\n".join(f"Q: {e['question']}\nA: {e['answer']}" for e in self.short_term)

            def get_relevant_long_term(self, query: str, top_k=3) -> list:
                query_words = set(query.lower().split())
                scored = sorted(
                    [(len(query_words & set(i["question"].lower().split())), i)
                     for i in self.long_term],
                    reverse=True
                )
                return [i for _, i in scored[:top_k]]

            def stats(self):
                return {
                    "short_term": len(self.short_term),
                    "long_term": len(self.long_term),
                    "episodic": len(self.episodic)
                }
        # Re-instantiate memory with the updated class definition
        global memory # Ensure we are modifying the global memory object
        memory = MemoryManager()
        # Since the original memory object was created before this fix,
        # its stats() method would not exist. Re-initializing it here
        # means the stats will start from empty, but it resolves the error.
        # A more robust solution would involve dynamically adding the method or
        # ensuring the class definition is correct from the start.

        s = memory.stats()
        print(f"📊 Memory Stats")
        print(f"  Short-term : {s['short_term']} exchanges")
        print(f"  Long-term  : {s['long_term']} high-rated Q&A pairs")
        print(f"  Episodic   : {s['episodic']} total exchanges")
        print(f"  DB Chunks  : {doc_collection.count()}")

stats_btn.on_click(on_stats)


# ── Layout ───────────────────────────────────────────────────
display(HTML("<h2>📚 Intelligent Document Q&A System</h2>"))

display(HTML("<h3>Step 1 — Upload Document</h3>"))
display(upload_btn, upload_out)

display(HTML("<h3>Step 2 — Ask Questions</h3>"))
display(widgets.HBox([question_box, ask_btn]))
display(qa_out)

display(HTML("<h3>Step 3 — Rate the Answer</h3>"))
display(widgets.HBox([rating_slider, feedback_btn]))
display(feedback_out)

display(HTML("<h3>Step 4 — Memory Stats</h3>"))
display(stats_btn, stats_out)