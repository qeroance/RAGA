import os
import re
import threading
import tkinter as tk
from tkinter import scrolledtext

import chromadb
import requests
from sentence_transformers import SentenceTransformer


# ======================
# PDF SUPPORT
# ======================
try:
    import fitz
    USE_FITZ = True
except Exception:
    from pypdf import PdfReader
    USE_FITZ = False


# ======================
# CONFIG
# ======================
DATA_FOLDER = "data"
CHROMA_PATH = "./chroma_db"

EMBED_MODEL_NAME = "BAAI/bge-m3"

CHUNK_SIZE = 1000
TOP_K = 7


# ======================
# MODELSЫ
# ======================
embedding_model = SentenceTransformer(EMBED_MODEL_NAME)

CHROMA_HOST = "93.183.75.47"   # ← сюда IP сервера
CHROMA_PORT = 8000

chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT
)

collection = chroma_client.get_or_create_collection(name="docs")


# ======================
# CLEAN TEXT
# ======================
def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[^\w\s.,!?():;\-\n]", "", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# ======================
# PDF EXTRACTION
# ======================
def extract_pdf_text(path):
    try:
        if USE_FITZ:
            doc = fitz.open(path)
            pages = []

            for page in doc:
                blocks = page.get_text("blocks")
                blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

                page_text = []
                for b in blocks:
                    t = b[4].strip()
                    if len(t) > 3:
                        page_text.append(t)

                pages.append("\n".join(page_text))

            return "\n\n".join(pages)

        else:
            reader = PdfReader(path)
            pages = []

            for page in reader.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)

            return "\n\n".join(pages)

    except Exception as e:
        print("PDF ERROR:", e)
        return ""


# ======================
# CHUNKING
# ======================
def split_text(text):
    paragraphs = text.split("\n")

    chunks = []
    current = ""
    chunk_id = 0

    for p in paragraphs:
        if not p.strip():
            continue

        if len(current) + len(p) < CHUNK_SIZE:
            current += " " + p
        else:
            chunks.append((current.strip(), chunk_id))
            chunk_id += 1
            current = p

    if current:
        chunks.append((current.strip(), chunk_id))

    return chunks


# ======================
# LOAD DOCS
# ======================
def load_all_docs(folder):
    docs = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        try:
            text = ""

            if file.endswith(".txt"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

            elif file.endswith(".pdf"):
                print("Processing:", file)
                text = extract_pdf_text(path)

            text = clean_text(text)

            if not text or len(text) < 50:
                continue

            chunks = split_text(text)

            for c, cid in chunks:
                docs.append({
                    "text": c,
                    "source": file,
                    "chunk_id": cid
                })

        except Exception as e:
            print("Error:", file, e)

    print("TOTAL CHUNKS:", len(docs))
    return docs


# ======================
# INDEX
# ======================
def build_index():
    docs = load_all_docs(DATA_FOLDER)

    for i, d in enumerate(docs):
        emb = embedding_model.encode(d["text"]).tolist()

        collection.add(
            documents=[d["text"]],
            embeddings=[emb],
            ids=[f"id_{i}"],
            metadatas=[{
                "source": d["source"],
                "chunk_id": d["chunk_id"]
            }]
        )

    print("INDEX READY:", len(docs))


if collection.count() == 0:
    build_index()
else:
    print("Using existing index:", collection.count())


# ======================
# RETRIEVAL
# ======================
def retrieve(query):
    q_emb = embedding_model.encode(query).tolist()

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    results = []

    for doc, meta, dist in zip(docs, metas, dists):
        results.append({
            "text": doc,
            "source": meta["source"],
            "chunk_id": meta["chunk_id"],
            "score": dist
        })

    return results


# ======================
# LLM
# ======================
def ask_llm_with_context(query, context):
    try:
        prompt = f"""
Ты — эксперт по документам.
Отвечай лишь на поставленный вопрос без дополнений.
Четко и кратко.

КОНТЕКСТ:
{context}

ВОПРОС:
{query}

ОТВЕТ:
"""

        r = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "gemma3:4b",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1
            },
            timeout=120
        )

        return r.json().get("response", "")

    except Exception as e:
        return f"Ошибка: {str(e)}"


# ======================
# PIPELINE
# ======================
def ask_llm(query):
    results = retrieve(query)

    context = "\n\n".join([r["text"] for r in results[:7]])

    return ask_llm_with_context(query, context)


# ======================
# DEBUG TOOL
# ======================
def debug_retrieval(query):
    results = retrieve(query)

    print("\n====================")
    print("RETRIEVAL DEBUG")
    print("====================")

    print("\nQUERY:", query)

    for i, r in enumerate(results):
        print(f"\n[{i+1}] score={r['score']:.4f}")
        print(r["text"][:250])

    return results


# ======================
# GUI
# ======================
BG = "#0a0e19"
TEXT = "#e2e8f0"

window = tk.Tk()
window.title("RAG Bot")
window.geometry("750x550")
window.configure(bg=BG)

chat = scrolledtext.ScrolledText(window, bg=BG, fg=TEXT, wrap=tk.WORD)
chat.pack(fill=tk.BOTH, expand=True)
chat.config(state=tk.DISABLED)

entry = tk.Entry(window, font=("Arial", 13))
entry.pack(fill=tk.X, padx=10, pady=8)


def add(sender, msg):
    chat.config(state=tk.NORMAL)
    chat.insert(tk.END, f"\n{sender}:\n{msg}\n")
    chat.config(state=tk.DISABLED)
    chat.see(tk.END)


def send(event=None):
    q = entry.get()
    if not q:
        return

    entry.delete(0, tk.END)
    add("Ты", q)

    chat.config(state=tk.NORMAL)
    idx = chat.index(tk.END)
    chat.insert(tk.END, "\nБот: думает...\n")
    chat.config(state=tk.DISABLED)

    def run():
        ans = ask_llm(q)

        def update():
            chat.config(state=tk.NORMAL)
            chat.delete(idx, tk.END)
            chat.config(state=tk.DISABLED)
            add("Бот", ans)

        window.after(0, update)

    threading.Thread(target=run, daemon=True).start()


entry.bind("<Return>", send)

if __name__ == "__main__":
    window.mainloop()