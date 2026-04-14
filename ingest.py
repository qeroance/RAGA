import os
import re
import chromadb
from sentence_transformers import SentenceTransformer

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
EMBED_MODEL_NAME = "BAAI/bge-m3"

CHROMA_HOST = "93.183.75.47"
CHROMA_PORT = 8000

CHUNK_SIZE = 1000

embedding_model = SentenceTransformer(EMBED_MODEL_NAME)

chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT
)

collection = chroma_client.get_or_create_collection(name="docs")


# ======================
# TEXT CLEANING
# ======================
def clean_text(text):
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[^\w\s.,!?():;\-\n]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ======================
# PDF
# ======================
def extract_pdf_text(path):
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
        return "\n\n".join([p.extract_text() or "" for p in reader.pages])


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
def load_docs():
    docs = []

    for file in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, file)

        try:
            if file.endswith(".txt"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

            elif file.endswith(".pdf"):
                print("Processing:", file)
                text = extract_pdf_text(path)
            else:
                continue

            text = clean_text(text)

            if len(text) < 50:
                continue

            for chunk, cid in split_text(text):
                docs.append({
                    "text": chunk,
                    "source": file,
                    "chunk_id": cid
                })

        except Exception as e:
            print("Error:", file, e)

    return docs


# ======================
# BUILD INDEX
# ======================
def build_index():
    docs = load_docs()

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

    print("INDEX BUILT:", len(docs))


if __name__ == "__main__":
    build_index()
