import chromadb
from sentence_transformers import SentenceTransformer
import requests

# ======================
# CONFIG
# ======================
EMBED_MODEL_NAME = "BAAI/bge-m3"
CHROMA_HOST = "93.183.75.47"
CHROMA_PORT = 8000
TOP_K = 7

embedding_model = SentenceTransformer(EMBED_MODEL_NAME)

chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT
)

collection = chroma_client.get_or_create_collection(name="docs")


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

    return [
        {
            "text": doc,
            "source": meta["source"],
            "chunk_id": meta["chunk_id"],
            "score": dist
        }
        for doc, meta, dist in zip(docs, metas, dists)
    ]


# ======================
# LLM CALL
# ======================
def ask_llm_with_context(query, context):
    prompt = f"""
Ты — эксперт по документам.
Отвечай кратко и точно.

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


# ======================
# PIPELINE
# ======================
def ask(query):
    results = retrieve(query)
    context = "\n\n".join([r["text"] for r in results])

    return ask_llm_with_context(query, context)


if __name__ == "__main__":
    q = input("Question: ")
    print(ask(q))
