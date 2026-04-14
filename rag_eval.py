import json
import numpy as np
from collections import Counter

from app_gui import retrieve, ask_llm_with_context


# ======================
# DEBUG CONTEXT
# ======================
def print_retrieved_context(query, results, top_k=5):
    print("\n==============================")
    print("RETRIEVAL DEBUG CONTEXT")
    print("==============================")

    print("\nЗапрос:")
    print(query)

    print("\nTOP-K RETRIEVAL:")
    for i, r in enumerate(results[:top_k]):
        print(f"\n[{i+1}] source={r['source']} chunk={r['chunk_id']} score={r['score']:.4f}")
        print(r["text"][:300])

    print("\n==============================\n")


# ======================
# DATASET
# ======================
def load_dataset(path="eval_dataset.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ======================
# METRICS
# ======================
def recall_at_k(results, relevant_docs, k=5):
    top = results[:k]

    for rel in relevant_docs:
        for r in top:
            if rel.lower() in r["text"].lower():
                return 1
    return 0


def mrr(results, relevant_docs):
    for i, r in enumerate(results):
        for rel in relevant_docs:
            if rel.lower() in r["text"].lower():
                return 1.0 / (i + 1)
    return 0


def normalize(text):
    return text.lower().strip()


def f1_score(pred, truth):
    pred_tokens = normalize(pred).split()
    truth_tokens = normalize(truth).split()

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)

    return 2 * precision * recall / (precision + recall)


def exact_match(pred, truth):
    return int(normalize(pred) == normalize(truth))


# ======================
# EVALUATION
# ======================
def evaluate(dataset):
    recall_scores = []
    mrr_scores = []
    f1_scores = []
    em_scores = []

    for i, sample in enumerate(dataset):
        q = sample["question"]
        answers = sample["answers"]
        relevant_docs = sample.get("relevant_docs", [])

        print(f"\n--- [{i+1}] {q}")

        # ===== RETRIEVE =====
        results = retrieve(q)

        r_at_k = recall_at_k(results, relevant_docs, k=5)
        recall_scores.append(r_at_k)

        print("Recall@5:", r_at_k)

        mrr_score = mrr(results, relevant_docs)
        mrr_scores.append(mrr_score)

        print("MRR:", round(mrr_score, 3))

        # ===== DEBUG CONTEXT =====
        print_retrieved_context(q, results)

        # ===== LLM =====
        context = "\n\n".join([r["text"] for r in results[:5]])
        pred = ask_llm_with_context(q, context)

        print("Prediction:", pred)

        # ===== GENERATION METRICS =====
        f1 = max(f1_score(pred, a) for a in answers)
        em = max(exact_match(pred, a) for a in answers)

        f1_scores.append(f1)
        em_scores.append(em)

        print("F1:", round(f1, 3), "| EM:", em)

    # ======================
    # FINAL RESULTS
    # ======================
    print("\n================= FINAL RESULTS =================")
    print("Recall@5:", round(np.mean(recall_scores), 3))
    print("MRR:", round(np.mean(mrr_scores), 3))
    print("F1:", round(np.mean(f1_scores), 3))
    print("Exact Match:", round(np.mean(em_scores), 3))


# ======================
# RUN
# ======================
if __name__ == "__main__":
    dataset = load_dataset()
    evaluate(dataset)