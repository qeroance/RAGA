import json
import numpy as np
from collections import Counter

# импорт из твоей системы
from app_gui import retrieve, ask_llm_with_context, TOP_K


# ======================
# DATASET
# ======================
def load_dataset(path="eval_dataset.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ======================
# RETRIEVAL METRICS
# ======================
def recall_at_k(results, relevant_docs, k=TOP_K):
    results = sorted(results, key=lambda x: x["score"])
    top = results[:k]

    for rel in relevant_docs:
        for r in top:
            if rel.lower() in r["text"].lower():
                return 1
    return 0


def mrr(results, relevant_docs):
    results = sorted(results, key=lambda x: x["score"])

    for i, r in enumerate(results):
        for rel in relevant_docs:
            if rel.lower() in r["text"].lower():
                return 1.0 / (i + 1)
    return 0


# ======================
# TEXT METRICS
# ======================
def normalize(text):
    return text.lower().strip()


def f1_score(pred, truth):
    pred_tokens = normalize(pred).split()
    truth_tokens = normalize(truth).split()

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens) if pred_tokens else 0
    recall = num_same / len(truth_tokens) if truth_tokens else 0

    if precision + recall == 0:
        return 0

    return 2 * precision * recall / (precision + recall)


def exact_match(pred, truth):
    return int(normalize(pred) == normalize(truth))


# ======================
# EVALUATION LOOP
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

        # сортировка (важно!)
        results = sorted(results, key=lambda x: x["score"])

        r_at_k = recall_at_k(results, relevant_docs)
        recall_scores.append(r_at_k)

        print("Recall@K:", r_at_k)

        mrr_score = mrr(results, relevant_docs)
        mrr_scores.append(mrr_score)

        print("MRR:", round(mrr_score, 3))

        # ===== CONTEXT (как в app_gui.py) =====
        context_blocks = []

        for j, r in enumerate(results[:TOP_K]):
            block = (
                f"[{j+1}] SOURCE: {r['source']} | CHUNK: {r['chunk_id']} | SCORE: {r['score']:.4f}\n"
                f"{r['text']}"
            )
            context_blocks.append(block)

        debug_context = "\n\n".join(context_blocks)
        context = "\n\n".join([r["text"] for r in results[:TOP_K]])

        print("\n=========== CONTEXT ===========\n")
        print(debug_context)
        print("\n===============================\n")

        # ===== LLM =====
        pred = ask_llm_with_context(q, context)

        print("Prediction:", pred)

        # ===== GENERATION METRICS =====
        f1 = max(f1_score(pred, a) for a in answers)
        em = max(exact_match(pred, a) for a in answers)

        f1_scores.append(f1)
        em_scores.append(em)

        print("F1:", round(f1, 3), "| EM:", em)

    # ======================
    # FINAL REPORT
    # ======================
    print("\n================= FINAL RESULTS =================")
    print("Recall@K:", round(np.mean(recall_scores), 3))
    print("MRR:", round(np.mean(mrr_scores), 3))
    print("F1:", round(np.mean(f1_scores), 3))
    print("Exact Match:", round(np.mean(em_scores), 3))


# ======================
# RUN
# ======================
if __name__ == "__main__":
    dataset = load_dataset()
    evaluate(dataset)