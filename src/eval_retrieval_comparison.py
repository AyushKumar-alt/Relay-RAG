import json
import time
from pathlib import Path
import numpy as np
from fastembed import TextEmbedding
from retriever import BM25Retriever

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
QUESTIONS_PATH = DATA_DIR / "benchmark_questions.json"
EMBEDDINGS_PATH = DATA_DIR / "dense_embeddings_bge.npy"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

def calc_percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)

class DenseRetriever:
    def __init__(self, chunks, embeddings_matrix, model_name=MODEL_NAME):
        self.chunks = chunks
        self.matrix = embeddings_matrix  # Pre-normalized L2
        self.model = TextEmbedding(model_name=model_name)

    def search(self, query: str, top_k: int = 10):
        # Embed query (BGE query instruction if needed, or raw query)
        q_emb = list(self.model.embed([query]))[0]
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        scores = np.dot(self.matrix, q_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = dict(self.chunks[idx])
            chunk["dense_score"] = float(round(scores[idx], 4))
            chunk["rank"] = rank
            results.append(chunk)
        return results

class HybridRRFRetriever:
    def __init__(self, bm25_retriever: BM25Retriever, dense_retriever: DenseRetriever, rrf_k: int = 60):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 10):
        # Fetch larger candidate pool for RRF
        pool_size = max(top_k * 3, 25)
        bm25_res = self.bm25.search(query, top_k=pool_size)
        dense_res = self.dense.search(query, top_k=pool_size)

        rrf_scores = {}
        chunks_map = {}

        for rank, item in enumerate(bm25_res, start=1):
            cid = item["chunk_id"]
            chunks_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, item in enumerate(dense_res, start=1):
            cid = item["chunk_id"]
            chunks_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        results = []
        for rank, cid in enumerate(sorted_cids[:top_k], start=1):
            chunk = dict(chunks_map[cid])
            chunk["rrf_score"] = round(rrf_scores[cid], 5)
            chunk["rank"] = rank
            results.append(chunk)

        return results

def evaluate_retriever(name, retriever_obj, questions):
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks = []
    latencies = []
    answerable_count = 0

    for q in questions:
        target_paper = q["target_paper"]
        is_refusal = q["is_refusal"]

        t0 = time.perf_counter()
        results = retriever_obj.search(q["question"], top_k=10)
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)

        if is_refusal:
            continue

        answerable_count += 1
        found_rank = None

        for rank, res in enumerate(results, start=1):
            if res["paper_id"] == target_paper:
                text_lower = res["text"].lower()
                matched_kw = any(kw.lower() in text_lower for kw in q.get("expected_answer_keywords", []))
                sec_match = any(w in res.get("section_title", "").lower() for w in q.get("target_section", "").lower().split("/"))
                if matched_kw or sec_match:
                    found_rank = rank
                    break

        if found_rank and found_rank <= 5:
            hits_at_5 += 1
        if found_rank and found_rank <= 10:
            hits_at_10 += 1

        rr = (1.0 / found_rank) if found_rank else 0.0
        reciprocal_ranks.append(rr)

    recall_5 = (hits_at_5 / answerable_count) * 100 if answerable_count else 0
    recall_10 = (hits_at_10 / answerable_count) * 100 if answerable_count else 0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0

    return {
        "name": name,
        "recall_at_5": round(recall_5, 1),
        "recall_at_10": round(recall_10, 1),
        "mrr": round(mrr, 3),
        "p50_latency_ms": round(calc_percentile(latencies, 50), 2),
        "p95_latency_ms": round(calc_percentile(latencies, 95), 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2)
    }

def main():
    print("=" * 75)
    print("   HEAD-TO-HEAD RETRIEVAL EVALUATION: BM25 vs DENSE vs HYBRID (RRF)")
    print("=" * 75)

    # Load corpus & questions
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if not EMBEDDINGS_PATH.exists():
        print(f"[ERROR] Precomputed embeddings not found at {EMBEDDINGS_PATH.name}!")
        print("Please run 'python src/dense_indexer.py' first.")
        return

    embeddings_matrix = np.load(EMBEDDINGS_PATH)
    print(f"Loaded {len(chunks)} chunks and {embeddings_matrix.shape} precomputed embeddings.")

    # 1. Initialize systems
    print("Initializing BM25 retriever...")
    bm25 = BM25Retriever()
    bm25.fit(chunks)

    print(f"Initializing Dense retriever ({MODEL_NAME})...")
    dense = DenseRetriever(chunks, embeddings_matrix, model_name=MODEL_NAME)

    print("Initializing Hybrid RRF retriever (k=60)...")
    hybrid = HybridRRFRetriever(bm25, dense, rrf_k=60)

    # 2. Evaluate each
    print("\nRunning benchmark across 15 questions for each retriever...")
    res_bm25 = evaluate_retriever("BM25 (Lexical)", bm25, questions)
    res_dense = evaluate_retriever("Dense (BGE-Small)", dense, questions)
    res_hybrid = evaluate_retriever("Hybrid (BM25 + Dense RRF)", hybrid, questions)

    # 3. Print side-by-side comparison table
    print("\n" + "=" * 80)
    print(f"{'Method':<26} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR':<8} | {'p50 (ms)':<10} | {'p95 (ms)':<10}")
    print("-" * 80)
    for r in [res_bm25, res_dense, res_hybrid]:
        print(f"{r['name']:<26} | {r['recall_at_5']:>8.1f}% | {r['recall_at_10']:>8.1f}% | {r['mrr']:>8.3f} | {r['p50_latency_ms']:>8.2f}ms | {r['p95_latency_ms']:>8.2f}ms")
    print("=" * 80)

    # Save results to data/retrieval_comparison_results.json
    out_path = DATA_DIR / "retrieval_comparison_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([res_bm25, res_dense, res_hybrid], f, indent=2)
    print(f"\nDetailed metrics saved to: {out_path.name}")

if __name__ == "__main__":
    main()

