import json
import time
from pathlib import Path
from retriever import BM25Retriever

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
QUESTIONS_PATH = DATA_DIR / "benchmark_questions.json"

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

def main():
    print("=" * 65)
    print("      PHASE 4: BM25 RETRIEVAL BASELINE EVALUATION")
    print("=" * 65)

    # 1. Load chunks
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} retrieval chunks.")

    # 2. Build BM25 Index
    t0 = time.perf_counter()
    retriever = BM25Retriever(k1=1.5, b=0.75)
    retriever.fit(chunks)
    index_build_ms = (time.perf_counter() - t0) * 1000
    print(f"BM25 index built in {index_build_ms:.2f} ms.")

    # 3. Load Questions
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Evaluating {len(questions)} gold benchmark questions...\n")

    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks = []
    latencies_ms = []

    answerable_count = 0

    print(f"{'ID':<4} | {'Target Paper':<16} | {'Hit@5':<5} | {'Hit@10':<6} | {'RR':<5} | {'Latency':<8} | {'Top Match Section'}")
    print("-" * 75)

    for q in questions:
        q_id = q["id"]
        q_text = q["question"]
        target_paper = q["target_paper"]
        is_refusal = q["is_refusal"]

        t_start = time.perf_counter()
        results = retriever.search(q_text, top_k=10)
        q_latency = (time.perf_counter() - t_start) * 1000
        latencies_ms.append(q_latency)

        if is_refusal:
            # For unanswerable questions, we check if top score is modest or if top matches lack the target
            top_sec = results[0]["section_title"] if results else "None"
            top_score = results[0]["bm25_score"] if results else 0.0
            print(f"{q_id:<4} | {'[REFUSAL TRAP]':<16} | {'N/A':<5} | {'N/A':<6} | {'N/A':<5} | {q_latency:6.2f}ms | Top score: {top_score:.1f} ({top_sec[:25]})")
            continue

        answerable_count += 1
        found_rank = None

        for rank, res in enumerate(results, start=1):
            # Check if retrieved chunk matches target paper
            if res["paper_id"] == target_paper:
                # Check for keyword overlap in text
                text_lower = res["text"].lower()
                matched_kw = any(kw.lower() in text_lower for kw in q.get("expected_answer_keywords", []))
                sec_match = any(w in res.get("section_title", "").lower() for w in q.get("target_section", "").lower().split("/"))
                if matched_kw or sec_match:
                    found_rank = rank
                    break

        if found_rank is not None and found_rank <= 5:
            hits_at_5 += 1
        if found_rank is not None and found_rank <= 10:
            hits_at_10 += 1

        rr = (1.0 / found_rank) if found_rank else 0.0
        reciprocal_ranks.append(rr)

        top_sec = results[0]["section_title"] if results else "None"
        h5_str = "YES" if (found_rank and found_rank <= 5) else "NO"
        h10_str = "YES" if (found_rank and found_rank <= 10) else "NO"
        rr_str = f"{rr:.2f}" if found_rank else "0.00"

        print(f"{q_id:<4} | {target_paper:<16} | {h5_str:<5} | {h10_str:<6} | {rr_str:<5} | {q_latency:6.2f}ms | {top_sec[:30]}")

    recall_at_5 = (hits_at_5 / answerable_count) * 100 if answerable_count > 0 else 0
    recall_at_10 = (hits_at_10 / answerable_count) * 100 if answerable_count > 0 else 0
    mrr_10 = (sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0

    print("\n" + "=" * 65)
    print("             RETRIEVAL BENCHMARK METRICS SUMMARY            ")
    print("=" * 65)
    print(f"Total Questions Evaluated:  {len(questions)} ({answerable_count} factual, {len(questions) - answerable_count} traps/refusal)")
    print(f"Recall@5:                   {recall_at_5:.1f}% ({hits_at_5}/{answerable_count})")
    print(f"Recall@10:                  {recall_at_10:.1f}% ({hits_at_10}/{answerable_count})")
    print(f"MRR@10:                     {mrr_10:.3f}")
    print("\n--- Retrieval Latency Distribution (ms) ---")
    print(f"  Min:  {min(latencies_ms):.2f} ms")
    print(f"  p50:  {calc_percentile(latencies_ms, 50):.2f} ms (Median)")
    print(f"  p90:  {calc_percentile(latencies_ms, 90):.2f} ms")
    print(f"  p95:  {calc_percentile(latencies_ms, 95):.2f} ms")
    print(f"  Max:  {max(latencies_ms):.2f} ms")
    print("=" * 65)

if __name__ == "__main__":
    main()

