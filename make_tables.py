import sys
import csv
import math
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

def load_results(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed = dict(r)
            parsed["start_time"] = float(r["start_time"]) if r.get("start_time") else 0.0
            parsed["first_token_ms"] = float(r["first_token_ms"]) if r.get("first_token_ms") and r["first_token_ms"].strip() else None
            parsed["total_ms"] = float(r["total_ms"]) if r.get("total_ms") and r["total_ms"].strip() else 0.0
            parsed["retries"] = int(r["retries"]) if r.get("retries") else 0
            parsed["tokens_in"] = int(r["tokens_in"]) if r.get("tokens_in") else 0
            parsed["tokens_out"] = int(r["tokens_out"]) if r.get("tokens_out") else 0
            rows.append(parsed)
    return rows

def pct(data: List[float], p: float) -> str:
    if not data:
        return "n/a"
    val = np.percentile(data, p, method="linear")
    return f"{val:,.1f}"

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found.")
        sys.exit(1)

    rows = load_results(csv_file)
    print(f"Loaded {len(rows)} rows from {csv_file}\n")

    # Scenarios in order
    scenarios_order = [
        "steady_real",
        "steady",
        "burst",
        "sustained_overload",
        "per_minute_limit_halved",
        "daily_budget_exhausted"
    ]

    # --- TABLE 2a: Latency ---
    print("### Table 2a: Latency (ms)")
    print("| Scenario | Source | Reqs | Dur (s) | Offered /min | First token p50 / p95 / p99 (n) | Total p50 / p95 / p99 |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for sc in scenarios_order:
        sc_rows = [r for r in rows if r["scenario"] == sc]
        if not sc_rows:
            continue
        source = sc_rows[0]["source"]
        reqs = len(sc_rows)
        
        # Duration: max completion time - min start time
        starts = [r["start_time"] for r in sc_rows]
        ends = [r["start_time"] + (r["total_ms"] / 1000.0) for r in sc_rows]
        dur = max(ends) - min(starts) if starts else 0.0
        offered = (reqs / dur * 60.0) if dur > 0 else 0.0

        # First token stats (only where model was invoked / first_token_ms not None)
        ft_vals = [r["first_token_ms"] for r in sc_rows if r["first_token_ms"] is not None]
        n_ft = len(ft_vals)
        if n_ft > 0:
            ft_str = f"{pct(ft_vals, 50)} / {pct(ft_vals, 95)} / {pct(ft_vals, 99)} ({n_ft})"
        else:
            ft_str = "n/a (no model call)"

        # Total time stats across all requests
        tot_vals = [r["total_ms"] for r in sc_rows]
        tot_str = f"{pct(tot_vals, 50)} / {pct(tot_vals, 95)} / {pct(tot_vals, 99)}"

        print(f"| {sc} | {source} | {reqs} | {dur:.1f} | {offered:.1f} | {ft_str} | {tot_str} |")

    # --- TABLE 2b: Outcomes ---
    print("\n### Table 2b: Outcomes")
    print("| Scenario | OK % | Verified ans % | Rate-limited / rejected % | GivenUp % | Retries/req | Peak queue | Tokens in/out per ans | Cost/1,000 ans ($) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for sc in scenarios_order:
        sc_rows = [r for r in rows if r["scenario"] == sc]
        if not sc_rows:
            continue
        reqs = len(sc_rows)
        ok_cnt = sum(1 for r in sc_rows if r["status"] == "ok")
        verified_cnt = sum(1 for r in sc_rows if r["answer_grade"] == "correct")
        ratelim_cnt = sum(1 for r in sc_rows if r["status"] == "rate_limited")
        givenup_cnt = sum(1 for r in sc_rows if r["status"] in ["timeout", "error"])
        retries_tot = sum(r["retries"] for r in sc_rows)

        ok_pct = (ok_cnt / reqs) * 100.0
        verified_pct = (verified_cnt / reqs) * 100.0
        ratelim_pct = (ratelim_cnt / reqs) * 100.0
        givenup_pct = (givenup_cnt / reqs) * 100.0
        retries_per_req = retries_tot / reqs

        # Peak in-flight
        events = []
        for r in sc_rows:
            events.append((r["start_time"], 1))
            events.append((r["start_time"] + (r["total_ms"] / 1000.0), -1))
        events.sort(key=lambda x: (x[0], -x[1]))
        peak_inflight = 0
        cur = 0
        for t, d in events:
            cur += d
            if cur > peak_inflight:
                peak_inflight = cur

        # Answered requests (ok and tokens_in > 0)
        ans_rows = [r for r in sc_rows if r["status"] == "ok" and r["tokens_in"] > 0]
        if ans_rows:
            avg_in = sum(r["tokens_in"] for r in ans_rows) / len(ans_rows)
            avg_out = sum(r["tokens_out"] for r in ans_rows) / len(ans_rows)
            cost_1k = ((avg_in * 0.15) + (avg_out * 0.60)) / 1_000_000 * 1000.0
            tok_str = f"{avg_in:,.0f} / {avg_out:.1f}"
            cost_str = f"${cost_1k:.3f}"
        else:
            tok_str = "0 / 0"
            cost_str = "n/a (0 ans)"

        print(f"| {sc} | {ok_pct:.1f}% | {verified_pct:.1f}% | {ratelim_pct:.1f}% | {givenup_pct:.1f}% | {retries_per_req:.2f} | {peak_inflight} | {tok_str} | {cost_str} |")

    # --- TABLE 3: Retrieval Settings ---
    print("\n### Table 3: Retrieval settings (evaluated across 15 questions)")
    print("| Setting | Qs | Factual correct & cited (n=12) | Incorrect factual (n=12) | Correct refusals (n=3) | Tok/Ans | First token p50 / p95 | Total p50 / p95 |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    ret_settings = [
        ("BM25, 350 tok, top-5, no re-rank", "bm25_top5"),
        ("Dense BGE-small, 350 tok, top-5, no re-rank", "dense_top5"),
        ("Hybrid BM25+dense, RRF k=60, no re-rank", "hybrid_rrf_top5"),
        ("Dense BGE-small, 350 tok, top-2, no re-rank", "dense_top2")
    ]

    for label, set_name in ret_settings:
        set_rows = [r for r in rows if r["retrieval_setting"] == set_name and r["scenario"] == "retrieval_eval"]
        if not set_rows:
            continue
        qs = len(set_rows)
        corr_cnt = sum(1 for r in set_rows if r["answer_grade"] == "correct")
        wrong_cnt = sum(1 for r in set_rows if r["answer_grade"] == "wrong")
        ref_corr = sum(1 for r in set_rows if r["answer_grade"] == "refused_correctly")

        corr_pct = (corr_cnt / 12.0) * 100.0
        wrong_pct = (wrong_cnt / 12.0) * 100.0
        ref_pct = (ref_corr / 3.0) * 100.0

        ans_rows = [r for r in set_rows if r["status"] == "ok" and r["tokens_out"] > 0]
        tok_out = (sum(r["tokens_out"] for r in ans_rows) / len(ans_rows)) if ans_rows else 0.0

        ft_vals = [r["first_token_ms"] for r in set_rows if r["first_token_ms"] is not None]
        tot_vals = [r["total_ms"] for r in set_rows]

        ft_str = f"{pct(ft_vals, 50)} / {pct(ft_vals, 95)}"
        tot_str = f"{pct(tot_vals, 50)} / {pct(tot_vals, 95)}"

        print(f"| {label} | {qs} | {corr_pct:.1f}% | {wrong_pct:.1f}% | {ref_pct:.1f}% | {tok_out:.1f} | {ft_str} | {tot_str} |")

if __name__ == "__main__":
    main()

