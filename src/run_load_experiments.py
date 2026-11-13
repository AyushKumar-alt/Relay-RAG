import sys
import os
import time
import csv
import json
import random
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from pipeline import RelayPipeline, GROQ_API_KEY
from standin_service import CalibratedStandinService

BASE_DIR = src_dir.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_CSV_PATH = BASE_DIR / "results.csv"
QUESTIONS_PATH = DATA_DIR / "benchmark_questions.json"
TRACE_LOG_PATH = DATA_DIR / "pipeline_trace_log.jsonl"

def calc_p(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return round(s[f], 1)
    return round(s[f] * (c - k) + s[c] * (k - f), 1)

def run_scenario(
    scenario_name: str,
    pipeline: RelayPipeline,
    questions: List[Dict[str, Any]],
    num_requests: int,
    concurrency: int,
    offered_rate_per_min: float,
    source: str = "standin"
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f" RUNNING SCENARIO: {scenario_name.upper()} ({source.upper()})")
    print(f" Requests: {num_requests} | Max Concurrency: {concurrency} | Offered Rate: {offered_rate_per_min}/min")
    print(f"=======================================================")

    results = []
    t_start = time.perf_counter()
    delay_between_requests = 60.0 / offered_rate_per_min if offered_rate_per_min > 0 else 0.0

    peak_queue = 0
    active_in_flight = 0
    queue_lock = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)

    def worker(idx: int, q_obj: Dict[str, Any]):
        nonlocal active_in_flight, peak_queue
        active_in_flight += 1
        if active_in_flight > peak_queue:
            peak_queue = active_in_flight

        req_id = f"{scenario_name[:4]}_{idx:04d}"
        q_text = q_obj["question"]

        try:
            res = pipeline.process_request(
                request_id=req_id,
                question=q_text,
                engine=source,
                scenario=scenario_name
            )
            return res
        except Exception as e:
            return {
                "request_id": req_id,
                "scenario": scenario_name,
                "source": source,
                "start_time": round(time.perf_counter(), 3),
                "first_token_ms": None,
                "refusal_latency_ms": 0.0,
                "total_ms": 0.0,
                "status": "error",
                "retries": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "retrieval_setting": "dense_bge_small_top5",
                "answer_grade": "wrong",
                "answer": str(e),
                "reason": "Exception in pipeline",
                "trace": {"error": str(e)}
            }
        finally:
            active_in_flight = max(0, active_in_flight - 1)

    futures = []
    for i in range(1, num_requests + 1):
        q = questions[(i - 1) % len(questions)]
        f = queue_lock.submit(worker, i, q)
        futures.append(f)
        if delay_between_requests > 0 and i < num_requests:
            time.sleep(delay_between_requests)

    for f in concurrent.futures.as_completed(futures):
        results.append(f.result())

    duration_sec = time.perf_counter() - t_start

    # STRICT METRIC RECONCILIATION:
    # Model TTFT is calculated strictly for requests where the model was invoked and first_token_ms is not None!
    model_invoked_results = [r for r in results if r["status"] == "ok" and r.get("first_token_ms") is not None]
    model_ttfts = [r["first_token_ms"] for r in model_invoked_results]
    
    # Total times for all successful requests
    totals = [r["total_ms"] for r in results if r["status"] == "ok"]

    succeeded = sum(1 for r in results if r["status"] == "ok")
    rate_limited = sum(1 for r in results if r["status"] == "rate_limited")
    errors_timeouts = sum(1 for r in results if r["status"] in ["timeout", "error"])

    retries_list = [r.get("retries", 0) for r in results]
    tokens_out_list = [r.get("tokens_out", 0) for r in results if r["status"] == "ok"]

    succ_pct = round((succeeded / num_requests) * 100, 1)
    rl_pct = round((rate_limited / num_requests) * 100, 1)
    timeout_pct = round((errors_timeouts / num_requests) * 100, 1)
    avg_retries = round(sum(retries_list) / num_requests, 2)
    avg_tokens = round(sum(tokens_out_list) / len(tokens_out_list), 1) if tokens_out_list else 0.0

    cost_per_1k = round((avg_tokens * 1000 / 1_000_000) * 0.05, 4)

    summary = {
        "scenario": scenario_name,
        "source": source,
        "requests_sent": num_requests,
        "duration_sec": round(duration_sec, 1),
        "offered_rate": offered_rate_per_min,
        "ttft_p50": calc_p(model_ttfts, 50) if model_ttfts else "N/A",
        "ttft_p95": calc_p(model_ttfts, 95) if model_ttfts else "N/A",
        "ttft_p99": calc_p(model_ttfts, 99) if model_ttfts else "N/A",
        "total_p50": calc_p(totals, 50),
        "total_p95": calc_p(totals, 95),
        "total_p99": calc_p(totals, 99),
        "succeeded_pct": succ_pct,
        "rate_limited_pct": rl_pct,
        "timeout_pct": timeout_pct,
        "retries_per_req": avg_retries,
        "peak_queue": peak_queue,
        "tokens_per_answer": avg_tokens,
        "cost_per_1k": cost_per_1k,
        "records": results
    }

    print(f"Scenario Finished in {duration_sec:.1f}s:")
    print(f"  Succeeded: {succ_pct}% | Rate-limited: {rl_pct}% | Timed out: {timeout_pct}%")
    ttft_str = f"{summary['ttft_p50']} / {summary['ttft_p95']} / {summary['ttft_p99']}"
    print(f"  Model TTFT p50/p95/p99 (excl. refusals): {ttft_str} ms")
    print(f"  Total p50/p95/p99: {summary['total_p50']} / {summary['total_p95']} / {summary['total_p99']} ms")
    print(f"  Peak In-flight Queue: {peak_queue} | Retries: {avg_retries}")

    return summary

def main():
    print("=" * 75)
    print("   THE RELAY: 5-SCENARIO LOAD EXPERIMENT HARNESS (AUDITED V3)")
    print("=" * 75)

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    all_records = []
    scenario_summaries = []

    # Scenario 0: Steady (Real Groq Service) - 5 requests
    if GROQ_API_KEY:
        pipeline_real = RelayPipeline()
        s0 = run_scenario(
            scenario_name="steady_real",
            pipeline=pipeline_real,
            questions=questions[:5],
            num_requests=5,
            concurrency=1,
            offered_rate_per_min=20.0,
            source="real"
        )
        all_records.extend(s0["records"])
        scenario_summaries.append(s0)

    # Scenario 1: Steady load (Stand-in, 20 requests)
    standin_steady = CalibratedStandinService(rpm_limit=1000, tpm_limit=8000)
    pipeline_steady = RelayPipeline(standin_service=standin_steady)
    s1 = run_scenario(
        scenario_name="steady",
        pipeline=pipeline_steady,
        questions=questions,
        num_requests=20,
        concurrency=2,
        offered_rate_per_min=30.0,
        source="standin"
    )
    all_records.extend(s1["records"])
    scenario_summaries.append(s1)

    # Scenario 2: Sudden burst (Stand-in, 30 requests)
    standin_burst = CalibratedStandinService(rpm_limit=1000, tpm_limit=8000)
    pipeline_burst = RelayPipeline(standin_service=standin_burst)
    s2 = run_scenario(
        scenario_name="burst",
        pipeline=pipeline_burst,
        questions=questions,
        num_requests=30,
        concurrency=10,
        offered_rate_per_min=600.0,
        source="standin"
    )
    all_records.extend(s2["records"])
    scenario_summaries.append(s2)

    # Scenario 3: Sustained overload (Stand-in, 40 requests)
    standin_overload = CalibratedStandinService(rpm_limit=1000, tpm_limit=8000)
    pipeline_overload = RelayPipeline(standin_service=standin_overload)
    s3 = run_scenario(
        scenario_name="sustained_overload",
        pipeline=pipeline_overload,
        questions=questions,
        num_requests=40,
        concurrency=12,
        offered_rate_per_min=1200.0,
        source="standin"
    )
    all_records.extend(s3["records"])
    scenario_summaries.append(s3)

    # Scenario 4: Per-minute limit halved (Stand-in, 25 requests)
    standin_halved = CalibratedStandinService(rpm_limit=500, tpm_limit=4000)
    pipeline_halved = RelayPipeline(standin_service=standin_halved)
    s4 = run_scenario(
        scenario_name="per_minute_limit_halved",
        pipeline=pipeline_halved,
        questions=questions,
        num_requests=25,
        concurrency=6,
        offered_rate_per_min=300.0,
        source="standin"
    )
    all_records.extend(s4["records"])
    scenario_summaries.append(s4)

    # Scenario 5: Daily budget exhausted (Stand-in, 15 requests - 15:00 UTC cutoff)
    standin_exhausted = CalibratedStandinService(rpm_limit=1000, tpm_limit=8000)
    standin_exhausted.set_daily_budget_exhausted(True)
    pipeline_exhausted = RelayPipeline(standin_service=standin_exhausted)
    s5 = run_scenario(
        scenario_name="daily_budget_exhausted",
        pipeline=pipeline_exhausted,
        questions=questions,
        num_requests=15,
        concurrency=3,
        offered_rate_per_min=180.0,
        source="standin"
    )
    all_records.extend(s5["records"])
    scenario_summaries.append(s5)

    # Write raw results.csv with NULL (empty string) for first_token_ms on non-provider requests
    csv_fields = [
        "request_id", "scenario", "source", "start_time", "first_token_ms",
        "total_ms", "status", "retries", "tokens_in", "tokens_out",
        "retrieval_setting", "answer_grade"
    ]
    with open(RESULTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in all_records:
            row = {}
            for k in csv_fields:
                val = r.get(k)
                if k == "first_token_ms" and val is None:
                    row[k] = ""  # Null representation in CSV
                else:
                    row[k] = val if val is not None else ""
            writer.writerow(row)

    # Write detailed pipeline trace log
    with open(TRACE_LOG_PATH, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 80)
    print(f"Successfully generated {len(all_records)} raw records to: {RESULTS_CSV_PATH.name}")
    print(f"Detailed pipeline trace log saved to: {TRACE_LOG_PATH.name}")
    print("=" * 80)

    # Print Table 2a
    print("\n" + "=" * 95)
    print("      TABLE 2a: LATENCY (MILLI-SECONDS, ONE ROW PER LOAD SCENARIO)")
    print("=" * 95)
    header_2a = f"{'Scenario':<26} | {'Source':<8} | {'Reqs':<5} | {'Dur (s)':<7} | {'Offered/m':<9} | {'First token p50/p95/p99 (ms)':<30} | {'Total time p50/p95/p99 (ms)'}"
    print(header_2a)
    print("-" * 120)
    for s in scenario_summaries:
        ft_str = f"{s['ttft_p50']} / {s['ttft_p95']} / {s['ttft_p99']}" if s['ttft_p50'] != "N/A" else "N/A (No model invoked)"
        tot_str = f"{s['total_p50']} / {s['total_p95']} / {s['total_p99']}"
        print(f"{s['scenario']:<26} | {s['source']:<8} | {s['requests_sent']:<5} | {s['duration_sec']:<7} | {s['offered_rate']:<9} | {ft_str:<30} | {tot_str}")

    # Print Table 2b
    print("\n" + "=" * 95)
    print("      TABLE 2b: OUTCOMES (ONE ROW PER LOAD SCENARIO)")
    print("=" * 95)
    header_2b = f"{'Scenario':<26} | {'Succ %':<7} | {'RateLim %':<9} | {'Timeout %':<9} | {'Retries':<7} | {'Queue':<6} | {'Tok/Ans':<8} | {'Cost/1k ($)'}"
    print(header_2b)
    print("-" * 105)
    for s in scenario_summaries:
        print(f"{s['scenario']:<26} | {s['succeeded_pct']:>5.1f}% | {s['rate_limited_pct']:>7.1f}% | {s['timeout_pct']:>7.1f}% | {s['retries_per_req']:>7.2f} | {s['peak_queue']:>6} | {s['tokens_per_answer']:>8.1f} | ${s['cost_per_1k']:<8.4f}")

    print("=" * 95)

if __name__ == "__main__":
    main()
