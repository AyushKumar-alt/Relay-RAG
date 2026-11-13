import sys
import time
import json
from pathlib import Path

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from standin_service import CalibratedStandinService

BASE_DIR = src_dir.parent
DATA_DIR = BASE_DIR / "data"
TRACE_OUTPUT_PATH = DATA_DIR / "standin_verification_trace.json"

def main():
    print("=" * 70)
    print("      INDEPENDENT STAND-IN SERVICE VERIFICATION SUITE")
    print("=" * 70)

    trace_results = {}

    # -------------------------------------------------------------
    # Test 1: Baseline Streaming & TTFT Measurement
    # -------------------------------------------------------------
    print("\n[Test 1] Baseline Streaming & TTFT Measurement...")
    standin = CalibratedStandinService(rpm_limit=1000, tpm_limit=8000, base_ttft_p50_ms=690.0)
    messages = [{"role": "user", "content": "What is multi-head attention?"}]
    
    t0 = time.perf_counter()
    events = list(standin.stream_chat_completion(messages, max_tokens=60))
    t_tot = (time.perf_counter() - t0) * 1000

    first_tok = next((e for e in events if e.get("type") == "first_token"), None)
    finish_ev = next((e for e in events if e.get("type") == "finish"), None)

    ttft = first_tok["ttft_ms"] if first_tok else 0.0
    tokens_out = finish_ev.get("tokens_out", 0) if finish_ev else 0

    print(f"  -> TTFT: {ttft:.1f} ms | Total Time: {t_tot:.1f} ms | Tokens Streamed: {tokens_out}")
    assert ttft > 0, "TTFT must be positive"
    assert tokens_out > 0, "Must stream tokens"
    trace_results["test1_baseline"] = {
        "ttft_ms": round(ttft, 1),
        "total_ms": round(t_tot, 1),
        "tokens_out": tokens_out,
        "status": "PASS"
    }

    # -------------------------------------------------------------
    # Test 2: Load-Dependent Concurrency Slowdown
    # Model: TTFT_eff = TTFT_base * exp(jitter) * (1 + alpha * (N - 1))
    # -------------------------------------------------------------
    print("\n[Test 2] Load-Dependent Concurrency Slowdown...")
    alpha = 0.15
    standin_slowdown = CalibratedStandinService(concurrency_penalty_alpha=alpha)
    
    # N=1 concurrency (multiplier = 1.0)
    standin_slowdown.in_flight = 1
    ev_c1 = list(standin_slowdown.stream_chat_completion(messages, max_tokens=10))
    ttft_c1 = next(e for e in ev_c1 if e["type"] == "first_token")["ttft_ms"]

    # N=6 concurrency (multiplier = 1 + 0.15 * 5 = 1.75)
    standin_slowdown.in_flight = 6
    ev_c6 = list(standin_slowdown.stream_chat_completion(messages, max_tokens=10))
    ttft_c6 = next(e for e in ev_c6 if e["type"] == "first_token")["ttft_ms"]

    standin_slowdown.in_flight = 0
    empirical_ratio = round(ttft_c6 / ttft_c1, 2)
    expected_ratio = round(1.0 + alpha * (6 - 1), 2)  # 1.75

    print(f"  -> Concurrency N=1 TTFT: {ttft_c1:.1f} ms")
    print(f"  -> Concurrency N=6 TTFT: {ttft_c6:.1f} ms")
    print(f"  -> Observed Ratio: {empirical_ratio}x (Theoretical algebraic multiplier: {expected_ratio}x, delta due to log-normal jitter)")
    
    assert ttft_c6 > ttft_c1, "High concurrency must degrade TTFT"
    trace_results["test2_concurrency_slowdown"] = {
        "formula": "TTFT_eff = TTFT_base * exp(epsilon) * (1 + alpha * max(0, N - 1))",
        "alpha": alpha,
        "jitter_sigma": 0.08,
        "ttft_concurrency_1_ms": round(ttft_c1, 1),
        "ttft_concurrency_6_ms": round(ttft_c6, 1),
        "empirical_inflation_ratio": empirical_ratio,
        "theoretical_multiplier": expected_ratio,
        "status": "PASS"
    }

    # -------------------------------------------------------------
    # Test 3: Rate Limiting & Authentic 429 Payloads
    # -------------------------------------------------------------
    print("\n[Test 3] Rate Limit Throttling (RPM & TPM Token Bucket)...")
    standin_rl = CalibratedStandinService(rpm_limit=2, tpm_limit=500)
    
    res1 = list(standin_rl.stream_chat_completion(messages, max_tokens=30))
    res2 = list(standin_rl.stream_chat_completion(messages, max_tokens=30))
    res3 = list(standin_rl.stream_chat_completion(messages, max_tokens=30))

    err3 = next((e for e in res3 if e.get("type") == "error"), None)
    print(f"  -> Request 1: Succeeded (Tokens recorded: {standin_rl.current_minute_tokens})")
    print(f"  -> Request 2: Succeeded (Requests recorded: {standin_rl.current_minute_requests})")
    print(f"  -> Request 3: HTTP {err3['payload']['status_code']} - {err3['payload']['error']['message']}")
    print(f"  -> Headers returned: {err3['payload']['headers']}")
    
    assert err3 is not None and err3["payload"]["status_code"] == 429, "Request 3 must return HTTP 429"
    trace_results["test3_rate_limiting_429"] = {
        "status_code": err3["payload"]["status_code"],
        "error_type": err3["payload"]["error"]["type"],
        "retry_after": err3["payload"]["headers"].get("retry-after"),
        "status": "PASS"
    }

    # -------------------------------------------------------------
    # Test 4: Daily Budget Exhaustion (The 15:00 UTC / 3:00 PM Scenario)
    # -------------------------------------------------------------
    print("\n[Test 4] Simulated Daily Budget Exhaustion (15:00 UTC Cutoff)...")
    standin_daily = CalibratedStandinService()
    standin_daily.set_daily_budget_exhausted(True)

    res_daily = list(standin_daily.stream_chat_completion(messages, max_tokens=20))
    err_daily = next(e for e in res_daily if e.get("type") == "error")

    print(f"  -> Daily exhausted status: HTTP {err_daily['payload']['status_code']}")
    print(f"  -> Daily error message:   {err_daily['payload']['error']['message']}")
    print(f"  -> Retry-after:           {err_daily['payload']['headers']['retry-after']}s (9 hours from 15:00 UTC to 00:00 UTC midnight reset)")
    
    assert err_daily["payload"]["error"]["type"] == "daily_quota_exceeded"
    trace_results["test4_daily_exhaustion"] = {
        "status_code": err_daily["payload"]["status_code"],
        "error_type": err_daily["payload"]["error"]["type"],
        "message": err_daily["payload"]["error"]["message"],
        "timezone_clarification": "15:00 UTC (3:00 PM UTC) with 9-hour reset window to 00:00 UTC next day",
        "retry_after_sec": err_daily["payload"]["headers"]["retry-after"],
        "status": "PASS"
    }

    with open(TRACE_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(trace_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f" Stand-in Verification Trace saved to: {TRACE_OUTPUT_PATH.name}")
    print(" ALL 4 STAND-IN VERIFICATION TESTS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
