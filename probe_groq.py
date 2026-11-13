import os
import time
import json
import math
import httpx
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY") or os.getenv("Grok_API_Key") or os.getenv("GROK_API_KEY")
if not api_key:
    print("ERROR: Groq API key not found in .env")
    exit(1)

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

model_name = "openai/gpt-oss-120b"
url_chat = "https://api.groq.com/openai/v1/chat/completions"

NUM_RUNS = 12
DELAY_BETWEEN_RUNS = 1.5  # seconds

def calc_percentile(data, p):
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)

print(f"============================================================")
print(f" PROBING GROQ SERVICE: {model_name}")
print(f" Iterations: {NUM_RUNS} | Delay between calls: {DELAY_BETWEEN_RUNS}s")
print(f"============================================================\n")

ttft_list = []
total_time_list = []
headers_history = []

for i in range(1, NUM_RUNS + 1):
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a research assistant."},
            {"role": "user", "content": "State one interesting property of neural networks in 15 words."}
        ],
        "max_tokens": 40,
        "stream": True
    }

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    captured_headers = {}

    try:
        with httpx.Client(timeout=30.0) as client:
            with client.stream("POST", url_chat, headers=headers, json=payload) as response:
                status_code = response.status_code
                
                # Capture rate limit headers
                for k, v in response.headers.items():
                    if any(x in k.lower() for x in ["ratelimit", "retry", "limit", "remaining", "reset"]):
                        captured_headers[k] = v

                if status_code != 200:
                    print(f"[{i}/{NUM_RUNS}] HTTP Error {status_code}: {response.read().decode('utf-8')}")
                    time.sleep(DELAY_BETWEEN_RUNS)
                    continue

                for line in response.iter_lines():
                    if line.startswith("data: ") and line.strip() != "data: [DONE]":
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_count += 1

        end_time = time.perf_counter()
        
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
        tot_ms = (end_time - start_time) * 1000
        
        ttft_list.append(ttft_ms)
        total_time_list.append(tot_ms)
        headers_history.append(captured_headers)

        rem_req = captured_headers.get("x-ratelimit-remaining-requests", "N/A")
        rem_tok = captured_headers.get("x-ratelimit-remaining-tokens", "N/A")
        rst_tok = captured_headers.get("x-ratelimit-reset-tokens", "N/A")

        print(f"Run {i:2d}/{NUM_RUNS} -> TTFT: {ttft_ms:6.1f} ms | Total: {tot_ms:6.1f} ms | Tokens: ~{token_count:2d} | RemReq: {rem_req} | RemTok: {rem_tok} | ResetTok: {rst_tok}")

    except Exception as e:
        print(f"Run {i:2d}/{NUM_RUNS} failed: {e}")

    if i < NUM_RUNS:
        time.sleep(DELAY_BETWEEN_RUNS)

print("\n" + "="*60)
print("             GROQ BENCHMARK STATISTICAL SUMMARY            ")
print("="*60)

if ttft_list:
    print(f"Total Successful Runs: {len(ttft_list)}/{NUM_RUNS}")
    print("\n--- Time to First Token (TTFT) ---")
    print(f"  Min:  {min(ttft_list):6.1f} ms")
    print(f"  p50:  {calc_percentile(ttft_list, 50):6.1f} ms  (Median)")
    print(f"  p90:  {calc_percentile(ttft_list, 90):6.1f} ms")
    print(f"  p95:  {calc_percentile(ttft_list, 95):6.1f} ms")
    print(f"  Max:  {max(ttft_list):6.1f} ms")

    print("\n--- Total Response Time ---")
    print(f"  Min:  {min(total_time_list):6.1f} ms")
    print(f"  p50:  {calc_percentile(total_time_list, 50):6.1f} ms  (Median)")
    print(f"  p90:  {calc_percentile(total_time_list, 90):6.1f} ms")
    print(f"  p95:  {calc_percentile(total_time_list, 95):6.1f} ms")
    print(f"  Max:  {max(total_time_list):6.1f} ms")

    print("\n--- Rate Limit Metadata (From Latest Successful Header) ---")
    latest_h = headers_history[-1] if headers_history else {}
    for k, v in latest_h.items():
        print(f"  {k}: {v}")

print("="*60)
