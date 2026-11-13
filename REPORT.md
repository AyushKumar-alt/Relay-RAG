# The Relay: Resilient RAG Pipeline & Load Study

*Cited research assistant over three arXiv papers, built on Groq's free tier and tested against a calibrated stand-in*

## 1. What I built and why

The assistant answers questions about three pinned arXiv papers (632 paragraphs, cut into 191 chunks that keep their page and section). Each question goes through four steps:

- **Retrieve:** `search_library` embeds the query with `BAAI/bge-small-en-v1.5` (384-dim, normalized) and returns the top 5 chunks.
- **Gate:** before any model call, a check refuses the question if its entities do not appear in the retrieved passages.
- **Synthesize:** Groq's `openai/gpt-oss-120b` (or my stand-in) streams an answer with citations in `[Paper, Section, Page]` form.
- **Validate:** a checker compares every citation against the chunks. On a mismatch it retries once with corrective feedback, then refuses.

**What "fast" means here:** first token within 2,000 ms and a full answer within 5,000 ms. The live full-RAG runs missed both: median first token was 5,058.8 ms and median total 6,585.5 ms. Short probes (25 input tokens) had a p50 of about 690 ms, so the main thing that changed between the two runs is prompt size: 25 tokens versus about 1,837.

**The decision that shaped the most:** putting the gate before generation. Dense similarity scores sit at 0.77–0.82 even for off-topic questions such as LLaMA-3's batch size, so retrieval scores alone cannot separate "answerable" from "not". Refusing up front costs zero model tokens. The price is that a heuristic gate may wrongly refuse a phrasing I did not test; on my 12 answerable benchmark questions it refused none.

## 2. The real service and my stand-in

I took limits from live response headers and Groq's documentation (Table 1). Tokens per minute is the limit that bites first. One answered request uses about 1,837 input and 192 output tokens (2,029 total), so an 8,000 TPM budget allows roughly 3.9 answers per minute.

**Two latency profiles.** Twelve warmed short probes (25 tokens in, 22 out) gave a first-token p50 of 692.6 ms and p95 of 780.1 ms, streaming at about 160 tok/s. I set the stand-in's base time-to-first-token (TTFT) to 690 ms from that. Full RAG requests of about 1,837 input tokens had a median TTFT of 5,058.8 ms. The API does not split that into prefill, queueing and inference, so I cannot say which part dominates.

**Stand-in (`standin_service.py`):** TTFT = 690 ms × lognormal noise (σ = 0.08) × (1 + 0.15 × max(0, in-flight − 1)); 25 ms per streamed token; a token bucket (8,000 TPM) plus a request bucket (30 RPM) that return 429-style replies.

**Where it is wrong:** it has no large-prompt prefill or reasoning delay; σ = 0.08 is wider than the measured 0.050; the 0.15 concurrency factor is my assumption; it streams at 40 tok/s against 160 measured; load-test answers are canned text; and it models no packet loss or cloud timeouts.

### Table 1: Limits found in the real service

| Limit | Value | How I found it | Confidence |
| :--- | :--- | :--- | :--- |
| Requests/min | 30 | Groq docs and console; not in headers | Medium |
| Tokens/min | 8,000 | Header `x-ratelimit-limit-tokens` | High |
| Requests/day | 1,000 | Header `x-ratelimit-limit-requests`; resets 00:00 UTC | High |
| Tokens/day | 200,000 | Groq docs; not in headers | Medium |
| Concurrency cap | None seen | Bursts hit token 429s first, so a cap never showed | Low |
| On breach | HTTP 429 | `rate_limit_exceeded` plus a `retry-after` header | High |
| Recovery | about 1.1 s (TPM) | Header `x-ratelimit-reset-tokens`; daily limit resets 00:00 UTC | High |

## 3. Results

Live runs (`steady_real`) were capped by free-tier access at 5 requests, so they are observations, not a latency distribution. The higher-volume tests ran against the stand-in. Total time includes fast rejections (11–16 ms), which is why overload p50s are around 12 ms. Answered stand-in requests take 1.7–1.8 s; real ones took 3.7–8.2 s. With 2–4 model calls per row, p95 is only indicative and I left p99 out.

### Table 2a: Latency (ms)

| Scenario | Source | Reqs | Offered/min | First token p50 / p95 (n) | Total p50 / p95 |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `steady_real` | real | 5 | 10.1 | 5,058.8 / 6,020.0 (4) | 6,585.5 / 8,249.7 |
| `steady` | stand-in | 20 | 31.6 | 703.8 / 749.4 (3) | 24.9 / 1,747.3 |
| `burst` | stand-in | 30 | 612.2 | 736.8 / 851.8 (3) | 13.2 / 1,797.1 |
| `sustained_overload` | stand-in | 40 | 1,183.1 | 839.5 / 877.8 (3) | 12.2 / 1,779.2 |
| `per_minute_limit_halved` | stand-in | 25 | 311.0 | 702.5 / 714.0 (2) | 11.9 / 1,388.4 |
| `daily_budget_exhausted` | stand-in | 15 | 191.7 | no model call | 12.0 / 16.0 |

### Table 2b: Outcomes

| Scenario | OK % | Verified % | Rejected % | Retries/req | Peak queue | Tokens in / out | Est. cost per 1,000 answers |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `steady_real` | 80.0 | 80.0 | 20.0 | 0.00 | 2 | 1,837 / 192 | $0.391 |
| `steady` | 30.0 | 15.0 | 70.0 | 0.05 | 1 | 1,757 / 40 | $0.288 |
| `burst` | 30.0 | 10.0 | 70.0 | 0.03 | 5 | 1,757 / 40 | $0.288 |
| `sustained_overload` | 22.5 | 7.5 | 77.5 | 0.03 | 5 | 1,757 / 40 | $0.288 |
| `per_minute_limit_halved` | 20.0 | 8.0 | 80.0 | 0.00 | 3 | 1,747 / 40 | $0.286 |
| `daily_budget_exhausted` | 20.0 | 0.0 | 80.0 | 0.00 | 1 | 0 / 0 | n/a |

OK % counts gate refusals (3 in steady, 6 in burst, 6 in overload, 3 in halved, 3 in daily). Verified % is graded-correct answers divided by requests sent. No request gave up in any scenario. The 80% in the daily scenario is quota exhaustion; the other rejections are per-minute token-bucket drops. Peak queue is derived from `start_time + total_ms`. Cost uses $0.15/M input and $0.60/M output tokens as a price-equivalent; nothing was actually billed on the free tier.

**Numbers I would not have guessed:** (a) at a steady 30 requests/min, only 3 of 20 got an answer, and under overload 3 of 40, so the token limit is the bottleneck, not latency; (b) going from top-5 to top-2 chunks cut prompt tokens by 56% (mean 1,610 to 709) with no loss in answerability; (c) live `gpt-oss-120b` needed a 350-token output window, because with 150 it spent the whole budget on hidden reasoning before writing a citation.

## 4. Where the time goes

Request `stea_0002`, a slow live call from `results.csv` and `pipeline_trace_log.jsonl`, took 8,255.6 ms in total:

- **Before first token, 5,283.6 ms (64%):** retrieval was 66.3 ms. The other 5,217.3 ms happened inside the provider (prefill, queueing, inference), which I cannot separate.
- **After first token, 2,972.0 ms (36%):** 127 output tokens at about 43 tok/s.

Retrieval was 66.3 ms of 8,255.6 ms (0.8%), so nothing on my side was slow; 63% of the total was the provider before the first token. This is one request, so all it shows is that a long prompt to a reasoning model is slow before the first token.

## 5. When a limit is hit, and retries

- **Per-minute limit:** the stand-in sheds excess requests in 11–16 ms with a 429 and `retry-after`. Gate refusals still return in about 12 ms at no token cost. Under load, 70–80% of requests were rejected.
- **Per-day limit:** no model call is made. In that test 12 of 15 requests got a quota-exhausted notice and the 3 trap questions still got their correct gate refusal.
- **Retry policy:** I shed rejected requests instead of queueing them. A 1,750-token request takes about 13 s to refill at 133 tok/s, so a queue under overload would make later requests time out. The only retries in `results.csv` are citation self-corrections.
- **Where retrying hurts:** on the live tier a retry costs another ~1,800 tokens of the minute's budget, which pushes the next request into a 429.

## 6. Retrieval: quality versus cost

I scored 15 hand-written questions against the stand-in: 12 answerable and 3 traps that should be refused.

### Table 3: Retrieval settings

| Setting | Correct and cited (of 12) | Incorrect (of 12) | Correct refusals (of 3) | Tok/ans | First token p50 / p95 | Total p50 / p95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| BM25, top-5 | 91.7% | 8.3% | 100% | 31.6 | 693.8 / 783.1 | 1,397.2 / 1,798.1 |
| Dense BGE-small, top-5 (chosen) | 91.7% | 8.3% | 100% | 33.7 | 768.1 / 849.6 | 1,561.0 / 1,864.0 |
| Hybrid BM25 + dense, RRF k=60, top-5 | 91.7% | 8.3% | 100% | 36.3 | 790.0 / 903.0 | 1,728.6 / 1,874.8 |
| Dense BGE-small, top-2 | 100% | 0% | 100% | 34.6 | 795.7 / 856.1 | 1,609.8 / 1,844.9 |

All settings use 350-token chunks and no re-ranker. I picked top-5 for recall on questions that span several sections. The numbers favor top-2 for short factual questions: it answered all 12 correctly with 56% fewer prompt tokens. With 12 questions the difference is one answer, so I treat top-2 as the fallback when tokens run short, and not yet as the default.

**Three ways this library misleads a naive retriever:**

- **Lexical hijacking (q14):** "What is Moshi's BLEU on WMT 2014?" retrieves the Transformer paper's 37.1 BLEU, which a naive model would credit to Moshi.
- **Polysemy (q07):** "How does chunking optimize memory?" mixes Moshi's fixed audio-frame chunking with Dynamic Chunking's learned boundaries.
- **Out-of-domain drift (q13):** "What batch size trained LLaMA-3?" returns generic pre-training paragraphs at similarity 0.77, and a naive system would invent a number.

## 7. What the person sees

When the daily budget runs out, the system currently says: *"Daily request budget exhausted. Quota resets at 00:00 UTC."* I would change it to: *"I've reached today's compute quota and can't write a synthesis right now. It resets at 00:00 UTC. Meanwhile, here are the top passages I found: [Attention Is All You Need, 3.2.2, p. 5] …"* The passages are already retrieved, so showing them costs no model tokens. I have not built this yet.

## 8. What I'm unsure about, left out, and learned

- **Least certain:** whether the stand-in captures how the live provider behaves. It returns canned text, so it never reproduces the reasoning-token use or the multi-second prefill I saw live.
- **Left out on purpose:** a vector database (191 chunks fit in a 290 KB numpy array) and a cross-encoder re-ranker (about 150 ms extra latency per query, with negligible answer-level gain).
- **Learned:** on a free tier, tokens per minute sets RAG throughput, not requests per minute. One request uses about a quarter of the 8,000-token allowance (~2,029 tokens), so cutting prompt tokens (k = 5 to 2, which saved 56%) is the largest lever I measured.

## Feedback on Vaani

- **What stood out:** responses felt immediate. The voice sounded natural, coped with interruptions and pauses mid-sentence, and understood Indian accents and Hinglish without me repeating myself.
- **What confused me:** it was not clear whether the agent can take real-time actions during a call (a booking, a CRM update) or only collect information for after the call. I also could not tell which integrations work out of the box and which need a custom webhook.
- **One change:** put a browser playground on the homepage where I can try my own system prompt and awkward audio (background noise, interruptions, going off-script), with native integrations and pricing documented next to it.

## How I worked with the assistant

The full, unedited conversation is in `chat_history.md`. Several turns are pasted from a second assistant I used as an outside reviewer of the code and outputs.

- **Three prompts that mattered:** (1) *"don't execute any cmd by yourself, first tell me and give the cmd"*, which made me review every script before it ran. (2) Pasting my first PDF extraction audit (122 records for 120 pages), which moved me from page splits to a line-level heading parser (632 paragraphs, 94 sections). (3) Pasting the outside reviewer's checkpoint, which led to excluding gate refusals from TTFT, building an independent check of the stand-in, and reconciling the trace logs.
- **An error I caught:** recomputing the report tables from raw `results.csv` showed that Table 1 had per-day limits labelled as per-minute, costs were about 300× too low, percentiles did not match the file, and retried 429s were logged as `status: ok`.
- **A decision I overruled:** the assistant first claimed dense retrieval beat BM25 and hybrid by 8.3 points. Answer-level scores tie at 11 of 12 in Table 3, so I dropped the claim.
- **What is mine:** reviewing every command before it ran, checking report numbers against raw CSV rows with a separate recomputation script, and designing the pre-generation gate to protect quota.