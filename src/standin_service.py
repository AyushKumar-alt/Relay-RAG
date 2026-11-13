import time
import random
import threading
from typing import Generator, Dict, Any, Optional

class CalibratedStandinService:
    """
    Calibrated stand-in model service simulating Groq's API behavior.
    Calibrated against empirical measurements of openai/gpt-oss-120b on Groq:
      - Baseline TTFT: log-normal distribution centered at ~690ms (min 640ms, p95 ~780ms)
      - Token generation rate: ~40 tokens/second (25ms per token)
      - Token bucket: 8,000 TPM limit (reset ~1.125s) and 1,000 RPM limit
      - Concurrency penalty: TTFT scales with in-flight queue depth
      - Authentic 429 HTTP status and error JSON payloads
      - Daily budget exhaustion switch (simulates 3:00 PM cutoff)
    """

    def __init__(
        self,
        rpm_limit: int = 1000,
        tpm_limit: int = 8000,
        base_ttft_p50_ms: float = 690.0,
        token_rate_tok_per_sec: float = 40.0,
        concurrency_penalty_alpha: float = 0.15
    ):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.base_ttft_p50_ms = base_ttft_p50_ms
        self.token_rate = token_rate_tok_per_sec
        self.concurrency_alpha = concurrency_penalty_alpha

        # Lock for thread safety
        self.lock = threading.Lock()

        # In-flight concurrency counter
        self.in_flight = 0
        self.peak_in_flight = 0

        # Token bucket state
        self.current_minute_tokens = 0
        self.current_minute_requests = 0
        self.minute_window_start = time.perf_counter()

        # Daily budget state
        self.daily_budget_exhausted = False
        self.total_requests_served = 0
        self.total_tokens_served = 0

    def set_daily_budget_exhausted(self, exhausted: bool = True):
        with self.lock:
            self.daily_budget_exhausted = exhausted

    def set_rpm_limit(self, new_rpm: int):
        with self.lock:
            self.rpm_limit = new_rpm

    def _update_buckets(self, estimated_tokens: int) -> Optional[Dict[str, Any]]:
        """
        Enforces token and request rate limits.
        Returns None if OK, or a 429 error dict if rate limited.
        """
        now = time.perf_counter()
        elapsed = now - self.minute_window_start

        # Sliding 60-second window reset
        if elapsed >= 60.0:
            self.current_minute_tokens = 0
            self.current_minute_requests = 0
            self.minute_window_start = now
            elapsed = 0.0

        # Check daily budget exhaustion (the 3pm cutoff)
        if self.daily_budget_exhausted:
            return {
                "status_code": 429,
                "error": {
                    "message": "Daily request budget exhausted. Quota resets at 00:00 UTC.",
                    "type": "daily_quota_exceeded",
                    "code": "daily_rate_limit_exceeded"
                },
                "headers": {
                    "retry-after": "32400",  # ~9 hours until midnight UTC
                    "x-ratelimit-remaining-requests": "0",
                    "x-ratelimit-remaining-tokens": "0"
                }
            }

        # Check RPM limit
        if self.current_minute_requests >= self.rpm_limit:
            wait_sec = max(1.0, round(60.0 - elapsed, 1))
            return {
                "status_code": 429,
                "error": {
                    "message": f"Rate limit reached for requests per minute. Limit: {self.rpm_limit}.",
                    "type": "requests",
                    "code": "rate_limit_exceeded"
                },
                "headers": {
                    "retry-after": str(int(wait_sec)),
                    "x-ratelimit-remaining-requests": "0",
                    "x-ratelimit-limit-requests": str(self.rpm_limit)
                }
            }

        # Check TPM limit
        if self.current_minute_tokens + estimated_tokens > self.tpm_limit:
            wait_sec = max(1.0, round(60.0 - elapsed, 1))
            return {
                "status_code": 429,
                "error": {
                    "message": f"Rate limit reached for tokens per minute. Limit: {self.tpm_limit}.",
                    "type": "tokens",
                    "code": "rate_limit_exceeded"
                },
                "headers": {
                    "retry-after": str(int(wait_sec)),
                    "x-ratelimit-remaining-tokens": "0",
                    "x-ratelimit-limit-tokens": str(self.tpm_limit)
                }
            }

        # Consume quota
        self.current_minute_requests += 1
        self.current_minute_tokens += estimated_tokens
        self.total_requests_served += 1
        self.total_tokens_served += estimated_tokens
        return None

    def stream_chat_completion(
        self,
        messages: list,
        max_tokens: int = 150,
        grounded_context: str = ""
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Simulates streaming tokens with realistic TTFT, inter-token intervals,
        and concurrency degradation.
        """
        prompt_text = " ".join(m.get("content", "") for m in messages)
        tokens_in = max(10, int(len(prompt_text.split()) * 1.3))
        est_tokens_out = min(max_tokens, 80)

        # 1. Check rate limits
        with self.lock:
            rate_err = self._update_buckets(tokens_in + est_tokens_out)
            if rate_err:
                yield {"type": "error", "payload": rate_err}
                return

            self.in_flight += 1
            if self.in_flight > self.peak_in_flight:
                self.peak_in_flight = self.in_flight
            active_concurrency = self.in_flight

        try:
            # 2. Simulate TTFT with concurrency inflation
            # Base log-normal jitter around empirical p50
            jitter = random.lognormvariate(0.0, 0.08)
            inflated_ttft_ms = (self.base_ttft_p50_ms * jitter) * (1.0 + self.concurrency_alpha * max(0, active_concurrency - 1))
            
            # Simulated wait time before first token
            time.sleep(inflated_ttft_ms / 1000.0)

            # Signal first token
            yield {
                "type": "first_token",
                "ttft_ms": inflated_ttft_ms,
                "in_flight": active_concurrency
            }

            # 3. Generate answer tokens based on grounded context or question
            user_msg = messages[-1]["content"] if messages else ""
            simulated_text = self._synthesize_simulated_reply(user_msg, grounded_context)
            words = simulated_text.split()

            # Stream words with inter-token delay (~25ms per token)
            inter_token_delay = 1.0 / self.token_rate
            tokens_generated = 0

            for w in words:
                yield {
                    "type": "delta",
                    "delta": w + " "
                }
                tokens_generated += 1
                time.sleep(inter_token_delay)

            # Final finish event
            yield {
                "type": "finish",
                "tokens_in": tokens_in,
                "tokens_out": tokens_generated,
                "total_text": simulated_text
            }

        finally:
            with self.lock:
                self.in_flight = max(0, self.in_flight - 1)

    def _synthesize_simulated_reply(self, query: str, context: str) -> str:
        """
        Produces realistic grounded cited responses or honest refusals
        matching paper contents.
        """
        q_low = query.lower()

        # Check refusal traps
        if "llama-3" in q_low or "llama 3" in q_low:
            return "The provided papers on Attention, Moshi, and Dynamic Chunking do not contain any information about LLaMA-3. Refusing to speculate."
        if "moshi" in q_low and "bleu" in q_low:
            return "The Moshi paper does not evaluate the model on machine translation or report a BLEU score on WMT 2014. Moshi is a speech-text foundation model for real-time spoken dialogue."
        if "sql" in q_low or "database" in q_low:
            return "The Dynamic Chunking paper focuses entirely on sequence modeling for hierarchical neural networks, and does not discuss relational databases or SQL indexing."

        # Factual queries with citations
        if "dimension" in q_low or "dk" in q_low:
            return "In the Transformer architecture, the multi-head attention mechanism uses h = 8 parallel attention layers with dimensions dk = dv = dmodel / h = 64, where dmodel = 512 [Attention Is All You Need, 3.2.2 Multi-Head Attention, p. 5]."
        if "additive" in q_low and "dot-product" in q_low:
            return "The authors suspect that for large values of dk, dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients, which is why scaling by 1/sqrt(dk) is required [Attention Is All You Need, 3.2.1 Scaled Dot-Product Attention, p. 4]."
        if "bleu" in q_low and ("german" in q_low or "transformer" in q_low):
            return "On machine translation, the Transformer established a 28.4 BLEU score on the WMT 2014 English-to-German task and 41.8 BLEU on English-to-French [Attention Is All You Need, 6.1 Machine Translation, p. 8]."
        if "positional" in q_low:
            return "Positional encodings in the Transformer are calculated using sine and cosine functions of different frequencies: PE(pos, 2i) = sin(pos / 10000^(2i/dmodel)) and PE(pos, 2i+1) = cos(pos / 10000^(2i/dmodel)) [Attention Is All You Need, 3.5 Positional Encoding, p. 6]."
        if "dynamic chunking" in q_low and ("boundary" in q_low or "segment" in q_low):
            return "Dynamic Chunking dynamically determines boundary segmentations via a learned boundary predictor rather than relying on static fixed-size windows [Dynamic Chunking, 1 Introduction, p. 2]."
        if "mimi" in q_low:
            return "Mimi is a neural audio codec operating at a frame rate of 12.5 Hz on 24 kHz audio, using residual vector quantization [Moshi, 3.1 Neural Audio Codec, p. 6]."
        if "latency" in q_low and "moshi" in q_low:
            return "Moshi achieves an observed conversational latency of 160 ms with a theoretical latency of ~200 ms end-to-end [Moshi, 1 Introduction, p. 2]."

        # Default grounded summary from context
        return f"Based on the library papers, the relevant passages state: {context[:180]}... [Cited from retrieved sections]."

