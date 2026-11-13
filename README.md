# The Relay: Resilient RAG Research Pipeline

This repository contains the complete implementation, load experiments, and 2-page write-up for **The Relay** assignment.

---

##  Repository & Submission Structure

```text
├── REPORT.md                         # Complete 2-page write-up with all 4 required tables
├── results.csv                       # Raw request-level log (135 requests across 6 scenarios)
├── requirements.txt                  # Python dependencies
├── .env.example                      # Template for API keys (never commit real keys)
│
├── src/                              # Source code
│   ├── pipeline.py                   # 4-step Relay pipeline (Retrieval -> Gate -> Model -> Guardrail)
│   ├── standin_service.py            # Calibrated stand-in model simulator
│   ├── retriever.py                  # Pure-Python BM25 lexical retriever
│   ├── dense_indexer.py              # BGE-Small vector embedding precomputer
│   ├── ingest_papers.py              # Page-level PDF extractor with section tracking
│   ├── eval_retrieval_comparison.py  # BM25 vs. Dense vs. Hybrid (RRF) benchmark
│   ├── inspect_and_freeze_retrieval.py # Question-level audit & metric verification
│   ├── freeze_manifest.py            # SHA-256 cryptographic manifest generator
│   ├── test_standin_service.py       # Stand-in verification test suite
│   └── run_load_experiments.py       # 6-scenario load testing engine
│
├── data/                             # Verified data artifacts & contracts
│   ├── paragraphs.jsonl              # 632 canonical paragraphs from the 3 papers
│   ├── chunks.jsonl                  # 191 retrieval-ready chunks with full provenance
│   ├── benchmark_questions.json      # 15 gold evaluation queries (12 factual, 3 traps)
│   ├── dense_embeddings_bge.npy      # Precomputed normalized BGE embeddings
│   ├── corpus_v1_manifest.json       # SHA-256 checksums of source papers & corpus
│   ├── retrieval_comparison_v1_manifest.json # Full question-level retrieval audit
│   ├── standin_verification_trace.json # Trace evidence of stand-in simulator behavior
│   └── pipeline_trace_log.jsonl      # Step-by-step trace log of all 135 load requests
│
└── 3 Papers/                         # Immutable source research papers
    ├── Attention is all you need.pdf (arXiv:1706.03762v7)
    ├── Dynamic Chunking.pdf          (arXiv:2507.07955v2)
    └── Moshi.pdf                     (arXiv:2410.00037v1)
```

---

##  Quickstart & Reproduction

### 1. Environment Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Verify Stand-in Service
```powershell
python src/test_standin_service.py
```

### 3. Re-run All 6 Load Scenarios (Regenerates `results.csv`)
```powershell
python src/run_load_experiments.py
```

