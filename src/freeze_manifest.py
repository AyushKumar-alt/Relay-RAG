import hashlib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "3 Papers"
DATA_DIR = BASE_DIR / "data"

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("=" * 65)
    print("      FREEZING CORPUS V1 & COMPUTING SHA-256 CHECKSUMS")
    print("=" * 65)

    files_to_hash = [
        PAPERS_DIR / "Attention is all you need.pdf",
        PAPERS_DIR / "Dynamic Chunking.pdf",
        PAPERS_DIR / "Moshi.pdf",
        DATA_DIR / "paragraphs.jsonl",
        DATA_DIR / "chunks.jsonl",
        DATA_DIR / "benchmark_questions.json"
    ]

    checksums = {}
    for p in files_to_hash:
        if p.exists():
            h = compute_sha256(p)
            checksums[p.name] = h
            print(f"  {p.name:<35} | {h[:16]}...{h[-8:]}")
        else:
            print(f"  [MISSING] {p.name}")

    manifest = {
        "corpus_version": "corpus_v1",
        "status": "frozen",
        "frozen_timestamp": "2026-11-10T11:07:00Z",
        "papers": 3,
        "pdf_pages": 120,
        "sections": 94,
        "paragraphs": 632,
        "chunks": 191,
        "manual_quality_audit": "passed_with_minor_issues",
        "known_issues": [
            "occasional_hyphenation_artifacts",
            "occasional_accented_character_corruption",
            "moshi_chunk_00188_contains_noisy_appendix_content"
        ],
        "bm25_baseline": {
            "recall_at_5": 0.917,
            "recall_at_10": 1.0,
            "mrr": 0.884,
            "median_latency_ms": 10.31,
            "p95_latency_ms": 13.12
        },
        "sha256_checksums": checksums
    }

    manifest_path = DATA_DIR / "corpus_v1_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 65)
    print(f"Manifest successfully frozen to: {manifest_path.name}")
    print("=" * 65)

if __name__ == "__main__":
    main()

