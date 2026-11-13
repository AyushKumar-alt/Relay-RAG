import json
import time
from pathlib import Path
import numpy as np
from fastembed import TextEmbedding

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
EMBEDDINGS_OUTPUT_PATH = DATA_DIR / "dense_embeddings_bge.npy"
MANIFEST_OUTPUT_PATH = DATA_DIR / "dense_index_manifest.json"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

def main():
    print("=" * 65)
    print(f"      PRECOMPUTING DENSE EMBEDDINGS: {MODEL_NAME}")
    print("=" * 65)

    # 1. Load chunks
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH.name}.")

    # Format texts for embedding (incorporate section context)
    docs = [f"{c.get('section_title', '')}: {c['text']}" for c in chunks]

    # 2. Initialize embedding model
    print(f"\nInitializing {MODEL_NAME} via fastembed (ONNX runtime)...")
    t_model_start = time.perf_counter()
    embedding_model = TextEmbedding(model_name=MODEL_NAME)
    init_ms = (time.perf_counter() - t_model_start) * 1000
    print(f"Model initialized in {init_ms:.1f} ms.")

    # 3. Compute embeddings
    print(f"Embedding {len(docs)} documents in batch...")
    t_embed_start = time.perf_counter()
    # fastembed generator produces numpy arrays
    embeddings_list = list(embedding_model.embed(docs))
    embeddings_matrix = np.array(embeddings_list)
    embed_ms = (time.perf_counter() - t_embed_start) * 1000

    # L2 normalize for cosine similarity via dot product
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized_matrix = embeddings_matrix / norms

    print(f"Embedding completed in {embed_ms:.1f} ms ({embed_ms/len(docs):.2f} ms/chunk).")
    print(f"Matrix shape: {normalized_matrix.shape} (dim={normalized_matrix.shape[1]})")

    # 4. Save embeddings
    np.save(EMBEDDINGS_OUTPUT_PATH, normalized_matrix)
    print(f"Saved normalized vectors to: {EMBEDDINGS_OUTPUT_PATH.name}")

    manifest = {
        "model_name": MODEL_NAME,
        "embedding_dim": normalized_matrix.shape[1],
        "chunks_indexed": len(chunks),
        "init_time_ms": round(init_ms, 2),
        "index_build_time_ms": round(embed_ms, 2),
        "output_file": EMBEDDINGS_OUTPUT_PATH.name
    }
    with open(MANIFEST_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved indexing manifest to: {MANIFEST_OUTPUT_PATH.name}")
    print("=" * 65)

if __name__ == "__main__":
    main()

