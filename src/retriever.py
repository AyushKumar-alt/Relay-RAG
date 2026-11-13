import math
import re
from collections import Counter
from typing import List, Dict, Any

class BM25Retriever:
    """
    Standard Okapi BM25 implementation for passage retrieval over chunks.jsonl.
    Zero external dependencies, fast and deterministic.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = Counter()
        self.idf: Dict[str, float] = {}
        self.tokenized_corpus: List[List[str]] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        # Lowercase, alphanumeric tokenization
        tokens = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())
        return tokens

    def fit(self, chunks: List[Dict[str, Any]]):
        self.corpus = chunks
        self.doc_lengths = []
        self.doc_freqs = Counter()
        self.tokenized_corpus = []

        total_tokens = 0
        for chunk in chunks:
            tokens = self.tokenize(chunk["text"] + " " + chunk.get("section_title", ""))
            self.tokenized_corpus.append(tokens)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_tokens += doc_len

            # Update document frequencies (unique terms per doc)
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] += 1

        n_docs = len(chunks)
        self.avg_doc_len = total_tokens / n_docs if n_docs > 0 else 0.0

        # Calculate Robertson-Spärck Jones IDF
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # BM25 IDF with smoothing
            self.idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5, paper_filter: str = None) -> List[Dict[str, Any]]:
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for idx, (doc_tokens, doc_len) in enumerate(zip(self.tokenized_corpus, self.doc_lengths)):
            chunk = self.corpus[idx]
            if paper_filter and chunk["paper_id"] != paper_filter:
                continue

            # Term frequencies in this document
            tf = Counter(doc_tokens)
            score = 0.0

            for q_term in query_tokens:
                if q_term not in tf:
                    continue
                term_tf = tf[q_term]
                idf_val = self.idf.get(q_term, 0.0)

                # Standard BM25 formula
                numerator = term_tf * (self.k1 + 1)
                denominator = term_tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf_val * (numerator / denominator)

            if score > 0:
                scores.append((score, chunk))

        # Sort descending by score
        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for rank, (score, chunk) in enumerate(scores[:top_k], start=1):
            res_item = dict(chunk)
            res_item["bm25_score"] = round(score, 4)
            res_item["rank"] = rank
            results.append(res_item)

        return results

