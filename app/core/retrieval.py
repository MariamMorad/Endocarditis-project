"""
Hybrid retrieval logic (BM25 + Vector) with result fusion and real confidence
scoring, followed by cross-encoder reranking. Matches notebook Cell 14.
"""
import math
import re

from app.core.indexing import RAGIndex

ABBREVIATIONS = {
    "IE": "infective endocarditis",
    "ESC": "European Society of Cardiology",
    "NICE": "National Institute for Health and Care Excellence",
    "TOE": "transoesophageal echocardiography",
    "TTE": "transthoracic echocardiography",
    "GDG": "Guideline Development Group",
}


def expand_for_bm25(query: str) -> str:
    expanded = query
    for abbr, full in ABBREVIATIONS.items():
        expanded = re.sub(rf'\b{abbr}\b', f'{abbr} {full}', expanded)
    return expanded


def detect_source_filter(query: str):
    q = query.lower()
    if "esc" in q or "2023" in q:
        return {"source": "ESC.pdf"}
    if "nice" in q:
        return {"source": "NICE.pdf"}
    return None


def get_significant_words(text: str, min_len: int = 4) -> set[str]:
    return set(w.lower().strip('.,;:()') for w in text.split() if len(w) >= min_len)


def compute_confidence(query, chunk_text, bm25_rank, vector_similarity,
                        in_bm25, in_vector, bm25_total=20):
    bm25_score = (bm25_total - bm25_rank) / bm25_total if in_bm25 else 0.0
    vector_score = vector_similarity if in_vector else 0.0

    agreement_bonus = 0.15 if (in_bm25 and in_vector) else 0.0

    query_words = get_significant_words(query)
    chunk_words = get_significant_words(chunk_text)
    coverage = len(query_words & chunk_words) / max(len(query_words), 1)

    raw_score = (0.40 * bm25_score) + (0.30 * vector_score) + agreement_bonus + (0.15 * coverage)
    return round(min(raw_score, 1.0), 3)


class RetrievalEngine:
    """Wraps a RAGIndex and exposes retrieve_fused / retrieve_with_rerank."""

    def __init__(self, index: RAGIndex):
        self.index = index

    def retrieve_fused(self, query: str, k: int = 14, pool_k: int = 45,
                        use_source_filter: bool = True, use_bm25_expansion: bool = True):
        bm25_query = expand_for_bm25(query) if use_bm25_expansion else query
        bm25_results = self.index.bm25_retriever.invoke(bm25_query)

        filter_ = detect_source_filter(query) if use_source_filter else None
        vector_results_scored = self.index.get_similarity_scores(query, k=pool_k, filter_=filter_)

        bm25_ids = {doc.metadata["chunk_id"]: i for i, doc in enumerate(bm25_results)}
        vector_scores = {doc.metadata["chunk_id"]: score for doc, score in vector_results_scored}

        all_docs = {doc.metadata["chunk_id"]: doc for doc in bm25_results}
        all_docs.update({doc.metadata["chunk_id"]: doc for doc, _ in vector_results_scored})

        scored = []
        for chunk_id, doc in all_docs.items():
            in_bm25 = chunk_id in bm25_ids
            in_vector = chunk_id in vector_scores
            conf = compute_confidence(
                query, doc.page_content,
                bm25_rank=bm25_ids.get(chunk_id, 45),
                vector_similarity=vector_scores.get(chunk_id, 0.0),
                in_bm25=in_bm25, in_vector=in_vector,
                bm25_total=45,
            )
            scored.append({"doc": doc, "confidence": conf})

        scored.sort(key=lambda x: (x["confidence"], x["doc"].metadata.get("chunk_id", "")), reverse=True)
        return scored[:pool_k]

    def retrieve_with_rerank(self, query: str, k: int = 14, pool_k: int = 45,
                              use_source_filter: bool = True, use_bm25_expansion: bool = True):
        candidates = self.retrieve_fused(
            query, k=pool_k, pool_k=pool_k,
            use_source_filter=use_source_filter,
            use_bm25_expansion=use_bm25_expansion,
        )
        if not candidates:
            return []

        pairs = [[query, c["doc"].page_content] for c in candidates]
        rerank_scores = self.index.reranker.predict(pairs)

        for c, s in zip(candidates, rerank_scores):
            s = float(s)
            c["rerank_score"] = round(1.0 / (1.0 + math.exp(-s)), 4)

        candidates.sort(key=lambda x: (x["rerank_score"], x["doc"].metadata.get("chunk_id", "")), reverse=True)
        return candidates[:k]
