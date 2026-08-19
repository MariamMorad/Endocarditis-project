"""
Never trust the LLM's own citations blindly. For every citation we check:
  1. chunk_id actually exists in the retrieved set (no hallucinated chunk_ids)
  2. document/section metadata matches the real chunk metadata
  3. the claimed excerpt genuinely appears in the chunk text (substring / longest match coverage)
Invalid citations/claims are stripped, and confidence is downgraded if anything was removed.
"""
from difflib import SequenceMatcher
from app.core.llm_schemas import GroundedAnswer


def excerpt_supported(excerpt: str, chunk_text: str, min_coverage: float = 0.7) -> bool:
    """
    Measures what fraction of the excerpt is covered by the longest contiguous
    span it shares with the chunk text. Robust to whitespace differences and
    minor tokenization artifacts.
    """
    excerpt_norm = " ".join(excerpt.lower().split())
    chunk_norm = " ".join(chunk_text.lower().split())
    if not excerpt_norm:
        return False
    if excerpt_norm in chunk_norm:
        return True
    match = SequenceMatcher(None, excerpt_norm, chunk_norm).find_longest_match(
        0, len(excerpt_norm), 0, len(chunk_norm)
    )
    coverage = match.size / len(excerpt_norm)
    return coverage >= min_coverage


def verify_and_repair_answer(answer: GroundedAnswer, relevant_chunks: list, excerpt_match_threshold: float = 0.7):
    chunk_lookup = {r["doc"].metadata["chunk_id"]: r["doc"] for r in relevant_chunks}

    log = []
    valid_citations = []
    for c in answer.citations:
        chunk = chunk_lookup.get(c.chunk_id)
        if chunk is None:
            log.append(f"DROPPED citation {c.chunk_id}: chunk_id not in retrieved set (hallucinated).")
            continue
        meta = chunk.metadata
        if meta["source"] != c.document or meta["section"] != c.section:
            log.append(f"DROPPED citation {c.chunk_id}: metadata mismatch vs actual chunk.")
            continue
        
        if not excerpt_supported(c.excerpt, chunk.page_content, min_coverage=excerpt_match_threshold):
            log.append(f"DROPPED citation {c.chunk_id}: excerpt not found in chunk text (possible fabrication).")
            continue
        valid_citations.append(c)

    valid_ids = {c.chunk_id for c in valid_citations}
    valid_evidence = [e for e in answer.supporting_evidence if e.chunk_id in valid_ids]
    if len(valid_evidence) < len(answer.supporting_evidence):
        log.append(f"DROPPED {len(answer.supporting_evidence) - len(valid_evidence)} supporting_evidence item(s) with no valid citation.")

    repaired = answer.model_copy(update={"citations": valid_citations, "supporting_evidence": valid_evidence})

    if log:
        confidence_order = ["Insufficient Evidence", "Low", "Medium", "High"]
        if repaired.confidence in confidence_order:
            idx = confidence_order.index(repaired.confidence)
            repaired.confidence = confidence_order[max(0, idx - 1)]
        repaired.safety_disclaimer += " (Confidence downgraded: one or more citations failed verification.)"

    return repaired, log
