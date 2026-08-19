"""
Never trust the LLM's own citations blindly. For every citation we check:
  1. chunk_id actually exists in the retrieved set (no hallucinated chunk_ids)
  2. document/section metadata matches the real chunk metadata
  3. the claimed excerpt genuinely appears in the chunk text (fuzzy match)
Invalid citations/claims are stripped, and confidence is downgraded if anything was removed.
"""
from difflib import SequenceMatcher

from app.core.llm_schemas import GroundedAnswer


def verify_and_repair_answer(answer: GroundedAnswer, relevant_chunks: list, excerpt_match_threshold: float = 0.5):
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
        match_ratio = SequenceMatcher(None, c.excerpt.lower(), chunk.page_content.lower()).ratio()
        excerpt_present = c.excerpt.lower()[:80] in chunk.page_content.lower() or match_ratio >= excerpt_match_threshold
        if not excerpt_present:
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
        idx = confidence_order.index(repaired.confidence)
        repaired.confidence = confidence_order[max(0, idx - 1)]
        repaired.safety_disclaimer += " (Confidence downgraded: one or more citations failed verification.)"

    return repaired, log
