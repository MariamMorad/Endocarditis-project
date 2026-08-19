"""
Build a structured "refusal" response when the question is out-of-scope, evidence
is insufficient, or citation verification failed.
"""
from app.core.llm_schemas import RetrievalEvaluation

REFUSAL_MESSAGES = {
    "out_of_scope": (
        "This assistant only answers questions about infective endocarditis (risk, prevention, "
        "diagnosis, and management) based on the NICE, ESC, and AHA guidelines it was built on. "
        "This question falls outside that scope."
    ),
    "citation_verification_failed": (
        "The retrieved guidelines do not provide evidence that could be reliably traced and cited "
        "for this question. Please consult the relevant clinical guideline or a qualified clinician."
    ),
    "insufficient_evidence": (
        "The retrieved guidelines do not provide sufficient evidence to answer this question "
        "reliably. Please consult the relevant clinical guideline or a qualified clinician."
    ),
}


def build_refusal(query: str, evaluation: RetrievalEvaluation, retrieved: list, reason: str) -> dict:
    message = REFUSAL_MESSAGES.get(reason, REFUSAL_MESSAGES["insufficient_evidence"])
    found = [r["doc"].metadata["section"] for r in retrieved[:3]]
    return {
        "query": query,
        "refused": True,
        "reason": reason,
        "message": message,
        "evidence_found_nearby": found,
        "evidence_gap": evaluation.evidence_gap,
        "confidence": "Insufficient Evidence",
    }
