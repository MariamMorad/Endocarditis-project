"""
Pydantic schemas that Gemini returns as structured output.
"""
from typing import List, Literal
from pydantic import BaseModel, Field


class RetrievalEvaluation(BaseModel):
    """Decision on scope and evidence sufficiency."""
    in_scope: bool = Field(
        description="True only if the question is about infective endocarditis (risk, prevention, "
                    "diagnosis, management) per NICE/ESC/AHA guidance."
    )
    sufficient_evidence: bool = Field(
        description="True only if retrieved chunks specifically support answering this question."
    )
    relevant_chunk_ids: List[str] = Field(
        description="chunk_ids that are actually usable evidence for this question."
    )
    evidence_gap: str = Field(description="What is missing, if anything. Empty string if evidence is sufficient.")
    reasoning: str = Field(description="Brief internal justification.")


class Citation(BaseModel):
    document: str
    section: str
    page: int
    chunk_id: str
    retrieval_score: float
    excerpt: str = Field(description="Short exact excerpt (<40 words) copied from the chunk that supports the claim.")


class EvidenceItem(BaseModel):
    claim: str
    chunk_id: str = Field(description="chunk_id this specific claim is grounded in.")


class GroundedAnswer(BaseModel):
    """Recommendation / Supporting Evidence / Citations / Confidence & Safety."""
    recommendation: str = Field(description="Short, direct answer. No patient-specific treatment.")
    supporting_evidence: List[EvidenceItem]
    citations: List[Citation]
    confidence: Literal["High", "Medium", "Low", "Insufficient Evidence"]
    safety_disclaimer: str = Field(description="Notes this supports clinicians and does not replace medical judgment.")
