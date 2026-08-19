"""
API-level request/response pydantic models.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The clinical question to ask")
    k: int = Field(8, ge=1, le=20, description="Number of chunks to use for generation")


class CitationOut(BaseModel):
    document: str
    section: str
    page: int
    chunk_id: str
    retrieval_score: float
    excerpt: str


class EvidenceItemOut(BaseModel):
    claim: str
    chunk_id: str


class AskResponse(BaseModel):
    query: str
    refused: bool
    reason: Optional[str] = None
    message: Optional[str] = None
    evidence_found_nearby: Optional[List[str]] = None
    evidence_gap: Optional[str] = None
    recommendation: Optional[str] = None
    supporting_evidence: Optional[List[EvidenceItemOut]] = None
    citations: Optional[List[CitationOut]] = None
    confidence: str
    safety_disclaimer: Optional[str] = None
    verification_log: Optional[List[str]] = None


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    total_chunks: int
