"""
Full pipeline orchestration: retrieval -> agentic evaluation -> generation -> citation
verification. Same flow as clinical_rag_answer in the notebook, built as a class so it
can be injected via Depends in FastAPI.
"""
from app.core.indexing import RAGIndex
from app.core.retrieval import RetrievalEngine
from app.core.evaluator import evaluate_retrieval
from app.core.generator import generate_grounded_answer
from app.core.verifier import verify_and_repair_answer
from app.core.refusal import build_refusal


class ClinicalRAGPipeline:
    def __init__(self, index: RAGIndex):
        self.index = index
        self.retriever = RetrievalEngine(index)

    def answer(self, query: str, k: int = 14) -> dict:
        # 1. Retrieval (hybrid fusion + cross-encoder rerank)
        retrieved = self.retriever.retrieve_with_rerank(query, k=k)

        if not retrieved:
            fake_eval = type("E", (), {"evidence_gap": "No chunks retrieved at all."})()
            return build_refusal(query, fake_eval, [], "insufficient_evidence")

        # 2. Agentic evaluation - scope + sufficiency
        evaluation = evaluate_retrieval(query, retrieved)

        if not evaluation.in_scope:
            return build_refusal(query, evaluation, retrieved, "out_of_scope")
        if not evaluation.sufficient_evidence or not evaluation.relevant_chunk_ids:
            return build_refusal(query, evaluation, retrieved, "insufficient_evidence")

        # 3. Restrict generation to evaluator-approved chunks only
        relevant = [r for r in retrieved if r["doc"].metadata["chunk_id"] in evaluation.relevant_chunk_ids]

        try:
            answer = generate_grounded_answer(query, relevant)
        except Exception:
            return build_refusal(query, evaluation, retrieved, "insufficient_evidence")

        # 4. Verify and repair citations
        answer, verification_log = verify_and_repair_answer(answer, relevant)

        if not answer.citations:
            return build_refusal(query, evaluation, retrieved, "citation_verification_failed")

        return {
            "query": query,
            "refused": False,
            "recommendation": answer.recommendation,
            "supporting_evidence": [e.model_dump() for e in answer.supporting_evidence],
            "citations": [c.model_dump() for c in answer.citations],
            "confidence": answer.confidence,
            "safety_disclaimer": answer.safety_disclaimer,
            "verification_log": verification_log,
        }
