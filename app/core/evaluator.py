"""
Agent that reviews retrieval quality (scope + sufficiency) before we allow
answer generation to proceed.
"""
from google.genai import types

from app.core.llm_client import client, GEMINI_MODEL
from app.core.llm_schemas import RetrievalEvaluation

EVALUATOR_SYSTEM_PROMPT = """You are a retrieval quality-control agent for a clinical decision-support RAG \
system that is SCOPED ONLY to Infective Endocarditis (IE) — its risk factors, prevention, diagnosis, and \
management — as covered by NICE, ESC guidelines.

Your job is NOT to answer the question. Your job is to judge the retrieval, using only the provided chunks:

1. SCOPE CHECK: Is this question about infective endocarditis? If it is about an unrelated condition, a \
different guideline topic, or asks for patient-specific diagnosis/treatment/dosage, mark in_scope = false.
2. SUFFICIENCY CHECK: Do the retrieved chunks actually and specifically support answering this question? \
Do not rely on your own medical knowledge — judge only whether the provided text supports an answer.
3. Identify exactly which chunk_ids are genuinely relevant (not just topically nearby).

Do not fix bad retrieval by inferring answers yourself. Be strict — a fluent but unsupported answer is worse \
than a refusal."""


def evaluate_retrieval(query: str, retrieved: list) -> RetrievalEvaluation:
    context_block = "\n\n".join(
        f"[chunk_id: {r['doc'].metadata['chunk_id']}] "
        f"(source: {r['doc'].metadata['source']}, section: {r['doc'].metadata['section']}, "
        f"score: {r['confidence']:.3f})\n{r['doc'].page_content}"
        for r in retrieved
    )
    user_prompt = f"User question:\n{query}\n\nRetrieved chunks:\n{context_block}"

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=EVALUATOR_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=RetrievalEvaluation,
                temperature=0.0,
            ),
        )
        return response.parsed
    except Exception:
        # Fail-safe: if the evaluator call itself fails, treat as insufficient evidence
        return RetrievalEvaluation(
            in_scope=True,
            sufficient_evidence=False,
            relevant_chunk_ids=[],
            evidence_gap="Evaluator call failed.",
            reasoning="Exception during evaluation call.",
        )
