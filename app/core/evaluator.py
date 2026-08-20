"""
Agent that reviews retrieval quality (scope + sufficiency) before we allow
answer generation to proceed.
"""
from app.core.llm_client import client, OPENAI_MODEL
from app.core.llm_schemas import RetrievalEvaluation

EVALUATOR_SYSTEM_PROMPT = """You are an expert clinical retrieval quality-control agent for an evidence-grounded decision-support system scoped to Infective Endocarditis (IE) guidelines (ESC 2023 & NICE CG64).

Your job is to evaluate whether the retrieved context contains relevant, factual guideline evidence to answer the user's clinical inquiry:

1. SCOPE CHECK:
- Is this question related to Infective Endocarditis (e.g. prevention, diagnosis, imaging like TTE/TOE/CT/PET, microbiology, blood cultures, surgical indications, complications, or antimicrobial management)? If yes, in_scope = true.
- If the question is completely unrelated to cardiac/endocarditis care, mark in_scope = false.

2. SUFFICIENCY & RELEVANCE CHECK:
- If the retrieved chunks contain relevant guideline recommendations, criteria, protocols, or clinical principles that address the main clinical aspects of the user's question, mark sufficient_evidence = true.
- Ignore distracting clinical story details (like 'patient with epigastric pain') and judge whether the retrieved chunks address the core medical topics (e.g. TTE/TOE imaging, diagnostic criteria, surgical urgency).
- Identify all chunk_ids that provide supporting evidence for answering the clinical question.
- Only mark sufficient_evidence = false if NONE of the retrieved chunks contain relevant clinical guidance for the inquiry or if the requested topic is completely absent from the retrieved passages."""


def evaluate_retrieval(query: str, retrieved: list) -> RetrievalEvaluation:
    context_block = "\n\n".join(
        f"[chunk_id: {r['doc'].metadata['chunk_id']}] "
        f"(source: {r['doc'].metadata['source']}, section: {r['doc'].metadata['section']}, "
        f"score: {r.get('rerank_score', r['confidence']):.3f})\n{r['doc'].page_content}"
        for r in retrieved
    )
    user_prompt = f"User question:\n{query}\n\nRetrieved chunks:\n{context_block}"

    try:
        response = client.beta.chat.completions.parse(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=RetrievalEvaluation,
        )
        choice = response.choices[0]
        if choice.message.parsed is None:
            raise ValueError(
                f"Evaluator returned no parsed structured output: {choice.message.content}"
            )
        return choice.message.parsed
    except Exception as e:
        # Fail-safe: if the evaluator call itself fails, treat as insufficient evidence
        return RetrievalEvaluation(
            in_scope=True,
            sufficient_evidence=False,
            relevant_chunk_ids=[],
            evidence_gap=f"Evaluator error: {e}",
            reasoning="Exception during evaluation call.",
        )
