"""
Generate a clinically-grounded answer using only the chunks the evaluator approved.
"""
from google.genai import types

from app.core.llm_client import client, GEMINI_MODEL
from app.core.llm_schemas import GroundedAnswer

GENERATION_SYSTEM_PROMPT = """You are an evidence-grounded clinical decision support assistant, scoped ONLY \
to Infective Endocarditis (IE) guidance from NICE, ESC.

Rules (do not violate any of these):
- Use ONLY the retrieved guideline context provided below. Do not use outside/training knowledge.
- If the context does not fully support a claim, do not state it.
- Do not provide patient-specific diagnosis or treatment.
- Do not infer patient-specific treatment, and do not invent missing thresholds or numbers not present in context.
- Every claim in supporting_evidence must cite the exact chunk_id it came from.
- Every citation must include document, section, page, chunk_id, retrieval_score, and a short exact excerpt \
copied from that chunk — do not cite a chunk that does not actually support the claim.
- Assign confidence (High / Medium / Low / Insufficient Evidence) based on how directly and completely the \
retrieved evidence answers the question — not on your own fluency or certainty.
- Always include a safety_disclaimer stating this supports clinicians and does not replace medical judgment."""


def generate_grounded_answer(query: str, relevant_chunks: list) -> GroundedAnswer:
    context_block = "\n\n".join(
        f"[chunk_id: {r['doc'].metadata['chunk_id']}] "
        f"document: {r['doc'].metadata['source']} | section: {r['doc'].metadata['section']} | "
        f"page: {r['doc'].metadata['section_start_page']} | retrieval_score: {r['confidence']:.3f}\n"
        f"{r['doc'].page_content}"
        for r in relevant_chunks
    )
    user_prompt = f"Clinical question:\n{query}\n\nRetrieved evidence:\n{context_block}"

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=GENERATION_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=GroundedAnswer,
            temperature=0.1,
        ),
    )
    return response.parsed
