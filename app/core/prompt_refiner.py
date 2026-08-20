"""
Prompt Refinement Assistant for EndoAI.
Transforms vague, composite, or vignette-laden user inputs into high-precision, guideline-grounded clinical prompts
optimized for ESC 2023 & NICE CG64 retrieval without triggering 'insufficient evidence'.
"""
import json
from pydantic import BaseModel, Field
from app.core.llm_client import client, OPENAI_MODEL


class RefinedPromptResult(BaseModel):
    refined_prompts: list[str] = Field(
        description="2-4 high-precision clinical questions explicitly phrased to match sections in ESC 2023 and NICE endocarditis guidelines"
    )
    detected_intent: str = Field(
        description="Summary of the clinical domain detected (e.g., Echocardiography Work-Up, Antibiotic Prophylaxis, Modified Duke Criteria, Urgent Surgery, Blood Cultures)"
    )
    target_guideline: str = Field(
        description="Most relevant guideline (ESC 2023, NICE CG64, or Both)"
    )
    tips: list[str] = Field(
        description="1-2 actionable tips on why these phrased prompts maximize evidence retrieval"
    )


PROMPT_REFINER_SYSTEM = """You are an expert Clinical Prompt Engineer for EndoAI, a clinical decision support system grounded in ESC 2023 and NICE CG64 Infective Endocarditis guidelines.

Your mission: Take any user input (rough keywords, vague questions, or complex patient vignettes like 'patient with epigastric pain and suspected IE') and reformulate it into 2-3 crystal-clear clinical questions that DIRECTLY match guideline headings and recommendation tables so the system can always retrieve real evidence.

CRITICAL RULES:
1. STRIP VIGNETTE NOISE: Remove patient-specific details ('45-year old', 'epigastric pain', 'presenting to ER'). Guidelines index disease concepts and recommendations, not individual cases.
2. BREAK COMPOSITE QUESTIONS: If the user asks about multiple topics at once (blood cultures AND echo AND surgery), split into separate focused questions — one topic each.
3. USE KNOWN GUIDELINE TOPIC PHRASING:
   - Echocardiography: Ask about TTE and TOE indications in suspected IE per ESC 2023
   - Blood Cultures / Microbiology: Ask about microbiological criteria or blood culture-negative endocarditis in ESC 2023 diagnostic criteria
   - Diagnostic Criteria: Ask about major/minor criteria, Duke criteria modifications, or imaging modalities in ESC 2023 IE diagnosis
   - Dental / Prophylaxis: Ask why NICE moved away from antibiotic prophylaxis, bacteremia risk from toothbrushing, or oral hygiene advice
   - Surgery: Ask about Class I indications, urgency timing, or valve type-specific indications for surgery in IE per ESC 2023
4. GUARANTEE EVIDENCE: Every question you generate MUST be directly answerable by the guidelines.

If real guideline chunks are provided below, use their exact section titles and terminology in your questions."""


def refine_user_prompt(query: str, rag_index=None) -> RefinedPromptResult:
    """
    Refines a user prompt into precise guideline-grounded questions.
    - If the RAG index is ready: retrieves top matching chunks and grounds the LLM in real content.
    - If the index is still building: calls LLM with guideline system prompt alone (no fallback to static templates).
    - Only falls back to static templates if the OpenAI API call itself fails.
    """
    clean_query = query.strip()
    if not clean_query:
        return _empty_result()

    # 1. Try to get real chunk context for grounding
    chunk_context = ""
    if rag_index is not None and getattr(rag_index, "ready", False):
        try:
            chunk_context = _build_chunk_context(clean_query, rag_index)
        except Exception as e:
            print(f"[PromptRefiner] Chunk grounding skipped: {e}", flush=True)

    # 2. Build user message — with or without chunk context
    if chunk_context:
        user_message = (
            f"User query: '{clean_query}'\n\n"
            f"Real guideline sections found in the index:\n\n"
            f"{chunk_context}\n\n"
            "Generate 2-3 precise questions grounded in these exact sections."
        )
    else:
        user_message = f"Transform this clinical inquiry into 2-3 high-precision ESC 2023 / NICE CG64 guideline questions:\n\n'{clean_query}'"

    # 3. Always call the LLM
    try:
        response = client.beta.chat.completions.parse(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": PROMPT_REFINER_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            response_format=RefinedPromptResult,
        )
        parsed = response.choices[0].message.parsed
        if parsed and parsed.refined_prompts:
            return parsed
    except Exception as e:
        print(f"[PromptRefiner] LLM call failed, using fallback: {e}", flush=True)

    # 4. Static fallback only if OpenAI API is completely unreachable
    return _fallback_refinement(clean_query)


def _build_chunk_context(query: str, rag_index, top_k: int = 8) -> str:
    """Retrieves top-k chunks and formats their section titles + snippets as LLM context."""
    results = rag_index.get_similarity_scores(query, k=top_k)
    lines = []
    seen = set()
    for doc, score in results:
        section = doc.metadata.get("section", "Unknown Section")
        source = doc.metadata.get("source", "")
        snippet = doc.page_content[:200].replace("\n", " ").strip()
        key = (source, section)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"[{source}] Section: \"{section}\" (score {score:.2f})\n  Excerpt: {snippet}...")
    return "\n\n".join(lines)


def _empty_result() -> RefinedPromptResult:
    return RefinedPromptResult(
        refined_prompts=[
            "What are the ESC 2023 guideline recommendations for performing TTE and TOE in suspected infective endocarditis?",
            "What are the main diagnostic criteria for infective endocarditis according to ESC guidelines?",
            "Why did NICE move away from routine antibiotic prophylaxis for dental procedures?"
        ],
        detected_intent="General Clinical Guidelines",
        target_guideline="ESC 2023 & NICE CG64",
        tips=["Choose a focused question to get direct page-level citations from the guidelines."]
    )


def _fallback_refinement(q: str) -> RefinedPromptResult:
    lower = q.lower()
    if "echo" in lower or "tte" in lower or "toe" in lower or "transesophageal" in lower or "transthoracic" in lower:
        return RefinedPromptResult(
            refined_prompts=[
                "What are the ESC 2023 guideline recommendations for performing TTE and TOE in suspected infective endocarditis?",
                "What are the specific indications for transoesophageal echocardiography (TOE) when initial TTE is negative or inconclusive in suspected IE?",
                "What role does echocardiography play in the 2023 ESC diagnostic criteria for infective endocarditis?"
            ],
            detected_intent="Echocardiography (TTE & TOE) Work-Up",
            target_guideline="ESC 2023",
            tips=["ESC guidelines recommend TTE as first-line for all suspected cases, with TOE indicated for high suspicion or inconclusive TTE."]
        )
    elif "brush" in lower or "tooth" in lower or "dental" in lower:
        return RefinedPromptResult(
            refined_prompts=[
                "Why might regular toothbrushing be considered a greater IE risk than a single dental procedure according to NICE?",
                "Does NICE recommend routine antibiotic prophylaxis for interventional dental procedures?",
                "What oral hygiene advice does NICE recommend for patients at high risk of infective endocarditis?"
            ],
            detected_intent="Dental Prophylaxis & Bacteremia",
            target_guideline="NICE CG64",
            tips=["NICE guidelines focus specifically on bacteremia from daily oral care vs single procedures."]
        )
    elif "surg" in lower or "operat" in lower or "urgent" in lower or "valve" in lower:
        return RefinedPromptResult(
            refined_prompts=[
                "What are the indications for urgent surgery in infective endocarditis according to ESC guidelines?",
                "What are the surgical indications for large vegetations (>10 mm) in infective endocarditis?",
                "How does ESC define emergent vs urgent surgery indications in native valve endocarditis?"
            ],
            detected_intent="Surgical Indications & Urgency",
            target_guideline="ESC 2023",
            tips=["Specify valve type (native vs prosthetic) or indication (heart failure, uncontrolled infection, embolism risk)."]
        )
    elif "diagnos" in lower or "duke" in lower or "criteria" in lower:
        return RefinedPromptResult(
            refined_prompts=[
                "What are the primary major and minor diagnostic criteria for infective endocarditis according to ESC guidelines?",
                "What are the 2023 ESC modifications to the Duke diagnostic criteria for infective endocarditis?",
                "What imaging modalities are included in the major diagnostic criteria for IE under ESC 2023?"
            ],
            detected_intent="Diagnostic Criteria & Imaging",
            target_guideline="ESC 2023",
            tips=["Asking about Major vs Minor criteria gives direct 100% matches with exact page citations."]
        )
    elif "culture" in lower or "microbio" in lower or "blood" in lower:
        return RefinedPromptResult(
            refined_prompts=[
                "How are positive blood culture findings defined in the ESC 2023 diagnostic criteria for infective endocarditis?",
                "What microbiological diagnostic algorithm is proposed for suspected infective endocarditis according to ESC guidelines?",
                "What diagnostic approach is recommended for blood culture-negative endocarditis (BCNE) in ESC guidelines?"
            ],
            detected_intent="Microbiology & Blood Culture Protocol",
            target_guideline="ESC 2023",
            tips=["Asking about diagnostic blood culture criteria or microbiological algorithms pulls exact guideline excerpts."]
        )
    else:
        return RefinedPromptResult(
            refined_prompts=[
                f"What do the ESC 2023 guidelines state regarding {q} in infective endocarditis?",
                f"What are the diagnostic and management recommendations for {q} according to ESC guidelines?",
                f"How is {q} clinically addressed in the infective endocarditis guidelines?"
            ],
            detected_intent="Clinical Inquiry",
            target_guideline="ESC 2023 / NICE",
            tips=["Phrasing your question around specific recommendations or management principles prevents 'insufficient evidence' refusals."]
        )
