"""
Prompt Refinement Assistant for EndoAI.
Transforms vague user inputs into high-precision, guideline-grounded clinical prompts
optimized for ESC 2023 & NICE CG64 retrieval without triggering 'insufficient evidence'.
"""
import json
from pydantic import BaseModel, Field
from app.core.llm_client import client, OPENAI_MODEL


class RefinedPromptResult(BaseModel):
    refined_prompts: list[str] = Field(
        description="2-4 high-precision clinical questions explicitly phrased for ESC 2023 and NICE endocarditis guidelines"
    )
    detected_intent: str = Field(
        description="Summary of the clinical domain detected (e.g., Antibiotic Prophylaxis, Modified Duke Criteria, Urgent Surgery, Blood Cultures)"
    )
    target_guideline: str = Field(
        description="Most relevant guideline (ESC 2023, NICE CG64, or Both)"
    )
    tips: list[str] = Field(
        description="1-2 actionable tips on why these phrased prompts maximize evidence retrieval"
    )


PROMPT_REFINER_SYSTEM = """You are an expert Clinical Informatics & Prompt Engineering Assistant specialized in Infective Endocarditis (IE).
Your job is to take a user's rough, colloquial, or underspecified clinical question and reformulate it into 2 to 3 precise, professional, guideline-aligned questions.

Guidelines in the database:
1. ESC 2023 Guidelines: Diagnostic criteria, Echocardiography (TTE/TEE), Class I/II indications for Urgent/Emergent Surgery, Native vs Prosthetic valve antimicrobial therapy, Cardiac Device Infections.
2. NICE Guidelines (CG64): Prophylaxis recommendations for dental and non-dental interventional procedures, cumulative bacteremia from daily toothbrushing vs single dental procedure, non-routine antibiotic stance.

Requirements:
- Transform vague queries (e.g., 'antibiotic rules', 'can I give amoxicillin?', 'surgery') into explicit, unambiguous questions that mention the specific guideline, clinical context, or patient condition.
- Avoid vague open-ended phrasing ('tell me all about endocarditis') which causes retrieval failure.
- Include key clinical discriminators (e.g., 'dental procedures', 'severe acute regurgitation', 'vegetations >=10mm', 'blood culture sets')."""


def refine_user_prompt(query: str) -> RefinedPromptResult:
    """Uses o4-mini to refine a user prompt into structured high-precision clinical queries."""
    clean_query = query.strip()
    if not clean_query:
        return RefinedPromptResult(
            refined_prompts=[
                "What are the main diagnostic criteria for infective endocarditis according to ESC guidelines?",
                "Does NICE recommend routine antibiotic prophylaxis for dental procedures in patients at risk of IE?",
                "What are the indications for urgent surgery in infective endocarditis according to ESC guidelines?"
            ],
            detected_intent="General Clinical Guidelines",
            target_guideline="ESC 2023 & NICE CG64",
            tips=["Choose a specific clinical domain (diagnosis, prophylaxis, surgery, or microbiology) for highest evidence accuracy."]
        )

    try:
        response = client.beta.chat.completions.parse(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": PROMPT_REFINER_SYSTEM},
                {"role": "user", "content": f"Please refine this clinical inquiry into high-retrieval prompts:\n\n'{clean_query}'"},
            ],
            response_format=RefinedPromptResult,
        )
        parsed = response.choices[0].message.parsed
        if parsed:
            return parsed
    except Exception as e:
        print(f"[PromptRefiner] Fallback due to error: {e}", flush=True)

    # Curated heuristic fallback if offline
    return _fallback_refinement(clean_query)


def _fallback_refinement(q: str) -> RefinedPromptResult:
    lower = q.lower()
    if "brush" in lower or "tooth" in lower or "dental" in lower:
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
    elif "diagnos" in lower or "duke" in lower or "criteria" in lower or "echo" in lower:
        return RefinedPromptResult(
            refined_prompts=[
                "What are the primary major and minor diagnostic criteria for infective endocarditis according to ESC guidelines?",
                "What is the role of transoesophageal echocardiography (TEE) vs transthoracic echocardiography (TTE) in suspected IE?",
                "How are blood culture criteria defined in the modified Duke classification?"
            ],
            detected_intent="Diagnostic Criteria & Imaging",
            target_guideline="ESC 2023",
            tips=["Asking about Major vs Minor criteria or TTE/TEE role gives direct 100% matches."]
        )
    elif "culture" in lower or "microbio" in lower or "blood" in lower:
        return RefinedPromptResult(
            refined_prompts=[
                "What are the recommended blood culture sampling protocols for suspected infective endocarditis?",
                "How many blood culture sets and bottles are recommended prior to starting antimicrobial therapy in IE?",
                "What diagnostic approach is recommended for blood culture-negative endocarditis (BCNE) in ESC guidelines?"
            ],
            detected_intent="Microbiology & Blood Culture Protocol",
            target_guideline="ESC 2023",
            tips=["Mentioning sample count, timing, or culture-negative protocols pulls exact guideline excerpts."]
        )
    else:
        return RefinedPromptResult(
            refined_prompts=[
                f"What do the ESC 2023 guidelines state regarding {q} in infective endocarditis?",
                f"How is {q} clinically managed according to NICE endocarditis recommendations?",
                f"What is the evidence base for {q} in patients with infective endocarditis?"
            ],
            detected_intent="Clinical Inquiry",
            target_guideline="ESC 2023 / NICE",
            tips=["Phrasing your question around specific recommendations or management principles prevents 'insufficient evidence' refusals."]
        )
