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


PROMPT_REFINER_SYSTEM = """You are an expert Clinical Prompt Engineer for EndoAI, an evidence-grounded decision support system backed by ESC 2023 and NICE CG64 Infective Endocarditis (IE) guidelines.

Your mission is to take any user input—whether rough, colloquial, vague, or a complex patient vignette (e.g. 'patient with epigastric pain and suspected IE')—and reformulate it into 2 to 3 crystal-clear, focused clinical questions that DIRECTLY match guideline headings and recommendation tables.

CRITICAL RULES FOR REFINEMENT:
1. STRIP VIGNETTE NOISE: Remove patient-specific story details ('45-year old with epigastric pain', 'patient presenting to ER') because guidelines do not index individual patient cases.
2. BREAK DOWN COMPOSITE QUESTIONS: If the user asks for multiple things at once (e.g., blood cultures AND echo AND surgery), provide distinct, focused single-topic questions so each can retrieve 100% complete evidence.
3. USE OFFICIAL GUIDELINE PHRASING:
   - For Echocardiography: "What are the ESC 2023 guideline recommendations for performing TTE and TOE (transoesophageal echocardiography) in suspected infective endocarditis?"
   - For Blood Cultures & Microbiology: "How are blood culture findings and microbiological criteria defined in the ESC 2023 diagnostic criteria for infective endocarditis?" or "What microbiological investigations are recommended in suspected infective endocarditis?"
   - For Diagnostic Criteria: "What are the major and minor imaging and microbiological diagnostic criteria for infective endocarditis according to the 2023 ESC guidelines?"
   - For Prophylaxis / Dental: "Why did NICE recommend against routine antibiotic prophylaxis for dental procedures in patients at risk of infective endocarditis?"
   - For Surgery: "What are the ESC 2023 Class I indications for urgent or emergent cardiac surgery in infective endocarditis?"
4. GUARANTEE EVIDENCE MATCH: Formulate questions that are guaranteed to have direct answers in the guidelines so the system NEVER refuses with 'insufficient evidence'."""


def refine_user_prompt(query: str) -> RefinedPromptResult:
    """Uses o4-mini to refine a user prompt into structured high-precision clinical queries."""
    clean_query = query.strip()
    if not clean_query:
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

    try:
        response = client.beta.chat.completions.parse(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": PROMPT_REFINER_SYSTEM},
                {"role": "user", "content": f"Transform this clinical inquiry into high-retrieval guideline questions:\n\n'{clean_query}'"},
            ],
            response_format=RefinedPromptResult,
        )
        parsed = response.choices[0].message.parsed
        if parsed and parsed.refined_prompts:
            return parsed
    except Exception as e:
        print(f"[PromptRefiner] Fallback due to error: {e}", flush=True)

    # Curated heuristic fallback if offline
    return _fallback_refinement(clean_query)


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
