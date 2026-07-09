"""
VetOnco — Persistent LLM Chat Agent
Stateful conversation per pet. Context: pet profile, drug rankings,
quarantine list, monitoring alerts, RAG literature retrieval.

LLM STRAITJACKET (NON-NEGOTIABLE):
  The LLM MUST NOT perform arithmetic, calculate doses, or determine gap ratios.
  All numbers come from deterministic backend tools. The LLM's role:
    1. Interpret and explain tool outputs in clinical language
    2. Answer questions about the literature (via RAG)
    3. Summarize monitoring trends
    4. Explain WHY a drug is quarantined or ranked
  The LLM MUST NOT:
    - Calculate or estimate any dose
    - Compute or estimate any gap ratio
    - State any ORR, IC50, Cmax, or PPB value not returned by a tool
    - Override or second-guess quarantine decisions
"""
from __future__ import annotations
import json
import os
from typing import AsyncGenerator, Optional

from services.rag_service import retrieve

# In-memory conversation store: {pet_id: [{"role": ..., "content": ...}]}
_CONVERSATION_STORE: dict[str, list[dict]] = {}
MAX_HISTORY = 20


def _build_system_prompt(pet_context: dict) -> str:
    pet_name = pet_context.get("pet_name", "this patient")
    breed = pet_context.get("breed", "unknown breed")
    weight_kg = pet_context.get("weight_kg", "unknown")
    braf_status = pet_context.get("braf_status", "unknown")
    creatinine = pet_context.get("creatinine_mg_dl")
    subtype = pet_context.get("subtype", "TCC-NOS")
    top_drugs = pet_context.get("top_drugs", [])
    quarantined = pet_context.get("quarantined_drugs", [])
    alerts = pet_context.get("monitoring_alerts", [])

    top_drugs_text = "\n".join(
        f"  - {d['drug'].upper()}: score={d['score']:.3f}, ORR={d.get('orr_pct','?')}%, "
        f"feasibility={d.get('feasibility_verdict','?')}, gap={d.get('gap_ratio','?')}×"
        for d in top_drugs
    ) or "  None (all drugs quarantined)"

    quarantine_text = "\n".join(
        f"  - {d['drug'].upper()}: {d.get('quarantine_reason','UNVERIFIED')}"
        for d in quarantined[:6]
    ) or "  None"

    alerts_text = "\n".join(f"  - {a}" for a in alerts) or "  No active alerts"
    creatinine_text = f"{creatinine} mg/dL" if creatinine else "not recorded"

    return f"""You are VetOnco, a veterinary oncology clinical decision support assistant.
You are helping a veterinarian manage {pet_name}, a {breed} with canine TCC (transitional cell carcinoma).

CURRENT PATIENT PROFILE:
  Name: {pet_name}
  Breed: {breed}
  Weight: {weight_kg} kg
  BRAF V595E status: {braf_status}
  Subtype: {subtype}
  Creatinine: {creatinine_text}

CURRENT DRUG RANKINGS (verified drugs only):
{top_drugs_text}

QUARANTINED DRUGS (pending primary canine PK sourcing):
{quarantine_text}

ACTIVE MONITORING ALERTS:
{alerts_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES — YOU MUST FOLLOW THESE WITHOUT EXCEPTION:

1. ARITHMETIC PROHIBITION: You MUST NOT calculate, estimate, or state any:
   - Drug dose (mg, mg/kg, mg/m²)
   - Gap ratio (IC50 / free_Cmax)
   - Free Cmax value
   - ORR percentage not already shown above
   - BSA calculation
   If asked for a dose or gap ratio, respond: "I cannot calculate doses or PK
   values. Use the Dosing Calculator or Compound Analysis tools for exact numbers."

2. QUARANTINE IS FINAL: You MUST NOT suggest that a quarantined drug could be
   used, even if the vet asks. Quarantine decisions are made by the deterministic
   pipeline, not by you. You may explain WHY a drug is quarantined.

3. CITE LITERATURE: When discussing ORR, PK, or trial data, cite the specific
   paper (e.g., "Henry et al. 2009, PMID 19185954"). Do not state numbers from
   memory — only numbers shown in the patient context above or returned by RAG.

4. CLINICAL LANGUAGE: Address the veterinarian directly. Be precise and concise.
   Do not use hedging language like "I think" or "possibly" for factual claims.
   Use "I don't have data on that" when you genuinely don't.

5. SCOPE: You support clinical decision-making. You do not replace the veterinarian.
   Always recommend confirming recommendations with a board-certified veterinary oncologist.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


def get_history(pet_id: str) -> list[dict]:
    return _CONVERSATION_STORE.get(pet_id, [])


def add_message(pet_id: str, role: str, content: str) -> None:
    if pet_id not in _CONVERSATION_STORE:
        _CONVERSATION_STORE[pet_id] = []
    _CONVERSATION_STORE[pet_id].append({"role": role, "content": content})
    if len(_CONVERSATION_STORE[pet_id]) > MAX_HISTORY:
        _CONVERSATION_STORE[pet_id] = _CONVERSATION_STORE[pet_id][-MAX_HISTORY:]


def clear_history(pet_id: str) -> None:
    _CONVERSATION_STORE[pet_id] = []


async def chat(
    pet_id: str,
    user_message: str,
    pet_context: dict,
) -> AsyncGenerator[str, None]:
    """
    Stream a chat response for a pet. Injects RAG context when relevant.

    pet_context must include:
        pet_name, breed, weight_kg, braf_status, subtype,
        top_drugs (list of dicts), quarantined_drugs (list of dicts),
        monitoring_alerts (list of str), creatinine_mg_dl (optional)
    """
    # RAG retrieval
    rag_chunks = retrieve(user_message, k=2)
    rag_context = ""
    if rag_chunks:
        rag_context = "\n\nRELEVANT LITERATURE (from VetOnco knowledge base):\n"
        for chunk in rag_chunks:
            rag_context += (
                f"  [{chunk['citation']}] {chunk['key_finding']} "
                f"(relevance: {chunk['bm25_score']})\n"
            )

    full_user_message = user_message + rag_context
    add_message(pet_id, "user", full_user_message)

    system_prompt = _build_system_prompt(pet_context)
    history = get_history(pet_id)
    messages = [{"role": "system", "content": system_prompt}] + history

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        yield "LLM service unavailable — OPENROUTER_API_KEY not set."
        return

    try:
        from openai import AsyncOpenAI
        stream_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=30.0,
        )

        full_response = ""
        stream = await stream_client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=messages,
            max_tokens=800,
            temperature=0.2,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_response += delta
                yield delta

        add_message(pet_id, "assistant", full_response)

        if rag_chunks:
            citations = [
                {"citation": c["citation"], "drug": c["drug"], "bm25_score": c["bm25_score"]}
                for c in rag_chunks
            ]
            yield f"\n\n__RAG_CITATIONS__{json.dumps(citations)}__END_CITATIONS__"

    except Exception as e:
        error_msg = f"Chat error: {str(e)}"
        yield error_msg
        add_message(pet_id, "assistant", error_msg)
