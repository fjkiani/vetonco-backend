"""
VetOnco — OpenRouter LLM Service
Generates plain-English narratives for test results and drug recommendation rationale.
Uses OpenAI SDK pointed at OpenRouter's API endpoint.
"""
from __future__ import annotations
import os
import asyncio
from typing import Any

from openai import AsyncOpenAI

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PRIMARY_MODEL = "anthropic/claude-3-haiku"
FALLBACK_MODEL = "openai/gpt-4o-mini"
TIMEOUT_SECONDS = 15.0

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            timeout=TIMEOUT_SECONDS,
        )
    return _client


async def _chat(system: str, user: str, model: str = PRIMARY_MODEL) -> str | None:
    """Send a chat completion request; return text or None on failure."""
    if not OPENROUTER_API_KEY:
        return None
    client = _get_client()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=600,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Try fallback model once
        if model == PRIMARY_MODEL:
            try:
                response = await client.chat.completions.create(
                    model=FALLBACK_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=600,
                    temperature=0.3,
                )
                return response.choices[0].message.content
            except Exception:
                pass
        return None


async def generate_test_narrative(
    test_type: str,
    findings: list[dict],
    alerts: list[str],
    pet_name: str,
    overall_grade: int,
) -> str | None:
    """
    Generate a plain-English summary of test results for a pet owner and vet.
    Returns None if LLM is unavailable (frontend shows raw data instead).
    """
    system = (
        "You are a veterinary oncology clinical assistant. "
        "Write a clear, compassionate, and medically accurate summary of laboratory test results "
        "for a dog with transitional cell carcinoma (TCC/UC). "
        "Address both the pet owner and the veterinarian. "
        "Keep it under 150 words. Use plain language for the owner section, "
        "and precise clinical language for the vet section. "
        "Do not recommend specific drugs — that is handled separately."
    )

    findings_text = "\n".join(
        f"- {f.get('parameter', '')}: {f.get('value', '')} {f.get('unit', '')} "
        f"(Grade {f.get('grade', 0)}: {f.get('interpretation', '')})"
        for f in findings
    )
    alerts_text = "\n".join(f"- {a}" for a in alerts) if alerts else "None"

    user = (
        f"Patient: {pet_name}\n"
        f"Test type: {test_type.upper()}\n"
        f"Overall VCOG-CTCAE grade: {overall_grade}\n\n"
        f"Findings:\n{findings_text}\n\n"
        f"Alerts:\n{alerts_text}\n\n"
        "Please write the summary."
    )

    return await _chat(system, user)


async def generate_drug_rationale(
    recommendations: list[dict],
    braf_status: str,
    breed: str,
    subtype: str,
    pet_name: str,
) -> str | None:
    """
    Generate a bullet-point rationale for the top drug recommendations.
    Returns None if LLM is unavailable.
    """
    system = (
        "You are a veterinary oncology clinical decision support system. "
        "Explain the drug ranking for a canine TCC patient in clear, evidence-based language. "
        "Write 2-4 bullet points covering: why the top drug is ranked first, "
        "how BRAF status influences the ranking, and any key monitoring considerations. "
        "Keep it under 200 words. Be specific and cite the mechanism of action."
    )

    top_drugs = recommendations[:4]  # Top 4 only
    drugs_text = "\n".join(
        f"- Rank {r.get('rank', '?')}: {r.get('drug', '').upper()} "
        f"(score {r.get('score', 0):.2f}) — {r.get('mechanism', '')} | "
        f"Rationale: {r.get('rationale', '')}"
        for r in top_drugs
    )

    user = (
        f"Patient: {pet_name} ({breed})\n"
        f"Subtype: {subtype}\n"
        f"BRAF status: {braf_status}\n\n"
        f"Top drug recommendations:\n{drugs_text}\n\n"
        "Please explain the ranking rationale."
    )

    return await _chat(system, user)


async def generate_pipeline_summary(
    pipeline_steps: list[dict],
    pet_name: str,
    breed: str,
    braf_status: str,
) -> str | None:
    """
    Generate a one-paragraph executive summary of the full pipeline output.
    """
    system = (
        "You are a veterinary oncology clinical assistant. "
        "Write a concise executive summary (under 120 words) of a TCC treatment pipeline result "
        "for a veterinarian. Include: subtype, top drug recommendation, key safety flags, "
        "and next monitoring steps. Be precise and clinical."
    )

    steps_text = "\n".join(
        f"- {s.get('step', '')}: {s.get('status', '')} — {s.get('summary', '')}"
        for s in pipeline_steps
    )

    user = (
        f"Patient: {pet_name} ({breed})\n"
        f"BRAF status: {braf_status}\n\n"
        f"Pipeline steps:\n{steps_text}\n\n"
        "Please write the executive summary."
    )

    return await _chat(system, user)
