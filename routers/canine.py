"""
VetOnco — Canine TCC API Router
All endpoints under /api/canine
"""
from __future__ import annotations
import asyncio
from dataclasses import asdict
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field

from services.tcc_scorer import score_tcc, TCCResult
from services.tcc_dosing import compute_canine_dose, compute_full_panel_dosage
from services.recipe_generator import generate_recipe_card
from services.tcc_test_analyzer import analyze_test_results
from services.chembl_client import enrich_compound, enrich_panel
from services.llm_service import (
    generate_test_narrative,
    generate_drug_rationale,
    generate_pipeline_summary,
)
from auth import get_current_user_id

router = APIRouter(prefix="/api/canine", tags=["canine"])

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ScoreRequest(BaseModel):
    expr: dict[str, float] = Field(default_factory=dict, description="Gene→log2FC expression map")
    braf_status: Literal["positive", "negative", "unknown"] = "unknown"
    breed: str = "other"
    msh2_loss: bool = False


class DosageRequest(BaseModel):
    drug: str
    weight_kg: float
    creatinine_mg_dl: Optional[float] = None
    alt_u_l: Optional[float] = None


class PanelDosageRequest(BaseModel):
    weight_kg: float
    creatinine_mg_dl: Optional[float] = None
    alt_u_l: Optional[float] = None
    drugs: Optional[list[str]] = None


class RecipeRequest(BaseModel):
    pet_name: str
    species: str = "Canis lupus familiaris"
    breed: str
    weight_kg: float
    prescribing_vet: str
    drugs: list[str]
    creatinine_mg_dl: Optional[float] = None
    alt_u_l: Optional[float] = None


class TestLogRequest(BaseModel):
    pet_id: str
    test_type: Literal["cbc", "chemistry", "urinalysis", "braf_urine", "imaging"]
    values: dict[str, Any]


class CompoundRequest(BaseModel):
    drugs: list[str]


class NarrativeRequest(BaseModel):
    test_type: str
    findings: list[dict[str, Any]]
    alerts: list[str]
    pet_name: str
    overall_grade: int = 0


class RationaleRequest(BaseModel):
    recommendations: list[dict[str, Any]]
    braf_status: str = "unknown"
    breed: str = "other"
    subtype: str = "TCC-NOS"
    pet_name: str = "Patient"


class PipelineRequest(BaseModel):
    pet_name: str
    breed: str
    weight_kg: float
    braf_status: Literal["positive", "negative", "unknown"] = "unknown"
    msh2_loss: bool = False
    expr: dict[str, float] = Field(default_factory=dict)
    creatinine_mg_dl: Optional[float] = None
    alt_u_l: Optional[float] = None
    prescribing_vet: str = "VetOnco System"


# In-memory test log (Phase 1 — no DB)
_test_log: dict[str, list[dict]] = {}

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/tcc/score")
async def tcc_score(req: ScoreRequest, user_id: str = Depends(get_current_user_id)):
    result = score_tcc(
        expr=req.expr,
        braf_status=req.braf_status,
        breed=req.breed,
        msh2_loss=req.msh2_loss,
    )
    return {
        "subtype": result.subtype,
        "braf_status": result.braf_status,
        "braf_probability": result.braf_probability,
        "msh2_loss": result.msh2_loss,
        "top_altered_genes": result.top_altered_genes,
        "summary": result.summary,
        "recommendations": [asdict(r) for r in result.recommendations],
        "quarantined": [asdict(r) for r in result.quarantined],
    }


@router.post("/tcc/compounds")
async def tcc_compounds(req: CompoundRequest, user_id: str = Depends(get_current_user_id)):
    compounds = await enrich_panel(req.drugs)
    return {"compounds": [asdict(c) for c in compounds]}


@router.post("/tcc/dosage")
async def tcc_dosage(req: DosageRequest, user_id: str = Depends(get_current_user_id)):
    result = compute_canine_dose(
        drug_name=req.drug,
        weight_kg=req.weight_kg,
        creatinine_mg_dl=req.creatinine_mg_dl,
        alt_u_l=req.alt_u_l,
    )
    return asdict(result)


@router.post("/tcc/dosage/panel")
async def tcc_dosage_panel(req: PanelDosageRequest, user_id: str = Depends(get_current_user_id)):
    results = compute_full_panel_dosage(
        weight_kg=req.weight_kg,
        creatinine_mg_dl=req.creatinine_mg_dl,
        alt_u_l=req.alt_u_l,
        drugs=req.drugs,
    )
    return {"dosages": [asdict(r) for r in results]}


@router.post("/tcc/recipe")
async def tcc_recipe(req: RecipeRequest, user_id: str = Depends(get_current_user_id)):
    """
    Full pharmacist recipe card — PharmacistDrugEntry schema (16 fields/drug) + pdf_b64.
    Frozen: delegates to compute_full_panel_dosage() and generate_recipe_card() only.
    """
    import base64
    from dataclasses import asdict as _asdict

    dose_results = compute_full_panel_dosage(
        weight_kg=req.weight_kg,
        creatinine_mg_dl=req.creatinine_mg_dl,
        alt_u_l=req.alt_u_l,
        drugs=req.drugs,
    )
    card = generate_recipe_card(
        pet_name=req.pet_name,
        species=req.species,
        breed=req.breed,
        weight_kg=req.weight_kg,
        prescribing_vet=req.prescribing_vet,
        dose_results=dose_results,
        generate_pdf=True,
    )

    drugs_serialized = [_asdict(d) for d in card.drugs]
    pdf_b64 = (
        base64.b64encode(card.pdf_bytes).decode("utf-8")
        if card.pdf_bytes
        else None
    )

    return {
        "pet_name": card.pet_name,
        "species": card.species,
        "breed": card.breed,
        "weight_kg": card.weight_kg,
        "bsa_m2": card.bsa_m2,
        "prescribing_vet": card.prescribing_vet,
        "rx_number": card.rx_number,
        "date_issued": card.date_issued,
        "label_text": card.label_text,
        "pdf_b64": pdf_b64,
        "drugs": drugs_serialized,
        "interactions": card.interactions,
        "monitoring": card.monitoring,
        "printable_text": card.printable_text,
    }


@router.post("/tcc/tests")
async def log_test(req: TestLogRequest, user_id: str = Depends(get_current_user_id)):
    if req.pet_id not in _test_log:
        _test_log[req.pet_id] = []
    entry = {"test_type": req.test_type, "values": req.values, "user_id": user_id}
    _test_log[req.pet_id].append(entry)
    return {"status": "logged", "pet_id": req.pet_id, "count": len(_test_log[req.pet_id])}


@router.get("/tcc/tests/{pet_id}")
async def get_tests(pet_id: str, user_id: str = Depends(get_current_user_id)):
    return {"pet_id": pet_id, "tests": _test_log.get(pet_id, [])}


@router.post("/tcc/analyze-tests")
async def analyze_tests(req: TestLogRequest, user_id: str = Depends(get_current_user_id)):
    result = analyze_test_results(
        test_type=req.test_type,
        values=req.values,
        pet_id=req.pet_id,
    )
    return {
        "test_type": result.test_type,
        "pet_id": result.pet_id,
        "findings": [asdict(f) for f in result.findings],
        "alerts": result.alerts,
        "overall_grade": result.overall_grade,
        "clinical_action": result.clinical_action,
        "braf_positive": result.braf_positive,
        "imaging_response": result.imaging_response,
    }


@router.post("/tcc/narrative")
async def tcc_narrative(req: NarrativeRequest, user_id: str = Depends(get_current_user_id)):
    narrative = await generate_test_narrative(
        test_type=req.test_type,
        findings=req.findings,
        alerts=req.alerts,
        pet_name=req.pet_name,
        overall_grade=req.overall_grade,
    )
    return {"narrative": narrative, "available": narrative is not None}


@router.post("/tcc/rationale")
async def tcc_rationale(req: RationaleRequest, user_id: str = Depends(get_current_user_id)):
    rationale = await generate_drug_rationale(
        recommendations=req.recommendations,
        braf_status=req.braf_status,
        breed=req.breed,
        subtype=req.subtype,
        pet_name=req.pet_name,
    )
    return {"rationale": rationale, "available": rationale is not None}


@router.post("/tcc/pipeline")
async def tcc_pipeline(req: PipelineRequest, user_id: str = Depends(get_current_user_id)):
    steps = []
    errors = []

    # Step 1: Score
    try:
        result = score_tcc(
            expr=req.expr,
            braf_status=req.braf_status,
            breed=req.breed,
            msh2_loss=req.msh2_loss,
        )
        top_drugs = [r.drug for r in result.recommendations[:3]]
        steps.append({
            "step": "score",
            "status": "ok",
            "summary": result.summary,
            "subtype": result.subtype,
            "top_drugs": top_drugs,
        })
    except Exception as e:
        errors.append(str(e))
        steps.append({"step": "score", "status": "error", "error": str(e)})
        return {"steps": steps, "errors": errors, "complete": False}

    # Step 2: ChEMBL enrichment
    try:
        compounds = await enrich_panel(top_drugs)
        steps.append({
            "step": "chembl",
            "status": "ok",
            "summary": f"Enriched {len(compounds)} compounds",
            "compounds": [asdict(c) for c in compounds],
        })
    except Exception as e:
        errors.append(str(e))
        steps.append({"step": "chembl", "status": "error", "error": str(e)})

    # Step 3: Dosage
    try:
        doses = compute_full_panel_dosage(
            weight_kg=req.weight_kg,
            creatinine_mg_dl=req.creatinine_mg_dl,
            alt_u_l=req.alt_u_l,
            drugs=top_drugs,
        )
        steps.append({
            "step": "dosage",
            "status": "ok",
            "summary": f"Computed doses for {len(doses)} drugs",
            "dosages": [asdict(d) for d in doses],
        })
    except Exception as e:
        errors.append(str(e))
        steps.append({"step": "dosage", "status": "error", "error": str(e)})
        doses = []

    # Step 4: Recipe card
    try:
        card = generate_recipe_card(
            pet_name=req.pet_name,
            species="Canis lupus familiaris",
            breed=req.breed,
            weight_kg=req.weight_kg,
            prescribing_vet=req.prescribing_vet,
            dose_results=doses,
        )
        steps.append({
            "step": "recipe",
            "status": "ok",
            "summary": f"Recipe card generated for {req.pet_name}",
            "recipe": {
                "drugs": card.drugs,
                "interactions": card.interactions,
                "monitoring": card.monitoring,
                "printable_text": card.printable_text,
            },
        })
    except Exception as e:
        errors.append(str(e))
        steps.append({"step": "recipe", "status": "error", "error": str(e)})

    # Step 5: LLM rationale
    try:
        rationale = await generate_drug_rationale(
            recommendations=[asdict(r) for r in result.recommendations[:4]],
            braf_status=req.braf_status,
            breed=req.breed,
            subtype=result.subtype,
            pet_name=req.pet_name,
        )
        steps.append({
            "step": "rationale",
            "status": "ok",
            "summary": "LLM rationale generated" if rationale else "LLM unavailable — raw data returned",
            "rationale": rationale,
        })
    except Exception as e:
        errors.append(str(e))
        steps.append({"step": "rationale", "status": "error", "error": str(e)})

    return {
        "steps": steps,
        "errors": errors,
        "complete": len(errors) == 0,
        "pet_name": req.pet_name,
        "breed": req.breed,
        "braf_status": req.braf_status,
        "subtype": result.subtype,
        "top_drugs": top_drugs,
    }


# ---------------------------------------------------------------------------
# Chat endpoint (SSE streaming)
# ---------------------------------------------------------------------------

from fastapi.responses import StreamingResponse
from services.chat_agent import chat as agent_chat, clear_history


class ChatRequest(BaseModel):
    message: str
    pet_context: dict = Field(default_factory=dict)


@router.post("/chat/{pet_id}")
async def pet_chat(
    pet_id: str,
    req: ChatRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Streaming SSE chat endpoint for a pet.
    Returns text/event-stream with chunked assistant response.
    RAG citations appended as __RAG_CITATIONS__[...]__END_CITATIONS__ sentinel.
    """
    async def generate():
        async for chunk in agent_chat(
            pet_id=f"{user_id}:{pet_id}",   # namespace by user to prevent cross-user leakage
            user_message=req.message,
            pet_context=req.pet_context,
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@router.delete("/chat/{pet_id}/history")
async def clear_chat_history(
    pet_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Clear conversation history for a pet."""
    clear_history(f"{user_id}:{pet_id}")
    return {"status": "cleared", "pet_id": pet_id}


@router.post("/rag/query")
async def rag_query(
    body: dict,
    user_id: str = Depends(get_current_user_id),
):
    """Direct RAG retrieval endpoint for testing."""
    from services.rag_service import retrieve
    query = body.get("query", "")
    k = int(body.get("k", 3))
    results = retrieve(query, k=k)
    return {"query": query, "results": results}
