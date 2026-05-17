"""
VetOnco — TCC Pipeline LangGraph Agent
StateGraph with 5 nodes + conditional edges.

Graph flow:
  score_node → chembl_node → dosing_node → recipe_node → llm_node → END
  
Conditional edges:
  - score_node: if error → error_node → END
  - score_node: if braf_status == "negative" → exclude trametinib in dosing_node
  - dosing_node: if any drug HOLD → add warning to trace before recipe_node
  
Streaming: nodes are run manually in sequence (not via graph.stream) so the
SSE endpoint can yield a trace event after each node completes in real time.
The compiled graph is used for validation and non-streaming invocations.
"""
from __future__ import annotations
import asyncio
import time
from dataclasses import asdict
from typing import Any

from langgraph.graph import StateGraph, END

from agents.state import (
    TCCAgentState, TraceEvent, make_trace, initial_tcc_state
)
from services.tcc_scorer import score_tcc
from services.chembl_client import enrich_panel
from services.tcc_dosing import compute_full_panel_dosage
from services.recipe_generator import generate_recipe_card
from services.llm_service import generate_drug_rationale, generate_pipeline_summary


# ---------------------------------------------------------------------------
# Node: score_node
# ---------------------------------------------------------------------------

def score_node(state: TCCAgentState) -> TCCAgentState:
    t0 = time.time()
    try:
        result = score_tcc(
            expr=state.get("expr", {}),
            braf_status=state.get("braf_status", "unknown"),
            breed=state.get("breed", "other"),
            msh2_loss=state.get("msh2_loss", False),
        )
        top_drugs = [r.drug for r in result.recommendations[:3]]

        # Conditional: exclude trametinib if BRAF negative
        exclude = []
        if state.get("braf_status") == "negative" and "trametinib" in top_drugs:
            top_drugs = [d for d in top_drugs if d != "trametinib"]
            exclude = ["trametinib"]

        trace_event = make_trace(
            node="score_node", status="ok", started_at=t0,
            summary=f"{result.subtype} | top drug: {result.recommendations[0].drug} ({result.recommendations[0].score:.0%})",
            output_preview={
                "subtype": result.subtype,
                "braf_probability": round(result.braf_probability, 2),
                "top_drug": result.recommendations[0].drug,
                "top_score": round(result.recommendations[0].score, 3),
                "drugs_ranked": len(result.recommendations),
                "quarantined": len(result.quarantined),
            },
        )
        new_state = dict(state)
        new_state.update({
            "score_result": result,
            "top_drugs": top_drugs,
            "exclude_drugs": exclude,
            "trace": list(state.get("trace", [])) + [trace_event],
        })
        return TCCAgentState(**new_state)
    except Exception as e:
        trace_event = make_trace(
            node="score_node", status="error", started_at=t0,
            summary=f"Score failed: {e}",
            output_preview={}, error=str(e),
        )
        new_state = dict(state)
        new_state.update({
            "trace": list(state.get("trace", [])) + [trace_event],
            "errors": list(state.get("errors", [])) + [f"score_node: {e}"],
        })
        return TCCAgentState(**new_state)


# ---------------------------------------------------------------------------
# Node: chembl_node
# ---------------------------------------------------------------------------

def chembl_node(state: TCCAgentState) -> TCCAgentState:
    t0 = time.time()
    top_drugs = state.get("top_drugs", [])
    if not top_drugs:
        trace_event = make_trace(
            node="chembl_node", status="skipped", started_at=t0,
            summary="Skipped — no drugs to enrich", output_preview={},
        )
        new_state = dict(state)
        new_state["trace"] = list(state.get("trace", [])) + [trace_event]
        return TCCAgentState(**new_state)

    try:
        loop = asyncio.new_event_loop()
        compounds = loop.run_until_complete(enrich_panel(top_drugs))
        loop.close()
        achievable = [c.name for c in compounds if c.achievable]
        trace_event = make_trace(
            node="chembl_node", status="ok", started_at=t0,
            summary=f"Enriched {len(compounds)} compounds | {len(achievable)} achievable",
            output_preview={
                "compounds": [
                    {"name": c.name, "ic50_nm": c.ic50_nm, "gap_ratio": c.gap_ratio, "achievable": c.achievable}
                    for c in compounds
                ]
            },
        )
        new_state = dict(state)
        new_state.update({"chembl_data": compounds, "trace": list(state.get("trace", [])) + [trace_event]})
        return TCCAgentState(**new_state)
    except Exception as e:
        trace_event = make_trace(
            node="chembl_node", status="error", started_at=t0,
            summary=f"ChEMBL enrichment failed: {e}", output_preview={}, error=str(e),
        )
        new_state = dict(state)
        new_state.update({
            "trace": list(state.get("trace", [])) + [trace_event],
            "errors": list(state.get("errors", [])) + [f"chembl_node: {e}"],
        })
        return TCCAgentState(**new_state)


# ---------------------------------------------------------------------------
# Node: dosing_node
# ---------------------------------------------------------------------------

def dosing_node(state: TCCAgentState) -> TCCAgentState:
    t0 = time.time()
    top_drugs = state.get("top_drugs", [])
    exclude = state.get("exclude_drugs", [])
    drugs_to_dose = [d for d in top_drugs if d not in exclude]

    extra_trace = []
    for drug in exclude:
        extra_trace.append(make_trace(
            node=f"dosing_node:{drug}", status="skipped", started_at=t0,
            summary=f"{drug} skipped — BRAF negative (MEK inhibitor not indicated)",
            output_preview={"drug": drug, "reason": "BRAF negative"},
        ))

    try:
        doses = compute_full_panel_dosage(
            weight_kg=state.get("weight_kg", 10.0),
            creatinine_mg_dl=state.get("creatinine_mg_dl"),
            alt_u_l=state.get("alt_u_l"),
            drugs=drugs_to_dose,
        )
        held = [d.drug for d in doses if d.final_dose_mg == 0]
        trace_event = make_trace(
            node="dosing_node", status="ok", started_at=t0,
            summary=f"Dosed {len(doses)} drugs" + (f" | HOLD: {', '.join(held)}" if held else ""),
            output_preview={
                "doses": [{"drug": d.drug, "final_dose_mg": d.final_dose_mg, "schedule": d.schedule} for d in doses],
                "held_drugs": held,
            },
        )
        new_state = dict(state)
        new_state.update({
            "dosage_data": doses,
            "trace": list(state.get("trace", [])) + extra_trace + [trace_event],
        })
        return TCCAgentState(**new_state)
    except Exception as e:
        trace_event = make_trace(
            node="dosing_node", status="error", started_at=t0,
            summary=f"Dosing failed: {e}", output_preview={}, error=str(e),
        )
        new_state = dict(state)
        new_state.update({
            "trace": list(state.get("trace", [])) + extra_trace + [trace_event],
            "errors": list(state.get("errors", [])) + [f"dosing_node: {e}"],
        })
        return TCCAgentState(**new_state)


# ---------------------------------------------------------------------------
# Node: recipe_node
# ---------------------------------------------------------------------------

def recipe_node(state: TCCAgentState) -> TCCAgentState:
    t0 = time.time()
    doses = state.get("dosage_data", [])
    if not doses:
        trace_event = make_trace(
            node="recipe_node", status="skipped", started_at=t0,
            summary="Skipped — no dosage data", output_preview={},
        )
        new_state = dict(state)
        new_state["trace"] = list(state.get("trace", [])) + [trace_event]
        return TCCAgentState(**new_state)

    try:
        card = generate_recipe_card(
            pet_name=state.get("pet_name", "Patient"),
            species="Canis lupus familiaris",
            breed=state.get("breed", "unknown"),
            weight_kg=state.get("weight_kg", 10.0),
            prescribing_vet=state.get("prescribing_vet", "VetOnco System"),
            dose_results=doses,
        )
        trace_event = make_trace(
            node="recipe_node", status="ok", started_at=t0,
            summary=f"Recipe card: {len(card.drugs)} drugs | {len(card.interactions)} interactions",
            output_preview={
                "drugs": [d["drug"] for d in card.drugs],
                "interactions": card.interactions,
                "monitoring_items": len(card.monitoring),
            },
        )
        new_state = dict(state)
        new_state.update({"recipe_data": card, "trace": list(state.get("trace", [])) + [trace_event]})
        return TCCAgentState(**new_state)
    except Exception as e:
        trace_event = make_trace(
            node="recipe_node", status="error", started_at=t0,
            summary=f"Recipe failed: {e}", output_preview={}, error=str(e),
        )
        new_state = dict(state)
        new_state.update({
            "trace": list(state.get("trace", [])) + [trace_event],
            "errors": list(state.get("errors", [])) + [f"recipe_node: {e}"],
        })
        return TCCAgentState(**new_state)


# ---------------------------------------------------------------------------
# Node: llm_node
# ---------------------------------------------------------------------------

def llm_node(state: TCCAgentState) -> TCCAgentState:
    t0 = time.time()
    score_result = state.get("score_result")
    if not score_result:
        trace_event = make_trace(
            node="llm_node", status="skipped", started_at=t0,
            summary="Skipped — no score result", output_preview={},
        )
        new_state = dict(state)
        new_state["trace"] = list(state.get("trace", [])) + [trace_event]
        return TCCAgentState(**new_state)

    try:
        loop = asyncio.new_event_loop()

        rationale = loop.run_until_complete(generate_drug_rationale(
            recommendations=[asdict(r) for r in score_result.recommendations[:4]],
            braf_status=state.get("braf_status", "unknown"),
            breed=state.get("breed", "other"),
            subtype=score_result.subtype,
            pet_name=state.get("pet_name", "Patient"),
        ))

        pipeline_steps = [
            {"step": t.node, "status": t.status, "summary": t.summary}
            for t in state.get("trace", [])
        ]
        summary = loop.run_until_complete(generate_pipeline_summary(
            pipeline_steps=pipeline_steps,
            pet_name=state.get("pet_name", "Patient"),
            breed=state.get("breed", "other"),
            braf_status=state.get("braf_status", "unknown"),
        ))
        loop.close()

        trace_event = make_trace(
            node="llm_node", status="ok", started_at=t0,
            summary=f"LLM {'rationale + summary generated' if rationale else 'unavailable'}",
            output_preview={
                "rationale_available": rationale is not None,
                "summary_available": summary is not None,
                "rationale_preview": (rationale or "")[:120] if rationale else None,
            },
        )
        new_state = dict(state)
        new_state.update({
            "rationale_text": rationale,
            "pipeline_summary": summary,
            "trace": list(state.get("trace", [])) + [trace_event],
        })
        return TCCAgentState(**new_state)
    except Exception as e:
        trace_event = make_trace(
            node="llm_node", status="error", started_at=t0,
            summary=f"LLM node failed: {e}", output_preview={}, error=str(e),
        )
        new_state = dict(state)
        new_state.update({
            "trace": list(state.get("trace", [])) + [trace_event],
            "errors": list(state.get("errors", [])) + [f"llm_node: {e}"],
        })
        return TCCAgentState(**new_state)


# ---------------------------------------------------------------------------
# Node: error_node
# ---------------------------------------------------------------------------

def error_node(state: TCCAgentState) -> TCCAgentState:
    t0 = time.time()
    errors = state.get("errors", [])
    trace_event = make_trace(
        node="error_node", status="error", started_at=t0,
        summary=f"Pipeline halted: {errors[-1] if errors else 'unknown error'}",
        output_preview={"errors": errors},
        error=errors[-1] if errors else "unknown",
    )
    new_state = dict(state)
    new_state["trace"] = list(state.get("trace", [])) + [trace_event]
    return TCCAgentState(**new_state)


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def route_after_score(state: TCCAgentState) -> str:
    if state.get("errors"):
        return "error_node"
    return "chembl_node"


# ---------------------------------------------------------------------------
# Build the compiled graph (for validation + non-streaming use)
# ---------------------------------------------------------------------------

def build_tcc_graph():
    graph = StateGraph(TCCAgentState)
    graph.add_node("score_node", score_node)
    graph.add_node("chembl_node", chembl_node)
    graph.add_node("dosing_node", dosing_node)
    graph.add_node("recipe_node", recipe_node)
    graph.add_node("llm_node", llm_node)
    graph.add_node("error_node", error_node)
    graph.set_entry_point("score_node")
    graph.add_conditional_edges(
        "score_node",
        route_after_score,
        {"chembl_node": "chembl_node", "error_node": "error_node"},
    )
    graph.add_edge("chembl_node", "dosing_node")
    graph.add_edge("dosing_node", "recipe_node")
    graph.add_edge("recipe_node", "llm_node")
    graph.add_edge("llm_node", END)
    graph.add_edge("error_node", END)
    return graph.compile()


_tcc_graph = None

def get_tcc_graph():
    global _tcc_graph
    if _tcc_graph is None:
        _tcc_graph = build_tcc_graph()
    return _tcc_graph


# ---------------------------------------------------------------------------
# Manual sequential runner (used by SSE endpoint for real-time streaming)
# Runs nodes one-by-one so the router can yield SSE events between nodes.
# ---------------------------------------------------------------------------

NODE_SEQUENCE = [
    ("score_node", score_node),
    ("chembl_node", chembl_node),
    ("dosing_node", dosing_node),
    ("recipe_node", recipe_node),
    ("llm_node", llm_node),
]


def run_tcc_pipeline_sync(state: TCCAgentState) -> TCCAgentState:
    """Run all nodes sequentially. Returns final accumulated state."""
    for node_name, node_fn in NODE_SEQUENCE:
        state = node_fn(state)
        # Stop on error after score_node
        if node_name == "score_node" and state.get("errors"):
            state = error_node(state)
            break
    return state


def build_pipeline_result(final_state: TCCAgentState) -> dict:
    """Serialize the final pipeline state to a JSON-safe dict."""
    from dataclasses import asdict as dc_asdict

    score_result = final_state.get("score_result")
    recipe_data = final_state.get("recipe_data")
    chembl_data = final_state.get("chembl_data", [])
    dosage_data = final_state.get("dosage_data", [])

    return {
        "pet_name": final_state.get("pet_name"),
        "breed": final_state.get("breed"),
        "braf_status": final_state.get("braf_status"),
        "subtype": score_result.subtype if score_result else None,
        "braf_probability": score_result.braf_probability if score_result else None,
        "recommendations": [dc_asdict(r) for r in score_result.recommendations] if score_result else [],
        "quarantined": [dc_asdict(r) for r in score_result.quarantined] if score_result else [],
        "compounds": [dc_asdict(c) for c in chembl_data],
        "dosages": [dc_asdict(d) for d in dosage_data],
        "recipe": {
            "drugs": recipe_data.drugs,
            "interactions": recipe_data.interactions,
            "monitoring": recipe_data.monitoring,
            "printable_text": recipe_data.printable_text,
        } if recipe_data else None,
        "rationale": final_state.get("rationale_text"),
        "pipeline_summary": final_state.get("pipeline_summary"),
        "trace": [t.to_dict() for t in final_state.get("trace", [])],
        "errors": final_state.get("errors", []),
        "complete": len(final_state.get("errors", [])) == 0,
    }
