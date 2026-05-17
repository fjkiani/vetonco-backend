"""
VetOnco — LangGraph Agent Router
SSE streaming endpoint for TCC pipeline agent.
Monitoring agent endpoint.
All under /api/canine/agent
"""
from __future__ import annotations
import asyncio
import json
import time
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from auth import get_current_user_id
from agents.state import initial_tcc_state, initial_monitoring_state
from agents.tcc_pipeline_agent import (
    run_tcc_pipeline_sync, build_pipeline_result, get_tcc_graph,
    score_node, chembl_node, dosing_node, recipe_node, llm_node,
)
from agents.monitoring_agent import run_monitoring_agent

router = APIRouter(prefix="/api/canine/agent", tags=["agent"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class AgentRunRequest(BaseModel):
    pet_name: str
    breed: str
    weight_kg: float
    braf_status: str = "unknown"
    msh2_loss: bool = False
    expr: dict[str, float] = Field(default_factory=dict)
    creatinine_mg_dl: Optional[float] = None
    alt_u_l: Optional[float] = None
    prescribing_vet: str = "VetOnco System"


class MonitorRequest(BaseModel):
    pet_id: str
    pet_name: str
    breed: str
    weight_kg: float
    braf_status: str = "unknown"
    test_history: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def sse_event(event_type: str, data: dict) -> str:
    """Format a single SSE event."""
    payload = json.dumps({"event": event_type, **data})
    return f"data: {payload}\n\n"


async def run_pipeline_stream(req: AgentRunRequest) -> AsyncGenerator[str, None]:
    """
    Run the TCC pipeline graph node-by-node, yielding SSE events.
    
    Strategy: run each node manually in sequence so we can yield
    a trace event after each node completes (LangGraph stream() 
    yields full state updates, not per-node events in real time).
    """
    state = initial_tcc_state(
        pet_name=req.pet_name,
        breed=req.breed,
        weight_kg=req.weight_kg,
        braf_status=req.braf_status,
        msh2_loss=req.msh2_loss,
        expr=req.expr,
        creatinine_mg_dl=req.creatinine_mg_dl,
        alt_u_l=req.alt_u_l,
        prescribing_vet=req.prescribing_vet,
    )

    # Yield pipeline start event
    yield sse_event("pipeline_start", {
        "pet_name": req.pet_name,
        "breed": req.breed,
        "braf_status": req.braf_status,
        "nodes": ["score_node", "chembl_node", "dosing_node", "recipe_node", "llm_node"],
    })

    # Node sequence — run manually so we can stream after each
    node_sequence = [
        ("score_node", score_node),
        ("chembl_node", chembl_node),
        ("dosing_node", dosing_node),
        ("recipe_node", recipe_node),
        ("llm_node", llm_node),
    ]

    for node_name, node_fn in node_sequence:
        # Signal node starting
        yield sse_event("node_start", {"node": node_name, "started_at": time.time()})

        # Run node in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        try:
            state = await loop.run_in_executor(None, node_fn, state)
        except Exception as e:
            yield sse_event("node_error", {"node": node_name, "error": str(e)})
            break

        # Find the trace event this node just added
        trace = state.get("trace", [])
        if trace:
            latest = trace[-1]
            yield sse_event("node_complete", latest.to_dict())

        # Check for errors after score_node (conditional edge)
        if node_name == "score_node" and state.get("errors"):
            yield sse_event("pipeline_error", {
                "message": "Pipeline halted after score_node",
                "errors": state.get("errors", []),
            })
            break

        # Small yield to let the event loop breathe
        await asyncio.sleep(0)

    # Final complete event with full result
    result = build_pipeline_result(state)
    yield sse_event("complete", result)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/run")
async def agent_run(
    req: AgentRunRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Run the TCC pipeline LangGraph agent.
    Returns text/event-stream SSE.
    Each event: {"event": "node_complete"|"complete"|..., ...data}
    Final event: {"event": "complete", ...full_result}
    """
    return StreamingResponse(
        run_pipeline_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.post("/monitor")
async def agent_monitor(
    req: MonitorRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Run the monitoring LangGraph agent on a pet's test history.
    Returns JSON with alerts, trend analysis, and agent trace.
    """
    if len(req.test_history) < 1:
        return {
            "pet_id": req.pet_id,
            "alerts": [],
            "trend_analysis": {},
            "monitoring_summary": "Insufficient test history for trend analysis. Log at least 2 test sessions.",
            "trace": [],
            "has_critical": False,
        }

    state = initial_monitoring_state(
        pet_id=req.pet_id,
        pet_name=req.pet_name,
        breed=req.breed,
        weight_kg=req.weight_kg,
        braf_status=req.braf_status,
        test_history=req.test_history,
    )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_monitoring_agent, state)
    return result


@router.get("/health")
async def agent_health():
    """Check that LangGraph graphs compile successfully."""
    try:
        from agents.tcc_pipeline_agent import get_tcc_graph
        from agents.monitoring_agent import get_monitoring_graph
        tcc = get_tcc_graph()
        mon = get_monitoring_graph()
        return {
            "status": "ok",
            "tcc_graph": "compiled",
            "monitoring_graph": "compiled",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph compilation failed: {e}")
