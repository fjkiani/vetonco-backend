"""
VetOnco — Shared LangGraph Agent State Types
TraceEvent, TCCAgentState (TypedDict), MonitoringAgentState (TypedDict), MonitoringAlert
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, TypedDict


# ---------------------------------------------------------------------------
# Trace event — emitted by every node, streamed to frontend via SSE
# ---------------------------------------------------------------------------

@dataclass
class TraceEvent:
    node: str
    status: Literal["running", "ok", "error", "skipped"]
    started_at: float
    finished_at: float
    duration_ms: int
    summary: str
    output_preview: dict[str, Any]
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "node": self.node,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "summary": self.summary,
            "output_preview": self.output_preview,
            "error": self.error,
        }


def make_trace(
    node: str,
    status: str,
    started_at: float,
    summary: str,
    output_preview: dict,
    error: str | None = None,
) -> TraceEvent:
    finished = time.time()
    return TraceEvent(
        node=node,
        status=status,
        started_at=started_at,
        finished_at=finished,
        duration_ms=int((finished - started_at) * 1000),
        summary=summary,
        output_preview=output_preview,
        error=error,
    )


# ---------------------------------------------------------------------------
# TCC Pipeline Agent State — TypedDict for LangGraph compatibility
# ---------------------------------------------------------------------------

class TCCAgentState(TypedDict, total=False):
    # Inputs
    pet_name: str
    breed: str
    weight_kg: float
    braf_status: str
    msh2_loss: bool
    expr: dict
    creatinine_mg_dl: Optional[float]
    alt_u_l: Optional[float]
    prescribing_vet: str
    # Computed
    top_drugs: list
    exclude_drugs: list
    # Outputs
    score_result: Any
    chembl_data: list
    dosage_data: list
    recipe_data: Any
    rationale_text: Optional[str]
    pipeline_summary: Optional[str]
    # Trace
    trace: list
    errors: list


def initial_tcc_state(
    pet_name: str,
    breed: str,
    weight_kg: float,
    braf_status: str,
    msh2_loss: bool = False,
    expr: dict | None = None,
    creatinine_mg_dl: float | None = None,
    alt_u_l: float | None = None,
    prescribing_vet: str = "VetOnco System",
) -> TCCAgentState:
    return TCCAgentState(
        pet_name=pet_name,
        breed=breed,
        weight_kg=weight_kg,
        braf_status=braf_status,
        msh2_loss=msh2_loss,
        expr=expr or {},
        creatinine_mg_dl=creatinine_mg_dl,
        alt_u_l=alt_u_l,
        prescribing_vet=prescribing_vet,
        top_drugs=[],
        exclude_drugs=[],
        score_result=None,
        chembl_data=[],
        dosage_data=[],
        recipe_data=None,
        rationale_text=None,
        pipeline_summary=None,
        trace=[],
        errors=[],
    )


# ---------------------------------------------------------------------------
# Monitoring Agent State — TypedDict
# ---------------------------------------------------------------------------

class MonitoringAgentState(TypedDict, total=False):
    pet_id: str
    pet_name: str
    breed: str
    weight_kg: float
    braf_status: str
    test_history: list
    grouped_history: dict
    trend_analysis: dict
    alerts: list
    monitoring_summary: Optional[str]
    trace: list


def initial_monitoring_state(
    pet_id: str,
    pet_name: str,
    breed: str,
    weight_kg: float,
    braf_status: str,
    test_history: list[dict],
) -> MonitoringAgentState:
    return MonitoringAgentState(
        pet_id=pet_id,
        pet_name=pet_name,
        breed=breed,
        weight_kg=weight_kg,
        braf_status=braf_status,
        test_history=test_history,
        grouped_history={},
        trend_analysis={},
        alerts=[],
        monitoring_summary=None,
        trace=[],
    )


# ---------------------------------------------------------------------------
# MonitoringAlert
# ---------------------------------------------------------------------------

@dataclass
class MonitoringAlert:
    severity: Literal["CRITICAL", "HIGH", "MODERATE", "LOW"]
    parameter: str
    trend: Literal["worsening", "improving", "stable"]
    message: str
    action: str
    drug_implicated: Optional[str] = None
    alert_id: str = field(default_factory=lambda: f"alert_{int(time.time()*1000)}")

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity,
            "parameter": self.parameter,
            "trend": self.trend,
            "message": self.message,
            "action": self.action,
            "drug_implicated": self.drug_implicated,
        }
