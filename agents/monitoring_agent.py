"""
VetOnco — Monitoring LangGraph Agent
Runs on a pet's full test history. Detects clinical trends. Returns structured alerts.

Graph flow:
  load_history_node → trend_node → alert_node → llm_summary_node → END
"""
from __future__ import annotations
import asyncio
import time
from typing import Any

from langgraph.graph import StateGraph, END

from agents.state import (
    MonitoringAgentState, MonitoringAlert, TraceEvent, make_trace,
    initial_monitoring_state,
)
from services.llm_service import _chat


# ---------------------------------------------------------------------------
# Trend thresholds (VCOG-CTCAE informed)
# ---------------------------------------------------------------------------

# (parameter_key, test_type, worsening_direction, critical_threshold, high_threshold)
TREND_CONFIG = [
    # CBC
    ("anc",         "cbc",       "decreasing", 0.5,  1.0),   # ANC ×10³/µL
    ("platelets",   "cbc",       "decreasing", 25,   50),    # ×10³/µL
    ("hematocrit",  "cbc",       "decreasing", 14,   20),    # %
    # Chemistry
    ("creatinine",  "chemistry", "increasing", 5.0,  3.0),   # mg/dL
    ("alt",         "chemistry", "increasing", 1000, 500),   # U/L
    ("bun",         "chemistry", "increasing", 100,  60),    # mg/dL
    # Imaging
    ("tumor_size_cm", "imaging", "increasing", None, None),  # any increase = worsening
]

DRUG_IMPLICATED_MAP = {
    "anc":        "toceranib / cytotoxics",
    "platelets":  "cytotoxics",
    "hematocrit": "cytotoxics",
    "creatinine": "carboplatin / piroxicam",
    "alt":        "mitoxantrone / vinblastine / trametinib",
    "bun":        "carboplatin",
}


# ---------------------------------------------------------------------------
# Node: load_history_node
# ---------------------------------------------------------------------------

def load_history_node(state: MonitoringAgentState) -> MonitoringAgentState:
    t0 = time.time()
    history = state.get("test_history", [])

    # Group by test_type
    grouped: dict[str, list[dict]] = {}
    for entry in history:
        tt = entry.get("test_type", "unknown")
        if tt not in grouped:
            grouped[tt] = []
        grouped[tt].append(entry)

    # Sort each group by timestamp (use index as proxy if no timestamp)
    for tt in grouped:
        grouped[tt] = sorted(grouped[tt], key=lambda x: x.get("logged_at", 0))

    trace_event = make_trace(
        node="load_history_node", status="ok", started_at=t0,
        summary=f"Loaded {len(history)} test sessions across {len(grouped)} test types",
        output_preview={
            "total_sessions": len(history),
            "test_types": {tt: len(v) for tt, v in grouped.items()},
        },
    )
    return {
        **state,
        "grouped_history": grouped,
        "trace": state.get("trace", []) + [trace_event],
    }


# ---------------------------------------------------------------------------
# Node: trend_node
# ---------------------------------------------------------------------------

def _compute_slope(values: list[float]) -> float:
    """Simple linear slope over the last N readings (least squares)."""
    n = len(values)
    if n < 2:
        return 0.0
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(values) / n
    num = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    den = sum((x[i] - x_mean) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


def _extract_values(sessions: list[dict], param: str) -> list[float]:
    """Extract numeric values for a parameter from a list of test sessions."""
    result = []
    for s in sessions:
        v = s.get("values", {}).get(param)
        if v is not None:
            try:
                result.append(float(v))
            except (TypeError, ValueError):
                pass
    return result


def trend_node(state: MonitoringAgentState) -> MonitoringAgentState:
    t0 = time.time()
    grouped = state.get("grouped_history", {})
    trend_analysis: dict[str, dict] = {}

    for param, test_type, direction, crit_thresh, high_thresh in TREND_CONFIG:
        sessions = grouped.get(test_type, [])
        values = _extract_values(sessions, param)

        if len(values) < 2:
            continue  # Need at least 2 readings for a trend

        slope = _compute_slope(values[-5:])  # Use last 5 readings max
        latest = values[-1]

        # Determine trend direction
        if direction == "decreasing":
            # Worsening = slope is negative (value going down)
            if slope < -0.05:
                trend = "worsening"
            elif slope > 0.05:
                trend = "improving"
            else:
                trend = "stable"
        else:
            # Worsening = slope is positive (value going up)
            if slope > 0.05:
                trend = "worsening"
            elif slope < -0.05:
                trend = "improving"
            else:
                trend = "stable"

        # Severity based on latest value vs thresholds
        if crit_thresh is not None:
            if direction == "decreasing":
                severity = "CRITICAL" if latest <= crit_thresh else ("HIGH" if latest <= high_thresh else "MODERATE")
            else:
                severity = "CRITICAL" if latest >= crit_thresh else ("HIGH" if latest >= high_thresh else "MODERATE")
        else:
            severity = "HIGH" if trend == "worsening" else "MODERATE"

        trend_analysis[param] = {
            "test_type": test_type,
            "values": values,
            "slope": round(slope, 4),
            "trend": trend,
            "latest": latest,
            "severity": severity,
            "n_readings": len(values),
        }

    worsening = [p for p, d in trend_analysis.items() if d["trend"] == "worsening"]
    trace_event = make_trace(
        node="trend_node", status="ok", started_at=t0,
        summary=f"Analyzed {len(trend_analysis)} parameters | {len(worsening)} worsening",
        output_preview={
            "parameters_analyzed": len(trend_analysis),
            "worsening": worsening,
            "trends": {p: d["trend"] for p, d in trend_analysis.items()},
        },
    )
    return {
        **state,
        "trend_analysis": trend_analysis,
        "trace": state.get("trace", []) + [trace_event],
    }


# ---------------------------------------------------------------------------
# Node: alert_node
# ---------------------------------------------------------------------------

def alert_node(state: MonitoringAgentState) -> MonitoringAgentState:
    t0 = time.time()
    trend_analysis = state.get("trend_analysis", {})
    alerts: list[MonitoringAlert] = []

    PARAM_LABELS = {
        "anc": "ANC (neutrophils)",
        "platelets": "Platelets",
        "hematocrit": "Hematocrit",
        "creatinine": "Creatinine",
        "alt": "ALT (liver enzyme)",
        "bun": "BUN",
        "tumor_size_cm": "Tumor size",
    }

    ACTION_MAP = {
        ("anc", "worsening", "CRITICAL"): "EMERGENCY — hospitalize; IV antibiotics; hold all cytotoxics",
        ("anc", "worsening", "HIGH"): "HOLD toceranib and cytotoxics; recheck CBC in 3-5 days",
        ("anc", "worsening", "MODERATE"): "Monitor closely; consider dose reduction",
        ("creatinine", "worsening", "CRITICAL"): "HOLD carboplatin and piroxicam; IV fluids; nephrology consult",
        ("creatinine", "worsening", "HIGH"): "Reduce nephrotoxic drugs 25%; recheck in 1 week",
        ("alt", "worsening", "CRITICAL"): "HOLD mitoxantrone/vinblastine/trametinib; hepatology consult",
        ("alt", "worsening", "HIGH"): "Reduce hepatically-cleared drugs 25%; recheck in 2 weeks",
        ("tumor_size_cm", "worsening", "HIGH"): "Progressive disease — consider protocol switch to second-line agents",
        ("platelets", "worsening", "CRITICAL"): "HOLD all cytotoxics; bleeding precautions; transfusion consider",
        ("platelets", "worsening", "HIGH"): "Delay cytotoxic therapy; recheck in 5-7 days",
    }

    for param, data in trend_analysis.items():
        trend = data["trend"]
        severity = data["severity"]

        if trend == "stable" and severity == "MODERATE":
            continue  # Don't alert on stable moderate findings

        label = PARAM_LABELS.get(param, param)
        latest = data["latest"]
        n = data["n_readings"]

        if trend == "worsening":
            message = f"{label} trending down over {n} readings (latest: {latest})"
        elif trend == "improving":
            message = f"{label} improving over {n} readings (latest: {latest})"
        else:
            message = f"{label} stable at {latest} over {n} readings"

        action = ACTION_MAP.get(
            (param, trend, severity),
            f"Monitor {label} closely; consult veterinary oncologist"
        )

        alerts.append(MonitoringAlert(
            severity=severity if trend == "worsening" else "LOW",
            parameter=param,
            trend=trend,
            message=message,
            action=action,
            drug_implicated=DRUG_IMPLICATED_MAP.get(param) if trend == "worsening" else None,
        ))

    # Sort by severity
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}
    alerts.sort(key=lambda a: severity_order.get(a.severity, 4))

    critical_count = sum(1 for a in alerts if a.severity == "CRITICAL")
    high_count = sum(1 for a in alerts if a.severity == "HIGH")

    trace_event = make_trace(
        node="alert_node", status="ok", started_at=t0,
        summary=f"Generated {len(alerts)} alerts | {critical_count} CRITICAL | {high_count} HIGH",
        output_preview={
            "total_alerts": len(alerts),
            "critical": critical_count,
            "high": high_count,
            "alert_parameters": [a.parameter for a in alerts if a.severity in ("CRITICAL", "HIGH")],
        },
    )
    return {
        **state,
        "alerts": alerts,
        "trace": state.get("trace", []) + [trace_event],
    }


# ---------------------------------------------------------------------------
# Node: llm_summary_node
# ---------------------------------------------------------------------------

def llm_summary_node(state: MonitoringAgentState) -> MonitoringAgentState:
    t0 = time.time()
    alerts = state.get("alerts", [])
    trend_analysis = state.get("trend_analysis", {})

    system = (
        "You are a veterinary oncology clinical assistant. "
        "Write a concise monitoring summary (under 150 words) for a dog with TCC/UC. "
        "Cover: overall trend direction, most concerning findings, and immediate next steps. "
        "Be specific and clinical. Address the veterinarian directly."
    )

    alerts_text = "\n".join(
        f"- {a.severity} | {a.parameter}: {a.message} | Action: {a.action}"
        for a in alerts[:6]
    )
    trends_text = "\n".join(
        f"- {p}: {d['trend']} (latest {d['latest']}, slope {d['slope']:+.3f})"
        for p, d in trend_analysis.items()
    )

    user = (
        f"Patient: {state.get('pet_name')} ({state.get('breed')})\n"
        f"BRAF status: {state.get('braf_status')}\n\n"
        f"Trend analysis:\n{trends_text or 'Insufficient data for trends'}\n\n"
        f"Alerts:\n{alerts_text or 'No significant alerts'}\n\n"
        "Please write the monitoring summary."
    )

    try:
        loop = asyncio.get_event_loop()
        summary = loop.run_until_complete(_chat(system, user))
        available = summary is not None
        trace_event = make_trace(
            node="llm_summary_node", status="ok", started_at=t0,
            summary=f"Monitoring summary {'generated' if available else 'unavailable'}",
            output_preview={"available": available, "preview": (summary or "")[:100]},
        )
        return {
            **state,
            "monitoring_summary": summary,
            "trace": state.get("trace", []) + [trace_event],
        }
    except Exception as e:
        trace_event = make_trace(
            node="llm_summary_node", status="error", started_at=t0,
            summary=f"LLM summary failed: {e}",
            output_preview={}, error=str(e),
        )
        return {
            **state,
            "trace": state.get("trace", []) + [trace_event],
        }


# ---------------------------------------------------------------------------
# Build the monitoring graph
# ---------------------------------------------------------------------------

def build_monitoring_graph() -> StateGraph:
    graph = StateGraph(MonitoringAgentState)

    graph.add_node("load_history_node", load_history_node)
    graph.add_node("trend_node", trend_node)
    graph.add_node("alert_node", alert_node)
    graph.add_node("llm_summary_node", llm_summary_node)

    graph.set_entry_point("load_history_node")
    graph.add_edge("load_history_node", "trend_node")
    graph.add_edge("trend_node", "alert_node")
    graph.add_edge("alert_node", "llm_summary_node")
    graph.add_edge("llm_summary_node", END)

    return graph.compile()


_monitoring_graph = None

def get_monitoring_graph():
    global _monitoring_graph
    if _monitoring_graph is None:
        _monitoring_graph = build_monitoring_graph()
    return _monitoring_graph


def run_monitoring_agent(state: MonitoringAgentState) -> dict:
    """Run the monitoring agent and return serializable result."""
    graph = get_monitoring_graph()
    final_state = None
    for chunk in graph.stream(state):
        for _, node_state in chunk.items():
            final_state = node_state
    final_state = final_state or state

    return {
        "pet_id": final_state.get("pet_id"),
        "pet_name": final_state.get("pet_name"),
        "trend_analysis": final_state.get("trend_analysis", {}),
        "alerts": [a.to_dict() for a in final_state.get("alerts", [])],
        "monitoring_summary": final_state.get("monitoring_summary"),
        "trace": [t.to_dict() for t in final_state.get("trace", [])],
        "has_critical": any(
            a.severity == "CRITICAL" for a in final_state.get("alerts", [])
        ),
    }
