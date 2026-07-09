"""
W2 — Feasibility gate unit tests
Ground truth: Henry et al. 2009 (PMID 19185954) — primary canine toceranib PK
All arithmetic verified against published values.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from services.feasibility_gate import compute_feasibility, _build_rationale


TOCERANIB_PK = {
    "pk_status": "VERIFIED",
    "quarantine_reason": None,
    "feasibility_gate": {
        "ic50_um": 0.010,
        "cmax_um": 0.220,
        "plasma_protein_binding": 0.93,
    },
}


# ── A2-01: Toceranib PASS — Henry 2009 ground truth ─────────────────────────
def test_toceranib_pass():
    result = compute_feasibility(TOCERANIB_PK)
    assert result["verdict"] == "PASS", f"Expected PASS, got {result['verdict']}"
    # gap = 0.010 / 0.0154 = 0.6494 (exact; 0.65 was a rounded approximation)
    assert result["gap_ratio"] == 0.6494, f"Expected gap=0.6494, got {result['gap_ratio']}"
    assert result["free_cmax"] == 0.0154, f"Expected free_cmax=0.0154, got {result['free_cmax']}"


# ── A2-02: Coverage ratio = 1.54× (NOT 25.7×) ───────────────────────────────
def test_toceranib_coverage_ratio():
    result = compute_feasibility(TOCERANIB_PK)
    rationale = result["rationale"]
    assert "1.54×" in rationale, f"Expected '1.54×' in rationale, got: {rationale}"
    assert "25.7×" not in rationale, f"Hallucinated '25.7×' found in rationale: {rationale}"


# ── A2-03: UNVERIFIED short-circuit ─────────────────────────────────────────
def test_unverified_short_circuit():
    result = compute_feasibility({
        "pk_status": "UNVERIFIED",
        "quarantine_reason": "No primary canine PK study",
        "feasibility_gate": {"ic50_um": 0.1, "cmax_um": 0.5, "plasma_protein_binding": 0.9},
    })
    assert result["verdict"] == "UNVERIFIED"
    assert result["gap_ratio"] is None
    assert result["free_cmax"] is None


# ── A2-04: Band PASS (gap < 5×) ──────────────────────────────────────────────
def test_band_pass():
    result = compute_feasibility({
        "pk_status": "VERIFIED",
        "quarantine_reason": None,
        "feasibility_gate": {"ic50_um": 0.10, "cmax_um": 0.20, "plasma_protein_binding": 0.0},
    })
    assert result["verdict"] == "PASS"
    assert result["gap_ratio"] == 0.5


# ── A2-05: Band CONDITIONAL (5× ≤ gap < 50×) ────────────────────────────────
def test_band_conditional():
    result = compute_feasibility({
        "pk_status": "VERIFIED",
        "quarantine_reason": None,
        "feasibility_gate": {"ic50_um": 1.0, "cmax_um": 0.20, "plasma_protein_binding": 0.0},
    })
    assert result["verdict"] == "CONDITIONAL", f"Expected CONDITIONAL, got {result['verdict']}"
    assert result["gap_ratio"] == 5.0


# ── A2-06: Band FAIL (gap ≥ 50×) ─────────────────────────────────────────────
def test_band_fail():
    result = compute_feasibility({
        "pk_status": "VERIFIED",
        "quarantine_reason": None,
        "feasibility_gate": {"ic50_um": 10.0, "cmax_um": 0.10, "plasma_protein_binding": 0.0},
    })
    assert result["verdict"] == "FAIL"
    assert result["gap_ratio"] == 100.0


# ── A2-07: Exact boundary at 5× → CONDITIONAL ───────────────────────────────
def test_band_boundary_exactly_5():
    result = compute_feasibility({
        "pk_status": "VERIFIED",
        "quarantine_reason": None,
        "feasibility_gate": {"ic50_um": 5.0, "cmax_um": 1.0, "plasma_protein_binding": 0.0},
    })
    assert result["verdict"] == "CONDITIONAL", f"gap=5.0 should be CONDITIONAL, got {result['verdict']}"


# ── A2-08: Exact boundary at 50× → FAIL ─────────────────────────────────────
def test_band_boundary_exactly_50():
    result = compute_feasibility({
        "pk_status": "VERIFIED",
        "quarantine_reason": None,
        "feasibility_gate": {"ic50_um": 50.0, "cmax_um": 1.0, "plasma_protein_binding": 0.0},
    })
    assert result["verdict"] == "FAIL", f"gap=50.0 should be FAIL, got {result['verdict']}"


# ── A2-09: Missing PK values → UNVERIFIED ────────────────────────────────────
def test_missing_pk_values():
    result = compute_feasibility({
        "pk_status": "VERIFIED",
        "quarantine_reason": None,
        "feasibility_gate": {"ic50_um": None, "cmax_um": 0.22, "plasma_protein_binding": 0.93},
    })
    assert result["verdict"] == "UNVERIFIED"


# ── A2-10: Rationale template keys ───────────────────────────────────────────
def test_rationale_template_keys():
    result = compute_feasibility(TOCERANIB_PK)
    rationale = result["rationale"]
    for key in ["PASS", "ic50=", "cmax=", "ppb=", "free_cmax=", "gap=", "free_cmax/ic50="]:
        assert key in rationale, f"Missing '{key}' in rationale: {rationale}"


# ── A2-11: _build_rationale arithmetic ───────────────────────────────────────
def test_build_rationale_arithmetic():
    r = _build_rationale(0.010, 0.220, 0.93, 0.0154, 0.65, "PASS")
    assert "0.0154" in r
    assert "0.65" in r
    assert "1.54" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
