"""
W3 — Dosing unit tests
Ground truth: BSA formula (Veterinary standard), published canine TCC dosing protocols.
All arithmetic is deterministic — exact comparisons.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from services.tcc_dosing import compute_canine_dose, compute_bsa, compute_full_panel_dosage


# ── A3-01: BSA formula — 12kg dog ────────────────────────────────────────────
def test_bsa_12kg():
    bsa = compute_bsa(12.0)
    assert bsa == 0.5298, f"Expected BSA=0.5298, got {bsa}"


# ── A3-02: BSA formula — 25kg dog ────────────────────────────────────────────
def test_bsa_25kg():
    bsa = compute_bsa(25.0)
    assert abs(bsa - 0.8636) < 0.001, f"Expected BSA≈0.8636, got {bsa}"


# ── A3-03: BSA formula — 8kg dog ─────────────────────────────────────────────
def test_bsa_8kg():
    bsa = compute_bsa(8.0)
    assert abs(bsa - 0.404) < 0.002, f"Expected BSA≈0.404, got {bsa}"


# ── A3-04: Toceranib 12kg ────────────────────────────────────────────────────
def test_toceranib_12kg():
    result = compute_canine_dose("toceranib", 12.0)
    assert result.dose_mg == 33.00, f"Expected 33.00mg, got {result.dose_mg}"
    assert result.final_dose_mg == 33.00
    assert result.schedule == "q48h"
    assert result.route == "PO"
    assert result.renal_adjustment == "none"
    assert result.hepatic_adjustment == "none"


# ── A3-05: Toceranib 25kg ────────────────────────────────────────────────────
def test_toceranib_25kg():
    result = compute_canine_dose("toceranib", 25.0)
    assert result.dose_mg == 68.75, f"Expected 68.75mg, got {result.dose_mg}"
    assert result.final_dose_mg == 68.75


# ── A3-06: Toceranib 8kg ─────────────────────────────────────────────────────
def test_toceranib_8kg():
    result = compute_canine_dose("toceranib", 8.0)
    assert result.dose_mg == 22.00, f"Expected 22.00mg, got {result.dose_mg}"


# ── A3-07: Carboplatin 12kg — BSA-based ──────────────────────────────────────
def test_carboplatin_12kg_no_adjustment():
    result = compute_canine_dose("carboplatin", 12.0)
    assert result.dose_mg == 158.94, f"Expected 158.94mg, got {result.dose_mg}"
    assert result.final_dose_mg == 158.94
    assert result.schedule == "q21d"
    assert result.route == "IV"


# ── A3-08: Carboplatin 12kg, Cr=1.8 → 25% reduction ─────────────────────────
def test_carboplatin_renal_25pct():
    result = compute_canine_dose("carboplatin", 12.0, creatinine_mg_dl=1.8)
    assert result.renal_adjustment == "25%", f"Expected 25%, got {result.renal_adjustment}"
    # 158.94 × 0.75 = 119.205 → rounded to 2dp
    assert abs(result.final_dose_mg - 119.21) < 0.01, f"Expected ~119.21mg, got {result.final_dose_mg}"
    assert len(result.warnings) > 0


# ── A3-09: Carboplatin Cr=2.2 → 50% reduction ────────────────────────────────
def test_carboplatin_renal_50pct():
    result = compute_canine_dose("carboplatin", 12.0, creatinine_mg_dl=2.2)
    assert result.renal_adjustment == "50%", f"Expected 50%, got {result.renal_adjustment}"
    assert abs(result.final_dose_mg - 79.47) < 0.01, f"Expected ~79.47mg, got {result.final_dose_mg}"


# ── A3-10: Carboplatin Cr=3.5 → HOLD ─────────────────────────────────────────
def test_carboplatin_hold():
    result = compute_canine_dose("carboplatin", 12.0, creatinine_mg_dl=3.5)
    assert result.renal_adjustment == "hold", f"Expected hold, got {result.renal_adjustment}"
    assert result.final_dose_mg == 0.0, f"Expected 0.0 (HOLD), got {result.final_dose_mg}"
    assert any("HOLD" in w for w in result.warnings), "Expected HOLD warning"


# ── A3-11: Toceranib ALT=450 → hepatic HOLD ──────────────────────────────────
def test_toceranib_hepatic_hold():
    result = compute_canine_dose("toceranib", 12.0, alt_u_l=450)
    assert result.hepatic_adjustment == "hold", f"Expected hold, got {result.hepatic_adjustment}"
    assert result.final_dose_mg == 0.0


# ── A3-12: Toceranib ALT=250 → 25% reduction ─────────────────────────────────
def test_toceranib_hepatic_25pct():
    result = compute_canine_dose("toceranib", 12.0, alt_u_l=250)
    assert result.hepatic_adjustment == "25%", f"Expected 25%, got {result.hepatic_adjustment}"
    assert abs(result.final_dose_mg - 24.75) < 0.01, f"Expected ~24.75mg, got {result.final_dose_mg}"


# ── A3-13: Unknown drug returns error gracefully ─────────────────────────────
def test_unknown_drug():
    result = compute_canine_dose("imatinib", 12.0)
    assert result.final_dose_mg == 0.0
    assert len(result.warnings) > 0


# ── A3-14: Full panel returns only PASS drugs ─────────────────────────────────
def test_full_panel_pass_only():
    results = compute_full_panel_dosage(12.0)
    drug_names = [r.drug for r in results]
    assert "toceranib" in drug_names
    quarantined = {"mitoxantrone", "piroxicam", "vinblastine", "carboplatin", "gemcitabine", "trametinib"}
    for q in quarantined:
        assert q not in drug_names, f"Quarantined drug {q} appeared in full panel"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
