"""
W1 — Scorer unit tests
Ground truth anchors from VETONCO_GROUND_TRUTH_BENCHMARKS.mdc
All assertions are exact numeric comparisons — no tolerance bands.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from services.tcc_scorer import score_tcc, build_score_rationale
from services.tcc_gene_panels import PASS_DRUGS, QUARANTINE_DRUGS


# ── A1-01: BRAF+ Scottish Terrier → toceranib rank 1, score 0.380 ──────────
def test_braf_positive_scottish_terrier():
    result = score_tcc(braf_status="positive", breed="scottish terrier")
    assert len(result.recommendations) == 1, f"Expected 1 PASS drug, got {len(result.recommendations)}"
    top = result.recommendations[0]
    assert top.drug == "toceranib", f"Expected toceranib rank 1, got {top.drug}"
    assert top.rank == 1
    assert top.score == 0.380, f"Expected score=0.380, got {top.score}"
    assert top.orr == 0.26
    assert top.pk_status == "VERIFIED"
    assert top.feasibility_verdict == "PASS"
    assert top.gap_ratio == 0.6494, f"Expected gap_ratio=0.6494 (0.010/0.0154), got {top.gap_ratio}"


# ── A1-02: BRAF unknown → toceranib score 0.310 ─────────────────────────────
def test_braf_unknown():
    result = score_tcc(braf_status="unknown", breed="other")
    top = result.recommendations[0]
    assert top.drug == "toceranib"
    assert top.score == 0.310, f"Expected 0.310, got {top.score}"


# ── A1-03: BRAF negative → toceranib score 0.260 ────────────────────────────
def test_braf_negative():
    result = score_tcc(braf_status="negative", breed="other")
    top = result.recommendations[0]
    assert top.drug == "toceranib"
    assert top.score == 0.260, f"Expected 0.260, got {top.score}"


# ── A1-04: Quarantine count = 6 ─────────────────────────────────────────────
def test_quarantine_count():
    result = score_tcc(braf_status="unknown")
    assert len(result.quarantined) == 6, f"Expected 6 quarantined, got {len(result.quarantined)}"
    quarantine_names = {r.drug for r in result.quarantined}
    expected = {"mitoxantrone", "piroxicam", "vinblastine", "carboplatin", "gemcitabine", "trametinib"}
    assert quarantine_names == expected, f"Quarantine mismatch: {quarantine_names}"


# ── A1-05: PASS count = 1 ────────────────────────────────────────────────────
def test_pass_count():
    result = score_tcc(braf_status="positive")
    assert len(result.recommendations) == 1
    assert result.recommendations[0].drug == "toceranib"


# ── A1-06: Rationale is hard-coded template, not natural language ────────────
def test_rationale_format():
    result = score_tcc(braf_status="positive", breed="scottish terrier")
    rationale = result.recommendations[0].rationale
    assert "score=" in rationale, f"Missing 'score=' in rationale: {rationale}"
    assert "orr_base=" in rationale, f"Missing 'orr_base=' in rationale: {rationale}"
    assert "braf_delta=" in rationale, f"Missing 'braf_delta=' in rationale: {rationale}"
    bad_phrases = ["I recommend", "This drug", "The patient", "Based on", "Therefore"]
    for phrase in bad_phrases:
        assert phrase not in rationale, f"LLM-style phrase found: '{phrase}'"


# ── A1-07: BRAF probability for Scottish Terrier ────────────────────────────
def test_braf_probability_scottish_terrier():
    result = score_tcc(braf_status="positive", breed="scottish terrier")
    assert result.braf_probability == 0.90, f"Expected 0.90, got {result.braf_probability}"


# ── A1-08: Subtype classification ────────────────────────────────────────────
def test_subtype_braf_positive():
    result = score_tcc(braf_status="positive")
    assert result.subtype == "BRAF-mutant TCC"

def test_subtype_msh2_loss():
    result = score_tcc(braf_status="negative", msh2_loss=True)
    assert result.subtype == "MMR-deficient TCC"

def test_subtype_nos():
    result = score_tcc(braf_status="unknown", msh2_loss=False)
    assert result.subtype == "TCC-NOS"


# ── A1-09: All quarantined drugs have quarantine_reason ─────────────────────
def test_quarantine_reasons_populated():
    result = score_tcc()
    for q in result.quarantined:
        assert q.quarantine_reason is not None, f"{q.drug} has no quarantine_reason"
        assert len(q.quarantine_reason) > 10, f"{q.drug} quarantine_reason too short"


# ── A1-10: Score is clipped to [0, 1] ────────────────────────────────────────
def test_score_clipping():
    high_expr = {t: 4.0 for t in ["VEGFR2", "PDGFR", "KIT", "BRAF"]}
    result = score_tcc(expr=high_expr, braf_status="positive")
    for rec in result.recommendations:
        assert 0.0 <= rec.score <= 1.0, f"Score out of bounds: {rec.score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
