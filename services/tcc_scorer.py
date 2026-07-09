"""
VetOnco — TCC Drug Scorer v2 (ORR-Anchored)
score_tcc(expr, braf_status, breed, msh2_loss) → TCCResult

FORMULA (v2):
    final_score = clip(orr_base × confidence + braf_delta + target_delta, 0, 1)

    orr_base    = published canine TCC ORR (0.0 if no data)
    confidence  = evidence tier weight (A=1.0, B=0.7, C=0.4)
    braf_delta  = patient-specific BRAF adjustment (from DrugEntry)
    target_delta = expression-weighted target score (range ±0.10)

RATIONALE STRINGS: All rationale strings are computed from arithmetic.
    No LLM text generation for any number in the rationale.
    Template: build_score_rationale() — deterministic, no natural language.

QUARANTINE POLICY:
    Only drugs with pk_status="VERIFIED" (all 3 PK values from primary canine study)
    are eligible for PASS. All others are QUARANTINE regardless of score.
    At launch: only toceranib is VERIFIED.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

from services.tcc_gene_panels import (
    DRUG_PANEL, PASS_DRUGS, QUARANTINE_DRUGS,
    TCC_PANEL_GENES_TIER_A, TCC_PANEL_GENES_TIER_B, TCC_PANEL_GENES_TIER_C,
    DrugEntry, get_braf_prior,
)
from services.feasibility_gate import compute_feasibility

BRAFStatus = Literal["positive", "negative", "unknown"]

TIER_WEIGHT = {"A": 1.0, "B": 0.7, "C": 0.4}


@dataclass
class DrugRecommendation:
    drug: str
    score: float
    rank: int
    mechanism: str
    targets: list[str]
    braf_requirement: str
    verdict: str
    quarantined: bool
    rationale: str
    evidence_tier: Literal["A", "B", "C"]
    pk_status: str = "UNVERIFIED"
    feasibility_verdict: str | None = None
    gap_ratio: float | None = None
    orr: float | None = None
    orr_source: str | None = None
    quarantine_reason: str | None = None


@dataclass
class TCCResult:
    subtype: str
    braf_status: BRAFStatus
    braf_probability: float
    msh2_loss: bool
    top_altered_genes: list[str]
    recommendations: list[DrugRecommendation]
    quarantined: list[DrugRecommendation]
    summary: str


# ---------------------------------------------------------------------------
# Evidence tier map (A/B/C scoring tier)
# ---------------------------------------------------------------------------

EVIDENCE_TIER_MAP: dict[str, Literal["A", "B", "C"]] = {
    "piroxicam":    "A",
    "toceranib":    "A",
    "mitoxantrone": "A",
    "vinblastine":  "B",
    "carboplatin":  "B",
    "gemcitabine":  "B",
    "trametinib":   "C",
}


def _gene_tier(gene: str) -> str:
    if gene in TCC_PANEL_GENES_TIER_A:
        return "A"
    if gene in TCC_PANEL_GENES_TIER_B:
        return "B"
    return "C"


def _target_delta(drug: DrugEntry, expr: dict[str, float]) -> float:
    """Expression-weighted target score. Range: approximately ±0.10."""
    if not expr or not drug.targets:
        return 0.0
    scores = []
    for t in drug.targets:
        if t in expr:
            val = expr[t]
            tier_w = TIER_WEIGHT.get(_gene_tier(t), 0.4)
            scores.append(min(max(val / 4.0, -1.0), 1.0) * tier_w * 0.1)
    return sum(scores) / len(scores) if scores else 0.0


def _get_braf_delta(drug: DrugEntry, braf_status: BRAFStatus) -> float:
    if braf_status == "positive":
        return drug.braf_delta_positive
    if braf_status == "negative":
        return drug.braf_delta_negative
    return drug.braf_delta_unknown


def build_score_rationale(
    drug: DrugEntry,
    braf_status: BRAFStatus,
    msh2_loss: bool,
    orr_base: float,
    confidence: float,
    braf_delta: float,
    target_d: float,
    final_score: float,
) -> str:
    """
    Hard-coded score rationale string. NO LLM. All numbers computed from inputs.
    """
    orr_pct = f"{orr_base:.0%}" if orr_base > 0 else "No canine ORR data"
    parts = [
        f"score={final_score:.3f}",
        f"orr_base={orr_pct}×{confidence:.1f}(confidence)={round(orr_base*confidence,3):.3f}",
    ]
    if braf_delta != 0.0:
        parts.append(f"braf_delta={braf_delta:+.2f}({braf_status})")
    if target_d != 0.0:
        parts.append(f"target_delta={target_d:+.3f}")
    if drug.braf_requirement == "preferred" and braf_status == "positive":
        parts.append("BRAF_V595E_confirmed:preferred_drug")
    elif drug.braf_requirement == "required" and braf_status == "positive":
        parts.append("BRAF_V595E_confirmed:required_drug")
    elif drug.braf_requirement == "none":
        parts.append("BRAF_agnostic")
    if msh2_loss and drug.name in ("carboplatin", "gemcitabine"):
        parts.append("MSH2_loss:MMR_deficiency_flag")
    return " | ".join(parts)


def score_tcc(
    expr: dict[str, float] | None = None,
    braf_status: BRAFStatus = "unknown",
    breed: str = "other",
    msh2_loss: bool = False,
) -> TCCResult:
    """
    Rank PASS drugs for a canine TCC patient using ORR-anchored formula.
    Only drugs with pk_status=VERIFIED are eligible for PASS.
    At launch, only toceranib qualifies.
    """
    expr = expr or {}
    braf_prob = get_braf_prior(breed)
    if braf_status == "positive":
        braf_prob = max(braf_prob, 0.90)
    elif braf_status == "negative":
        braf_prob = min(braf_prob, 0.10)

    subtype = (
        "BRAF-mutant TCC" if braf_status == "positive"
        else "MMR-deficient TCC" if msh2_loss
        else "TCC-NOS"
    )

    top_genes = sorted(expr.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    top_altered = [g for g, _ in top_genes]

    # Score PASS drugs (only VERIFIED PK drugs)
    scored: list[DrugRecommendation] = []
    for drug in PASS_DRUGS:
        orr_base   = drug.orr or 0.0
        confidence = drug.evidence_confidence
        braf_delta = _get_braf_delta(drug, braf_status)
        target_d   = _target_delta(drug, expr)

        raw = orr_base * confidence + braf_delta + target_d
        final_score = round(min(max(raw, 0.0), 1.0), 3)

        ev_tier = EVIDENCE_TIER_MAP.get(drug.name, "C")

        gate_input = {
            "pk_status": drug.pk_status,
            "quarantine_reason": drug.quarantine_reason,
            "feasibility_gate": {
                "ic50_um": drug.pk_ic50_um,
                "cmax_um": drug.pk_cmax_um,
                "plasma_protein_binding": drug.pk_ppb,
            },
        }
        gate_result = compute_feasibility(gate_input)

        rationale = build_score_rationale(
            drug, braf_status, msh2_loss,
            orr_base, confidence, braf_delta, target_d, final_score,
        )

        scored.append(DrugRecommendation(
            drug=drug.name,
            score=final_score,
            rank=0,
            mechanism=drug.mechanism,
            targets=drug.targets,
            braf_requirement=drug.braf_requirement,
            verdict=drug.verdict,
            quarantined=False,
            rationale=rationale,
            evidence_tier=ev_tier,
            pk_status=drug.pk_status,
            feasibility_verdict=gate_result.get("verdict"),
            gap_ratio=gate_result.get("gap_ratio"),
            orr=drug.orr,
            orr_source=drug.orr_source,
        ))

    scored.sort(key=lambda x: x.score, reverse=True)
    for i, r in enumerate(scored):
        r.rank = i + 1

    # Quarantined drugs
    quarantined_recs: list[DrugRecommendation] = []
    for drug in QUARANTINE_DRUGS:
        ev_tier = EVIDENCE_TIER_MAP.get(drug.name, "C")
        quarantined_recs.append(DrugRecommendation(
            drug=drug.name,
            score=0.0,
            rank=99,
            mechanism=drug.mechanism,
            targets=drug.targets,
            braf_requirement=drug.braf_requirement,
            verdict="QUARANTINE",
            quarantined=True,
            rationale=drug.quarantine_reason or "Quarantined",
            evidence_tier=ev_tier,
            pk_status=drug.pk_status,
            feasibility_verdict="UNVERIFIED",
            gap_ratio=None,
            orr=drug.orr,
            orr_source=drug.orr_source,
            quarantine_reason=drug.quarantine_reason,
        ))

    top_drug = scored[0] if scored else None
    summary = (
        f"{subtype} | BRAF probability {braf_prob:.0%} | "
        f"{'Top drug: ' + top_drug.drug + ' (score ' + str(top_drug.score) + ')' if top_drug else 'No verified drugs available'} | "
        f"{'MSH2 loss detected' if msh2_loss else 'MSH2 intact'} | "
        f"Verified drugs: {len(scored)} | Quarantined: {len(quarantined_recs)}"
    )

    return TCCResult(
        subtype=subtype,
        braf_status=braf_status,
        braf_probability=braf_prob,
        msh2_loss=msh2_loss,
        top_altered_genes=top_altered,
        recommendations=scored,
        quarantined=quarantined_recs,
        summary=summary,
    )
