"""
VetOnco — TCC Drug Scorer
score_tcc(expr, braf_status, breed, msh2_loss) → TCCResult
Ranks PASS drugs by multi-factor score; flags QUARANTINE drugs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

from services.tcc_gene_panels import (
    DRUG_PANEL, PASS_DRUGS, QUARANTINE_DRUGS,
    TCC_PANEL_GENES_TIER_A, TCC_PANEL_GENES_TIER_B, TCC_PANEL_GENES_TIER_C,
    DrugEntry, get_braf_prior,
)

BRAFStatus = Literal["positive", "negative", "unknown"]


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
# Scoring weights
# ---------------------------------------------------------------------------

TIER_WEIGHT = {"A": 1.0, "B": 0.7, "C": 0.4}

BRAF_SCORE_MAP = {
    # (braf_requirement, braf_status) → bonus
    ("required", "positive"): 1.0,
    ("required", "negative"): -0.5,
    ("required", "unknown"): 0.2,
    ("preferred", "positive"): 0.6,
    ("preferred", "negative"): 0.0,
    ("preferred", "unknown"): 0.2,
    ("none", "positive"): 0.0,
    ("none", "negative"): 0.0,
    ("none", "unknown"): 0.0,
}

EVIDENCE_TIER_MAP: dict[str, Literal["A", "B", "C"]] = {
    "piroxicam": "A",
    "toceranib": "A",
    "mitoxantrone": "A",
    "vinblastine": "B",
    "carboplatin": "B",
    "gemcitabine": "B",
    "trametinib": "C",
}


def _gene_tier(gene: str) -> str:
    if gene in TCC_PANEL_GENES_TIER_A:
        return "A"
    if gene in TCC_PANEL_GENES_TIER_B:
        return "B"
    return "C"


def _target_score(drug: DrugEntry, expr: dict[str, float]) -> float:
    """Score based on expression of drug targets in the panel."""
    if not expr or not drug.targets:
        return 0.0
    scores = []
    for t in drug.targets:
        if t in expr:
            # Normalize log2FC: positive expression → higher score
            val = expr[t]
            tier_w = TIER_WEIGHT.get(_gene_tier(t), 0.4)
            scores.append(min(max(val / 4.0, -1.0), 1.0) * tier_w)
    return sum(scores) / len(scores) if scores else 0.0


def _build_rationale(drug: DrugEntry, braf_status: BRAFStatus, msh2_loss: bool, score: float) -> str:
    parts = []
    if drug.braf_requirement == "required" and braf_status == "positive":
        parts.append("BRAF V595E confirmed — direct target match")
    elif drug.braf_requirement == "preferred" and braf_status == "positive":
        parts.append("BRAF V595E positive — enhanced response expected")
    elif drug.braf_requirement == "none":
        parts.append("BRAF-agnostic mechanism")
    if msh2_loss and drug.name in ("carboplatin", "gemcitabine"):
        parts.append("MSH2 loss → MMR deficiency may enhance platinum/nucleoside response")
    if drug.notes:
        parts.append(drug.notes)
    return "; ".join(parts) if parts else f"Score: {score:.2f}"


def score_tcc(
    expr: dict[str, float] | None = None,
    braf_status: BRAFStatus = "unknown",
    breed: str = "other",
    msh2_loss: bool = False,
) -> TCCResult:
    """
    Rank PASS drugs for a canine TCC patient.

    Parameters
    ----------
    expr : dict gene→log2FC (optional; from RNA-seq or panel)
    braf_status : "positive" | "negative" | "unknown"
    breed : dog breed string (used for BRAF prior)
    msh2_loss : MSH2 IHC loss flag
    """
    expr = expr or {}
    braf_prob = get_braf_prior(breed)
    if braf_status == "positive":
        braf_prob = max(braf_prob, 0.90)
    elif braf_status == "negative":
        braf_prob = min(braf_prob, 0.10)

    # Determine subtype
    if braf_status == "positive":
        subtype = "BRAF-mutant TCC"
    elif msh2_loss:
        subtype = "MMR-deficient TCC"
    else:
        subtype = "TCC-NOS"

    # Top altered genes from expression
    top_genes = sorted(expr.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    top_altered = [g for g, _ in top_genes]

    # Score PASS drugs
    scored: list[DrugRecommendation] = []
    for drug in PASS_DRUGS:
        braf_bonus = BRAF_SCORE_MAP.get((drug.braf_requirement, braf_status), 0.0)
        target_s = _target_score(drug, expr)
        ev_tier = EVIDENCE_TIER_MAP.get(drug.name, "C")
        ev_bonus = TIER_WEIGHT[ev_tier]
        raw = 0.4 * ev_bonus + 0.4 * braf_bonus + 0.2 * target_s
        score = round(min(max(raw, 0.0), 1.0), 3)
        rationale = _build_rationale(drug, braf_status, msh2_loss, score)
        scored.append(DrugRecommendation(
            drug=drug.name,
            score=score,
            rank=0,
            mechanism=drug.mechanism,
            targets=drug.targets,
            braf_requirement=drug.braf_requirement,
            verdict=drug.verdict,
            quarantined=False,
            rationale=rationale,
            evidence_tier=ev_tier,
        ))

    scored.sort(key=lambda x: x.score, reverse=True)
    for i, r in enumerate(scored):
        r.rank = i + 1

    # Flag QUARANTINE drugs
    quarantined_recs: list[DrugRecommendation] = []
    for drug in QUARANTINE_DRUGS:
        quarantined_recs.append(DrugRecommendation(
            drug=drug.name,
            score=0.0,
            rank=99,
            mechanism=drug.mechanism,
            targets=drug.targets,
            braf_requirement=drug.braf_requirement,
            verdict="QUARANTINE",
            quarantined=True,
            rationale=drug.quarantine_reason or "Quarantined — see reason",
            evidence_tier="C",
            quarantine_reason=drug.quarantine_reason,
        ))

    summary = (
        f"{subtype} | BRAF probability {braf_prob:.0%} | "
        f"Top drug: {scored[0].drug} (score {scored[0].score:.2f}) | "
        f"{'MSH2 loss detected' if msh2_loss else 'MSH2 intact'}"
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
