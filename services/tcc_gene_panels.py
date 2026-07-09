"""
VetOnco — Canine TCC Gene Panels & Drug Panel v2
19-gene panel (Tier A/B/C), strict PK quarantine policy.

QUARANTINE POLICY (NON-NEGOTIABLE):
    A drug is VERIFIED only if ALL THREE of the following come from a
    PRIMARY CANINE STUDY (not a formulary, not human data, not in vitro only):
        1. IC50 (µM) — canine TCC cell line or in vivo
        2. Cmax (µM) — canine PK study at clinical dose
        3. PPB (%) — canine plasma protein binding

    At launch: ONLY toceranib is VERIFIED.
    All other drugs are QUARANTINED until primary canine PK is sourced.

Sources:
    Gene panel: Dhawan et al. 2018 (GSE110661), Knapp et al. 2014, Decker et al. 2015
    Toceranib PK: Henry et al. 2009 (PMID 19185954)
    ORR data: see VETONCO_GROUND_TRUTH_BENCHMARKS.mdc
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# Gene panel
# ---------------------------------------------------------------------------

TCC_PANEL_GENES_TIER_A = [
    "FGFR3", "EGFR", "ERBB2", "PIK3CA", "HRAS",
]

TCC_PANEL_GENES_TIER_B = [
    "BRAF", "MAP2K1", "RB1", "CDKN2A", "TP53",
    "MDM2", "PTEN", "TSC1",
]

TCC_PANEL_GENES_TIER_C = [
    "MSH2", "MLH1", "MSH6", "PMS2", "BRCA2",
    "ERCC2",
]

ALL_PANEL_GENES = TCC_PANEL_GENES_TIER_A + TCC_PANEL_GENES_TIER_B + TCC_PANEL_GENES_TIER_C


# ---------------------------------------------------------------------------
# DrugEntry dataclass
# ---------------------------------------------------------------------------

@dataclass
class DrugEntry:
    name: str
    mechanism: str
    targets: list[str]
    braf_requirement: Literal["required", "preferred", "none"]
    verdict: str

    # ORR ground truth (published canine TCC trials)
    orr: Optional[float]          # fraction (0.0–1.0), None if no data
    orr_source: Optional[str]     # citation string
    orr_n: Optional[int]          # trial n

    # Evidence confidence (for ORR × confidence formula)
    evidence_confidence: float    # A=1.0, B=0.7, C=0.4

    # BRAF deltas (patient-specific score adjustments)
    braf_delta_positive: float    # braf_status == "positive"
    braf_delta_negative: float    # braf_status == "negative"
    braf_delta_unknown: float     # braf_status == "unknown"

    # PK fields — ALL must come from primary canine study to be VERIFIED
    pk_status: Literal["VERIFIED", "UNVERIFIED"]
    pk_ic50_um: Optional[float]   # IC50 in µM (canine TCC cell line)
    pk_cmax_um: Optional[float]   # Cmax in µM (canine PK study)
    pk_ppb: Optional[float]       # plasma protein binding fraction (0.0–1.0)
    pk_source: Optional[str]      # primary canine PK citation

    # Quarantine
    quarantine_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# BRAF breed priors
# ---------------------------------------------------------------------------

BRAF_BREED_PRIORS: dict[str, float] = {
    "scottish terrier": 0.85,
    "beagle": 0.70,
    "shetland sheepdog": 0.65,
    "west highland white terrier": 0.60,
    "fox terrier": 0.55,
    "other": 0.40,
}


def get_braf_prior(breed: str) -> float:
    return BRAF_BREED_PRIORS.get(breed.lower().strip(), 0.40)


# ---------------------------------------------------------------------------
# Drug panel
# ---------------------------------------------------------------------------

DRUG_PANEL: list[DrugEntry] = [

    # ── VERIFIED ──────────────────────────────────────────────────────────
    DrugEntry(
        name="toceranib",
        mechanism="Multi-kinase inhibitor (VEGFR2, PDGFR, KIT)",
        targets=["VEGFR2", "PDGFR", "KIT", "BRAF"],
        braf_requirement="preferred",
        verdict="PASS",
        orr=0.26,
        orr_source="Bernabe et al. 2013 (PMID 23279175)",
        orr_n=30,
        evidence_confidence=1.0,
        braf_delta_positive=+0.12,
        braf_delta_negative=0.00,
        braf_delta_unknown=+0.05,
        pk_status="VERIFIED",
        pk_ic50_um=0.010,
        pk_cmax_um=0.220,
        pk_ppb=0.93,
        pk_source="Henry et al. 2009 (PMID 19185954)",
        quarantine_reason=None,
    ),

    # ── QUARANTINED — PPB not from primary canine study ───────────────────
    DrugEntry(
        name="mitoxantrone",
        mechanism="Topoisomerase II inhibitor",
        targets=["TOP2A", "TOP2B"],
        braf_requirement="none",
        verdict="QUARANTINE",
        orr=0.35,
        orr_source="Henry et al. 2003 (PMID 12825866) — combination with piroxicam",
        orr_n=48,
        evidence_confidence=1.0,
        braf_delta_positive=0.00,
        braf_delta_negative=0.00,
        braf_delta_unknown=0.00,
        pk_status="UNVERIFIED",
        pk_ic50_um=0.10,
        pk_cmax_um=0.75,
        pk_ppb=None,   # PPB not from primary canine study
        pk_source="Ogilvie et al. 1994 — Cmax/IC50 only; PPB unconfirmed",
        quarantine_reason="PPB not confirmed from primary canine study (Ogilvie 1994 reports Cmax/IC50 only)",
    ),

    # ── QUARANTINED — No primary canine PK study ──────────────────────────
    DrugEntry(
        name="piroxicam",
        mechanism="COX-1/COX-2 inhibitor (NSAID)",
        targets=["PTGS1", "PTGS2"],
        braf_requirement="none",
        verdict="QUARANTINE",
        orr=0.18,
        orr_source="Knapp et al. 1994 (PMID 8188077)",
        orr_n=34,
        evidence_confidence=1.0,
        braf_delta_positive=0.00,
        braf_delta_negative=0.00,
        braf_delta_unknown=0.00,
        pk_status="UNVERIFIED",
        pk_ic50_um=None,
        pk_cmax_um=None,
        pk_ppb=None,
        pk_source=None,
        quarantine_reason="No primary canine PK study. IC50, Cmax, PPB not confirmed from canine data.",
    ),

    DrugEntry(
        name="vinblastine",
        mechanism="Vinca alkaloid — microtubule inhibitor",
        targets=["TUBB", "TUBA1A"],
        braf_requirement="none",
        verdict="QUARANTINE",
        orr=0.22,
        orr_source="Allstadt et al. 2015 (PMID 25823835)",
        orr_n=36,
        evidence_confidence=0.7,
        braf_delta_positive=0.00,
        braf_delta_negative=0.00,
        braf_delta_unknown=0.00,
        pk_status="UNVERIFIED",
        pk_ic50_um=None,
        pk_cmax_um=None,
        pk_ppb=None,
        pk_source=None,
        quarantine_reason="No primary canine PK study. IC50, Cmax, PPB not confirmed from canine data.",
    ),

    DrugEntry(
        name="carboplatin",
        mechanism="Platinum alkylating agent — DNA crosslinker",
        targets=["ERCC1", "ERCC2", "MSH2"],
        braf_requirement="none",
        verdict="QUARANTINE",
        orr=0.38,
        orr_source="Boria et al. 2005 (PMID 15822463)",
        orr_n=31,
        evidence_confidence=0.7,
        braf_delta_positive=0.00,
        braf_delta_negative=0.00,
        braf_delta_unknown=0.00,
        pk_status="UNVERIFIED",
        pk_ic50_um=None,
        pk_cmax_um=None,
        pk_ppb=None,
        pk_source=None,
        quarantine_reason="No primary canine Cmax study. PPB ~2% is established pharmacology but Cmax not confirmed from primary canine study. No PARTIAL loopholes.",
    ),

    DrugEntry(
        name="gemcitabine",
        mechanism="Nucleoside analog — ribonucleotide reductase inhibitor",
        targets=["RRM1", "RRM2"],
        braf_requirement="none",
        verdict="QUARANTINE",
        orr=0.40,
        orr_source="Robat et al. 2012 (PMID 22251430) — combination with piroxicam, n=10",
        orr_n=10,
        evidence_confidence=0.7,
        braf_delta_positive=0.00,
        braf_delta_negative=0.00,
        braf_delta_unknown=0.00,
        pk_status="UNVERIFIED",
        pk_ic50_um=None,
        pk_cmax_um=None,
        pk_ppb=None,
        pk_source=None,
        quarantine_reason="No primary canine PK study. ORR from n=10 combination trial (Robat 2012) — insufficient for monotherapy claim.",
    ),

    DrugEntry(
        name="trametinib",
        mechanism="MEK1/2 inhibitor",
        targets=["MAP2K1", "MAP2K2"],
        braf_requirement="required",
        verdict="QUARANTINE",
        orr=None,
        orr_source=None,
        orr_n=None,
        evidence_confidence=0.4,
        braf_delta_positive=+0.20,
        braf_delta_negative=-0.15,
        braf_delta_unknown=+0.05,
        pk_status="UNVERIFIED",
        pk_ic50_um=None,
        pk_cmax_um=None,
        pk_ppb=None,
        pk_source=None,
        quarantine_reason="No published canine TCC ORR. No primary canine PK study. Human PK data only.",
    ),
]

# ---------------------------------------------------------------------------
# Convenience partitions
# ---------------------------------------------------------------------------

PASS_DRUGS = [d for d in DRUG_PANEL if d.pk_status == "VERIFIED"]
QUARANTINE_DRUGS = [d for d in DRUG_PANEL if d.pk_status != "VERIFIED"]
