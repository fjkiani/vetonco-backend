"""
VetOnco — Canine TCC Gene Panels & Drug Panel
19-gene panel (Tier A/B/C), 7 PASS drugs + 3 QUARANTINE, breed BRAF priors.
Source: Dhawan et al. 2018 (GSE110661), Knapp et al. 2014, Decker et al. 2015.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

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
# Drug panel
# ---------------------------------------------------------------------------

@dataclass
class DrugEntry:
    name: str
    mechanism: str
    targets: list[str]
    braf_requirement: Literal["required", "preferred", "none"]
    verdict: Literal["PASS", "QUARANTINE"]
    quarantined: bool
    quarantine_reason: str | None = None
    chembl_id: str | None = None
    typical_dose_mg_kg: float | None = None
    schedule: str | None = None
    notes: str | None = None


DRUG_PANEL: list[DrugEntry] = [
    DrugEntry(
        name="piroxicam",
        mechanism="COX-1/COX-2 inhibitor (NSAID)",
        targets=["PTGS1", "PTGS2"],
        braf_requirement="none",
        verdict="PASS",
        quarantined=False,
        chembl_id="CHEMBL527",
        typical_dose_mg_kg=0.3,
        schedule="q24h",
        notes="First-line for canine TCC; GI protection recommended",
    ),
    DrugEntry(
        name="toceranib",
        mechanism="Multi-kinase inhibitor (VEGFR2, PDGFR, KIT, FLT3)",
        targets=["KDR", "PDGFRB", "KIT", "FLT3"],
        braf_requirement="preferred",
        verdict="PASS",
        quarantined=False,
        chembl_id="CHEMBL1289926",
        typical_dose_mg_kg=2.75,
        schedule="q48h",
        notes="Preferred in BRAF+ cases; monitor CBC weekly",
    ),
    DrugEntry(
        name="mitoxantrone",
        mechanism="Topoisomerase II inhibitor / DNA intercalator",
        targets=["TOP2A"],
        braf_requirement="none",
        verdict="PASS",
        quarantined=False,
        chembl_id="CHEMBL58",
        typical_dose_mg_kg=None,
        schedule="q21d IV",
        notes="5–6 mg/m² IV; requires BSA calculation",
    ),
    DrugEntry(
        name="vinblastine",
        mechanism="Vinca alkaloid — microtubule destabilizer",
        targets=["TUBB"],
        braf_requirement="none",
        verdict="PASS",
        quarantined=False,
        chembl_id="CHEMBL255863",
        typical_dose_mg_kg=None,
        schedule="q7d IV",
        notes="2 mg/m² IV weekly; myelosuppression monitoring required",
    ),
    DrugEntry(
        name="carboplatin",
        mechanism="Platinum alkylating agent — DNA crosslinker",
        targets=["BRCA2", "ERCC2"],
        braf_requirement="none",
        verdict="PASS",
        quarantined=False,
        chembl_id="CHEMBL11359",
        typical_dose_mg_kg=None,
        schedule="q21d IV",
        notes="300 mg/m² IV; nephrotoxicity monitoring; avoid in renal disease",
    ),
    DrugEntry(
        name="gemcitabine",
        mechanism="Nucleoside analog — ribonucleotide reductase inhibitor",
        targets=["RRM1", "RRM2"],
        braf_requirement="none",
        verdict="PASS",
        quarantined=False,
        chembl_id="CHEMBL888",
        typical_dose_mg_kg=None,
        schedule="q7d IV",
        notes="800 mg/m² IV; often combined with carboplatin",
    ),
    DrugEntry(
        name="trametinib",
        mechanism="MEK1/2 inhibitor",
        targets=["MAP2K1", "MAP2K2"],
        braf_requirement="required",
        verdict="PASS",
        quarantined=False,
        chembl_id="CHEMBL2103875",
        typical_dose_mg_kg=0.03,
        schedule="q24h",
        notes="BRAF V595E required; off-label; limited canine PK data",
    ),
    # QUARANTINE drugs
    DrugEntry(
        name="vemurafenib",
        mechanism="BRAF V600E inhibitor",
        targets=["BRAF"],
        braf_requirement="required",
        verdict="QUARANTINE",
        quarantined=True,
        quarantine_reason="Human BRAF V600E inhibitor; canine mutation is V595E — paradoxical activation risk",
        chembl_id="CHEMBL1229517",
    ),
    DrugEntry(
        name="dabrafenib",
        mechanism="BRAF V600E/K inhibitor",
        targets=["BRAF"],
        braf_requirement="required",
        verdict="QUARANTINE",
        quarantined=True,
        quarantine_reason="Same V600E vs V595E mismatch as vemurafenib; insufficient canine safety data",
        chembl_id="CHEMBL2028663",
    ),
    DrugEntry(
        name="erdafitinib",
        mechanism="Pan-FGFR inhibitor",
        targets=["FGFR1", "FGFR2", "FGFR3", "FGFR4"],
        braf_requirement="none",
        verdict="QUARANTINE",
        quarantined=True,
        quarantine_reason="No published canine PK/PD data; FGFR3 amplification not validated as canine TCC driver",
        chembl_id="CHEMBL3545110",
    ),
]

PASS_DRUGS = [d for d in DRUG_PANEL if not d.quarantined]
QUARANTINE_DRUGS = [d for d in DRUG_PANEL if d.quarantined]

# ---------------------------------------------------------------------------
# Breed BRAF priors
# ---------------------------------------------------------------------------

BREED_BRAF_PRIOR: dict[str, float] = {
    "scottish terrier": 0.85,
    "shetland sheepdog": 0.72,
    "beagle": 0.68,
    "west highland white terrier": 0.65,
    "fox terrier": 0.60,
    "airedale terrier": 0.58,
    "mixed breed": 0.35,
    "labrador retriever": 0.30,
    "golden retriever": 0.28,
    "other": 0.35,
}

def get_braf_prior(breed: str) -> float:
    """Return breed-specific BRAF V595E prior probability."""
    return BREED_BRAF_PRIOR.get(breed.lower().strip(), 0.35)
