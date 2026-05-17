"""
VetOnco — Canine TCC Dosing Calculator
Weight-based dosing with BSA, renal/hepatic adjustments.
BSA formula: 0.101 × weight_kg^0.667 (Veterinary standard)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

from services.tcc_gene_panels import PASS_DRUGS, DrugEntry

ReductionLevel = Literal["none", "25%", "50%", "hold"]


@dataclass
class DoseResult:
    drug: str
    weight_kg: float
    bsa_m2: float
    dose_mg: float
    dose_per_kg: float
    schedule: str
    route: str
    renal_adjustment: ReductionLevel
    hepatic_adjustment: ReductionLevel
    final_dose_mg: float
    notes: str
    warnings: list[str]


def compute_bsa(weight_kg: float) -> float:
    """Canine BSA in m² using Veterinary standard formula."""
    return round(0.101 * (weight_kg ** 0.667), 4)


def _renal_reduction(drug_name: str, creatinine_mg_dl: float | None) -> ReductionLevel:
    if creatinine_mg_dl is None:
        return "none"
    if drug_name == "carboplatin":
        if creatinine_mg_dl > 3.0:
            return "hold"
        if creatinine_mg_dl > 2.0:
            return "50%"
        if creatinine_mg_dl > 1.5:
            return "25%"
    elif drug_name in ("gemcitabine", "piroxicam"):
        if creatinine_mg_dl > 2.5:
            return "hold"
        if creatinine_mg_dl > 1.8:
            return "25%"
    return "none"


def _hepatic_reduction(drug_name: str, alt_u_l: float | None) -> ReductionLevel:
    if alt_u_l is None:
        return "none"
    if drug_name in ("mitoxantrone", "vinblastine", "trametinib"):
        if alt_u_l > 500:
            return "hold"
        if alt_u_l > 250:
            return "50%"
        if alt_u_l > 150:
            return "25%"
    elif drug_name in ("toceranib",):
        if alt_u_l > 400:
            return "hold"
        if alt_u_l > 200:
            return "25%"
    return "none"


def _apply_reduction(dose_mg: float, level: ReductionLevel) -> float:
    if level == "none":
        return dose_mg
    if level == "25%":
        return round(dose_mg * 0.75, 2)
    if level == "50%":
        return round(dose_mg * 0.50, 2)
    return 0.0  # hold


# BSA-based dosing (mg/m²)
BSA_DOSE_MAP: dict[str, tuple[float, str, str]] = {
    # drug: (mg_per_m2, schedule, route)
    "mitoxantrone": (5.5, "q21d", "IV"),
    "vinblastine": (2.0, "q7d", "IV"),
    "carboplatin": (300.0, "q21d", "IV"),
    "gemcitabine": (800.0, "q7d", "IV"),
}

# Weight-based dosing (mg/kg)
WEIGHT_DOSE_MAP: dict[str, tuple[float, str, str]] = {
    # drug: (mg_per_kg, schedule, route)
    "piroxicam": (0.3, "q24h", "PO"),
    "toceranib": (2.75, "q48h", "PO"),
    "trametinib": (0.03, "q24h", "PO"),
}


def compute_canine_dose(
    drug_name: str,
    weight_kg: float,
    creatinine_mg_dl: float | None = None,
    alt_u_l: float | None = None,
) -> DoseResult:
    """Compute a single drug dose for a canine patient."""
    bsa = compute_bsa(weight_kg)
    warnings: list[str] = []

    if drug_name in BSA_DOSE_MAP:
        mg_per_m2, schedule, route = BSA_DOSE_MAP[drug_name]
        base_dose = round(mg_per_m2 * bsa, 2)
        dose_per_kg = round(base_dose / weight_kg, 3)
    elif drug_name in WEIGHT_DOSE_MAP:
        mg_per_kg, schedule, route = WEIGHT_DOSE_MAP[drug_name]
        base_dose = round(mg_per_kg * weight_kg, 2)
        dose_per_kg = mg_per_kg
    else:
        return DoseResult(
            drug=drug_name, weight_kg=weight_kg, bsa_m2=bsa,
            dose_mg=0.0, dose_per_kg=0.0, schedule="unknown", route="unknown",
            renal_adjustment="none", hepatic_adjustment="none",
            final_dose_mg=0.0,
            notes=f"Drug '{drug_name}' not in dosing database",
            warnings=[f"Unknown drug: {drug_name}"],
        )

    renal_adj = _renal_reduction(drug_name, creatinine_mg_dl)
    hepatic_adj = _hepatic_reduction(drug_name, alt_u_l)

    # Apply the more conservative adjustment
    reduction_order = ["none", "25%", "50%", "hold"]
    effective_adj = max(renal_adj, hepatic_adj, key=lambda x: reduction_order.index(x))
    final_dose = _apply_reduction(base_dose, effective_adj)

    if renal_adj != "none":
        warnings.append(f"Renal adjustment ({renal_adj}): creatinine {creatinine_mg_dl} mg/dL")
    if hepatic_adj != "none":
        warnings.append(f"Hepatic adjustment ({hepatic_adj}): ALT {alt_u_l} U/L")
    if final_dose == 0.0:
        warnings.append(f"HOLD {drug_name} — lab values exceed safe threshold")

    # Drug-specific notes
    notes_map = {
        "piroxicam": "Administer with food; consider misoprostol GI protection",
        "toceranib": "Monitor CBC weekly; hold for Grade 3+ neutropenia",
        "mitoxantrone": "IV slow infusion; vesicant — avoid extravasation",
        "vinblastine": "IV push; myelosuppression nadir day 7",
        "carboplatin": "IV 30-min infusion; pre-hydrate; monitor BUN/creatinine",
        "gemcitabine": "IV 30-min infusion; often combined with carboplatin",
        "trametinib": "Off-label; monitor for dermatologic toxicity",
    }

    return DoseResult(
        drug=drug_name,
        weight_kg=weight_kg,
        bsa_m2=bsa,
        dose_mg=base_dose,
        dose_per_kg=dose_per_kg,
        schedule=schedule,
        route=route,
        renal_adjustment=renal_adj,
        hepatic_adjustment=hepatic_adj,
        final_dose_mg=final_dose,
        notes=notes_map.get(drug_name, ""),
        warnings=warnings,
    )


def compute_full_panel_dosage(
    weight_kg: float,
    creatinine_mg_dl: float | None = None,
    alt_u_l: float | None = None,
    drugs: list[str] | None = None,
) -> list[DoseResult]:
    """Compute doses for all PASS drugs (or a specified subset)."""
    target_drugs = drugs or [d.name for d in PASS_DRUGS]
    return [
        compute_canine_dose(drug, weight_kg, creatinine_mg_dl, alt_u_l)
        for drug in target_drugs
    ]
