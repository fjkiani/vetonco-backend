"""
VetOnco — Vet Pharmacist Recipe Card Generator
Produces a structured recipe card (JSON + printable text) for compounding pharmacists.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from services.tcc_dosing import DoseResult


@dataclass
class RecipeCard:
    pet_name: str
    species: str
    breed: str
    weight_kg: float
    bsa_m2: float
    prescribing_vet: str
    date_issued: str
    drugs: list[dict[str, Any]]
    interactions: list[str]
    monitoring: list[str]
    printable_text: str


# Known drug interactions relevant to canine TCC protocols
DRUG_INTERACTIONS: dict[frozenset, str] = {
    frozenset({"piroxicam", "carboplatin"}): (
        "Piroxicam + carboplatin: additive nephrotoxicity risk — monitor BUN/creatinine closely; "
        "ensure adequate hydration before carboplatin infusion"
    ),
    frozenset({"piroxicam", "gemcitabine"}): (
        "Piroxicam + gemcitabine: potential additive GI toxicity — monitor for GI ulceration"
    ),
    frozenset({"toceranib", "piroxicam"}): (
        "Toceranib + piroxicam: increased GI hemorrhage risk — consider misoprostol prophylaxis; "
        "monitor for melena and hematochezia"
    ),
    frozenset({"trametinib", "toceranib"}): (
        "Trametinib + toceranib: overlapping myelosuppression — CBC monitoring q7d; "
        "dose-reduce toceranib first if Grade 3+ neutropenia"
    ),
    frozenset({"mitoxantrone", "vinblastine"}): (
        "Mitoxantrone + vinblastine: additive myelosuppression — stagger administration; "
        "CBC nadir monitoring required"
    ),
}

MONITORING_MAP: dict[str, list[str]] = {
    "piroxicam": ["BUN/creatinine q4w", "Urinalysis q4w", "GI symptom check q2w"],
    "toceranib": ["CBC q7d (first month)", "Chemistry panel q4w", "Blood pressure q4w"],
    "mitoxantrone": ["CBC day 7 post-infusion", "Cardiac echo q3 cycles"],
    "vinblastine": ["CBC day 7 post-infusion", "Neurologic exam q4w"],
    "carboplatin": ["BUN/creatinine pre-dose", "CBC day 10-14 post-infusion"],
    "gemcitabine": ["CBC q7d", "Chemistry panel q4w"],
    "trametinib": ["Dermatologic exam q2w", "Ophthalmologic exam q4w", "CBC q4w"],
}


def _detect_interactions(drug_names: list[str]) -> list[str]:
    found = []
    for pair, msg in DRUG_INTERACTIONS.items():
        if pair.issubset(set(drug_names)):
            found.append(msg)
    return found


def _build_monitoring(drug_names: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for drug in drug_names:
        for item in MONITORING_MAP.get(drug, []):
            if item not in seen:
                seen.add(item)
                result.append(item)
    return result


def _build_printable(
    pet_name: str,
    species: str,
    breed: str,
    weight_kg: float,
    bsa_m2: float,
    prescribing_vet: str,
    date_issued: str,
    drugs: list[dict],
    interactions: list[str],
    monitoring: list[str],
) -> str:
    lines = [
        "=" * 60,
        "  VETONCO COMPOUNDING RECIPE CARD",
        "=" * 60,
        f"  Patient:   {pet_name} ({species} — {breed})",
        f"  Weight:    {weight_kg} kg  |  BSA: {bsa_m2} m²",
        f"  Vet:       {prescribing_vet}",
        f"  Date:      {date_issued}",
        "=" * 60,
        "",
        "PRESCRIBED MEDICATIONS",
        "-" * 60,
    ]
    for d in drugs:
        lines += [
            f"  Drug:      {d['drug'].upper()}",
            f"  Dose:      {d['final_dose_mg']} mg  ({d['dose_per_kg']} mg/kg)",
            f"  Schedule:  {d['schedule']}  |  Route: {d['route']}",
            f"  BSA dose:  {d['dose_mg']} mg (pre-adjustment)",
        ]
        if d.get("renal_adjustment") != "none":
            lines.append(f"  Renal adj: {d['renal_adjustment']}")
        if d.get("hepatic_adjustment") != "none":
            lines.append(f"  Hepatic adj: {d['hepatic_adjustment']}")
        if d.get("warnings"):
            for w in d["warnings"]:
                lines.append(f"  ⚠ {w}")
        lines.append(f"  Notes:     {d.get('notes', '')}")
        lines.append("")

    if interactions:
        lines += ["DRUG INTERACTIONS", "-" * 60]
        for ix in interactions:
            lines.append(f"  ⚠ {ix}")
        lines.append("")

    if monitoring:
        lines += ["MONITORING SCHEDULE", "-" * 60]
        for m in monitoring:
            lines.append(f"  • {m}")
        lines.append("")

    lines += [
        "=" * 60,
        "  FOR VETERINARY USE ONLY — VetOnco Clinical Decision Support",
        "  Verify all doses before dispensing.",
        "=" * 60,
    ]
    return "\n".join(lines)


def generate_recipe_card(
    pet_name: str,
    species: str,
    breed: str,
    weight_kg: float,
    prescribing_vet: str,
    dose_results: list[DoseResult],
    date_issued: str | None = None,
) -> RecipeCard:
    """Generate a vet pharmacist recipe card from a list of DoseResult objects."""
    if date_issued is None:
        date_issued = date.today().isoformat()

    bsa_m2 = dose_results[0].bsa_m2 if dose_results else 0.0
    drug_names = [d.drug for d in dose_results]

    drugs_json = [
        {
            "drug": d.drug,
            "dose_mg": d.dose_mg,
            "dose_per_kg": d.dose_per_kg,
            "final_dose_mg": d.final_dose_mg,
            "schedule": d.schedule,
            "route": d.route,
            "renal_adjustment": d.renal_adjustment,
            "hepatic_adjustment": d.hepatic_adjustment,
            "notes": d.notes,
            "warnings": d.warnings,
        }
        for d in dose_results
    ]

    interactions = _detect_interactions(drug_names)
    monitoring = _build_monitoring(drug_names)

    printable = _build_printable(
        pet_name, species, breed, weight_kg, bsa_m2,
        prescribing_vet, date_issued, drugs_json, interactions, monitoring,
    )

    return RecipeCard(
        pet_name=pet_name,
        species=species,
        breed=breed,
        weight_kg=weight_kg,
        bsa_m2=bsa_m2,
        prescribing_vet=prescribing_vet,
        date_issued=date_issued,
        drugs=drugs_json,
        interactions=interactions,
        monitoring=monitoring,
        printable_text=printable,
    )
