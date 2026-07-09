"""
VetOnco — Compounding Pharmacist Recipe Card Generator v2
Produces:
  1. Structured JSON (PharmacistDrugEntry schema)
  2. Printable ASCII text (backward-compatible)
  3. PDF via reportlab (pharmacist-facing document)
  4. Dispensing label text per drug

Compounding data sourced from:
  - USP <795> (non-sterile) and <797> (sterile) guidelines
  - Published canine veterinary formularies (Plumb's Veterinary Drug Handbook)
  - Manufacturer stability data

POLICY: All dose values come from tcc_dosing.py (deterministic arithmetic).
        This module formats and presents — it does NOT compute doses.
"""
from __future__ import annotations
import io
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional

from services.tcc_dosing import DoseResult

# ---------------------------------------------------------------------------
# Compounding data (hardcoded, sourced — no fabrication)
# ---------------------------------------------------------------------------

# (concentration_mg_ml, vehicle, beyond_use_days, storage, preparation_notes)
COMPOUNDING_DATA: dict[str, tuple[float | None, str | None, int, str, str]] = {
    "toceranib": (
        10.0,
        "Ora-Plus/Ora-Sweet 1:1",
        60,
        "Refrigerate 2–8°C. Protect from light.",
        "Crush tablets. Triturate with small volume of Ora-Plus to form paste. "
        "Geometrically incorporate Ora-Sweet to final volume. Shake well before use. "
        "Source: USP <795>; stability data from manufacturer (Pfizer).",
    ),
    "piroxicam": (
        1.0,
        "Ora-Plus/Ora-Sweet 1:1",
        30,
        "Store at room temperature 15–30°C. Protect from light.",
        "Crush capsule contents. Triturate with Ora-Plus. Qs to volume with Ora-Sweet. "
        "Source: USP <795>.",
    ),
    "trametinib": (
        0.5,
        "Ora-Plus/Ora-Sweet 1:1",
        30,
        "Refrigerate 2–8°C. Protect from light.",
        "Crush tablets. Triturate with Ora-Plus. Qs to volume with Ora-Sweet. "
        "Off-label compounding — verify stability before dispensing. Source: USP <795>.",
    ),
    "mitoxantrone": (
        2.0,
        "0.9% NaCl or D5W",
        7,
        "Refrigerate 2–8°C. Do NOT freeze. Protect from light.",
        "Dilute commercially available 2 mg/mL concentrate in 0.9% NaCl or D5W. "
        "Administer as slow IV infusion over 30 min. Vesicant — avoid extravasation. "
        "Source: USP <797>.",
    ),
    "vinblastine": (
        1.0,
        "0.9% NaCl",
        28,
        "Refrigerate 2–8°C. Do NOT freeze.",
        "Dilute in 0.9% NaCl. Administer as IV push over 1 min or short infusion. "
        "Vesicant — use central line if possible. Source: USP <797>.",
    ),
    "carboplatin": (
        10.0,
        "D5W",
        24,
        "Store at room temperature. Protect from light. Use within 24h of preparation.",
        "Dilute in D5W. Do NOT use NaCl (chloride displaces carbonate ligands). "
        "Administer as 30-min IV infusion. Pre-hydrate patient. Source: USP <797>.",
    ),
    "gemcitabine": (
        38.0,
        "0.9% NaCl",
        24,
        "Store at room temperature 20–25°C. Use within 24h.",
        "Reconstitute lyophilized powder with 0.9% NaCl (5 mL per 200 mg vial). "
        "Further dilute in 0.9% NaCl. Administer as 30-min IV infusion. Source: USP <797>.",
    ),
}

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
    "piroxicam":    ["BUN/creatinine q4w", "Urinalysis q4w", "GI symptom check q2w"],
    "toceranib":    ["CBC q7d (first month)", "Chemistry panel q4w", "Blood pressure q4w"],
    "mitoxantrone": ["CBC day 7 post-infusion", "Cardiac echo q3 cycles"],
    "vinblastine":  ["CBC day 7 post-infusion", "Neurologic exam q4w"],
    "carboplatin":  ["BUN/creatinine pre-dose", "CBC day 10–14 post-infusion"],
    "gemcitabine":  ["CBC q7d", "Chemistry panel q4w"],
    "trametinib":   ["Dermatologic exam q2w", "Ophthalmologic exam q4w", "CBC q4w"],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PharmacistDrugEntry:
    drug: str
    final_dose_mg: float
    dose_per_kg: float
    schedule: str
    route: str
    renal_adjustment: str
    hepatic_adjustment: str
    warnings: list[str]
    notes: str
    pk_status: str
    orr_source: Optional[str]
    # Compounding-specific fields
    concentration_mg_ml: Optional[float]
    vehicle: Optional[str]
    quantity_to_dispense: str
    beyond_use_date_days: int
    beyond_use_date: str          # ISO date string
    storage: str
    preparation_notes: str
    label_text: str


@dataclass
class RecipeCard:
    # Identity
    rx_number: str
    pet_name: str
    species: str
    breed: str
    weight_kg: float
    bsa_m2: float
    patient_id: Optional[str]
    prescribing_vet: str
    vet_license: Optional[str]
    clinic_name: Optional[str]
    clinic_address: Optional[str]
    date_issued: str
    refills: int
    # Drugs
    drugs: list[PharmacistDrugEntry]
    # Clinical
    interactions: list[str]
    monitoring: list[str]
    # Outputs
    printable_text: str
    label_text: str               # combined label for all drugs
    pdf_bytes: Optional[bytes]    # PDF binary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _quantity_to_dispense(drug: str, final_dose_mg: float, schedule: str, days: int = 30) -> str:
    """Compute quantity to dispense for a 30-day supply."""
    comp = COMPOUNDING_DATA.get(drug)
    if not comp:
        return "Quantity per prescriber"
    conc, vehicle, bud, _, _ = comp
    route_map = {
        "toceranib": "PO", "piroxicam": "PO", "trametinib": "PO",
        "mitoxantrone": "IV", "vinblastine": "IV", "carboplatin": "IV", "gemcitabine": "IV",
    }
    route = route_map.get(drug, "")
    if route == "PO" and conc:
        # Oral suspension: volume per dose × doses per supply
        doses_per_day = {"q24h": 1, "q48h": 0.5, "q12h": 2}.get(schedule, 1)
        total_doses = int(days * doses_per_day) + 2  # +2 buffer
        vol_per_dose_ml = round(final_dose_mg / conc, 2)
        total_vol_ml = round(vol_per_dose_ml * total_doses, 1)
        return f"{total_vol_ml} mL oral suspension ({vol_per_dose_ml} mL per dose × {total_doses} doses)"
    elif route == "IV":
        return "Per treatment cycle — dispense per administration"
    return "Quantity per prescriber"


def _label_text(
    drug: str,
    pet_name: str,
    breed: str,
    weight_kg: float,
    final_dose_mg: float,
    schedule: str,
    route: str,
    prescribing_vet: str,
    rx_number: str,
    date_issued: str,
    bud: str,
    storage: str,
    warnings: list[str],
) -> str:
    lines = [
        f"Rx# {rx_number}",
        f"Patient: {pet_name} ({breed}, {weight_kg} kg)",
        f"Drug: {drug.upper()}",
        f"Dose: {final_dose_mg} mg  {schedule}  {route}",
        f"Prescriber: {prescribing_vet}",
        f"Date: {date_issued}  |  BUD: {bud}",
        f"Storage: {storage}",
    ]
    if warnings:
        lines.append("WARNINGS: " + "; ".join(warnings))
    lines.append("FOR VETERINARY USE ONLY")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ASCII printable text (backward-compatible)
# ---------------------------------------------------------------------------

def _build_printable(
    rx_number: str,
    pet_name: str,
    species: str,
    breed: str,
    weight_kg: float,
    bsa_m2: float,
    prescribing_vet: str,
    vet_license: Optional[str],
    clinic_name: Optional[str],
    date_issued: str,
    drugs: list[PharmacistDrugEntry],
    interactions: list[str],
    monitoring: list[str],
) -> str:
    lines = [
        "=" * 64,
        "  VETONCO COMPOUNDING PHARMACIST RECIPE CARD",
        "=" * 64,
        f"  Rx#:       {rx_number}",
        f"  Patient:   {pet_name} ({species} — {breed})",
        f"  Weight:    {weight_kg} kg  |  BSA: {bsa_m2} m²",
        f"  Vet:       {prescribing_vet}" + (f"  |  License: {vet_license}" if vet_license else ""),
        f"  Clinic:    {clinic_name or 'Not specified'}",
        f"  Date:      {date_issued}",
        "=" * 64,
        "",
        "PRESCRIBED MEDICATIONS",
        "-" * 64,
    ]
    for d in drugs:
        lines += [
            f"  Drug:          {d.drug.upper()}",
            f"  Final dose:    {d.final_dose_mg} mg  ({d.dose_per_kg} mg/kg)",
            f"  Schedule:      {d.schedule}  |  Route: {d.route}",
            f"  PK status:     {d.pk_status}",
        ]
        if d.renal_adjustment != "none":
            lines.append(f"  Renal adj:     {d.renal_adjustment}")
        if d.hepatic_adjustment != "none":
            lines.append(f"  Hepatic adj:   {d.hepatic_adjustment}")
        if d.warnings:
            for w in d.warnings:
                lines.append(f"  ⚠  {w}")
        lines += [
            "",
            "  COMPOUNDING INSTRUCTIONS:",
            f"  Concentration: {d.concentration_mg_ml} mg/mL" if d.concentration_mg_ml else "  Concentration: N/A (tablet/capsule)",
            f"  Vehicle:       {d.vehicle or 'N/A'}",
            f"  Quantity:      {d.quantity_to_dispense}",
            f"  BUD:           {d.beyond_use_date} ({d.beyond_use_date_days} days)",
            f"  Storage:       {d.storage}",
            f"  Preparation:   {d.preparation_notes}",
            f"  Notes:         {d.notes}",
            "",
        ]

    if interactions:
        lines += ["DRUG INTERACTIONS", "-" * 64]
        for ix in interactions:
            lines.append(f"  ⚠  {ix}")
        lines.append("")

    if monitoring:
        lines += ["MONITORING SCHEDULE", "-" * 64]
        for m in monitoring:
            lines.append(f"  •  {m}")
        lines.append("")

    lines += [
        "ATTESTATION",
        "-" * 64,
        "  I verify that this prescription is accurate, complete, and",
        "  appropriate for the patient named above.",
        "",
        "  Prescribing Vet Signature: _______________________  Date: ________",
        "  Pharmacist Signature:      _______________________  Date: ________",
        "",
        "=" * 64,
        "  FOR VETERINARY USE ONLY — VetOnco Clinical Decision Support",
        "  Verify all doses before dispensing. Not for human use.",
        "=" * 64,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF generator (reportlab)
# ---------------------------------------------------------------------------

def _build_pdf(
    rx_number: str,
    pet_name: str,
    species: str,
    breed: str,
    weight_kg: float,
    bsa_m2: float,
    prescribing_vet: str,
    vet_license: Optional[str],
    clinic_name: Optional[str],
    clinic_address: Optional[str],
    date_issued: str,
    drugs: list[PharmacistDrugEntry],
    interactions: list[str],
    monitoring: list[str],
) -> bytes:
    """Generate a pharmacist-facing PDF using reportlab."""
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, KeepTogether,
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=LETTER,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        styles = getSampleStyleSheet()
        DARK = colors.HexColor("#1a1a2e")
        BLUE = colors.HexColor("#2563eb")
        AMBER = colors.HexColor("#d97706")
        GREEN = colors.HexColor("#059669")
        LIGHT_GRAY = colors.HexColor("#f3f4f6")
        MID_GRAY = colors.HexColor("#6b7280")

        title_style = ParagraphStyle("Title", parent=styles["Heading1"],
            fontSize=16, textColor=DARK, spaceAfter=4, alignment=TA_CENTER)
        subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"],
            fontSize=9, textColor=MID_GRAY, spaceAfter=12, alignment=TA_CENTER)
        section_style = ParagraphStyle("Section", parent=styles["Heading2"],
            fontSize=10, textColor=BLUE, spaceBefore=10, spaceAfter=4,
            borderPad=2)
        body_style = ParagraphStyle("Body", parent=styles["Normal"],
            fontSize=8.5, textColor=DARK, spaceAfter=2, leading=12)
        warn_style = ParagraphStyle("Warn", parent=styles["Normal"],
            fontSize=8, textColor=AMBER, spaceAfter=2)
        small_style = ParagraphStyle("Small", parent=styles["Normal"],
            fontSize=7.5, textColor=MID_GRAY, spaceAfter=2)
        attest_style = ParagraphStyle("Attest", parent=styles["Normal"],
            fontSize=8, textColor=DARK, spaceAfter=6, leading=14)

        story = []

        # ── Header ──────────────────────────────────────────────────────────
        story.append(Paragraph("VetOnco Compounding Pharmacist Recipe Card", title_style))
        story.append(Paragraph(
            f"Rx# {rx_number}  ·  {date_issued}  ·  FOR VETERINARY USE ONLY",
            subtitle_style,
        ))
        story.append(HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceAfter=8))

        # ── Patient / Prescriber block ───────────────────────────────────────
        story.append(Paragraph("Patient & Prescriber", section_style))
        patient_data = [
            ["Patient", f"{pet_name}", "Species", f"{species}"],
            ["Breed", f"{breed}", "Weight", f"{weight_kg} kg"],
            ["BSA", f"{bsa_m2} m²", "Prescribing Vet", f"{prescribing_vet}"],
            ["Vet License", f"{vet_license or '—'}", "Clinic", f"{clinic_name or '—'}"],
            ["Address", f"{clinic_address or '—'}", "Date Issued", f"{date_issued}"],
        ]
        pt = Table(patient_data, colWidths=[1.1*inch, 2.2*inch, 1.1*inch, 2.2*inch])
        pt.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (0, -1), MID_GRAY),
            ("TEXTCOLOR", (2, 0), (2, -1), MID_GRAY),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
            ("FONTNAME", (3, 0), (3, -1), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_GRAY, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(pt)
        story.append(Spacer(1, 10))

        # ── Drug entries ─────────────────────────────────────────────────────
        story.append(Paragraph("Prescribed Medications", section_style))

        for d in drugs:
            drug_block = []

            # Drug header row
            header_data = [[
                Paragraph(f"<b>{d.drug.upper()}</b>", ParagraphStyle("DH",
                    fontSize=9, textColor=colors.white)),
                Paragraph(f"<b>{d.final_dose_mg} mg  ·  {d.schedule}  ·  {d.route}</b>",
                    ParagraphStyle("DH2", fontSize=9, textColor=colors.white)),
                Paragraph(
                    f"<font color='#86efac'>✓ {d.pk_status}</font>" if d.pk_status == "VERIFIED"
                    else f"<font color='#fca5a5'>⚠ {d.pk_status}</font>",
                    ParagraphStyle("DH3", fontSize=8, textColor=colors.white)),
            ]]
            ht = Table(header_data, colWidths=[1.5*inch, 3.5*inch, 1.6*inch])
            ht.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), DARK),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]))
            drug_block.append(ht)

            # Dose details
            dose_rows = [
                ["Dose (mg/kg)", f"{d.dose_per_kg} mg/kg",
                 "Renal adj.", d.renal_adjustment],
                ["Hepatic adj.", d.hepatic_adjustment,
                 "ORR source", d.orr_source or "—"],
            ]
            if d.concentration_mg_ml:
                dose_rows.append([
                    "Concentration", f"{d.concentration_mg_ml} mg/mL",
                    "Vehicle", d.vehicle or "—",
                ])
            dose_rows += [
                ["Quantity", d.quantity_to_dispense, "BUD", d.beyond_use_date],
                ["Storage", d.storage, "", ""],
            ]
            dt = Table(dose_rows, colWidths=[1.1*inch, 2.2*inch, 1.1*inch, 2.2*inch])
            dt.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TEXTCOLOR", (0, 0), (0, -1), MID_GRAY),
                ("TEXTCOLOR", (2, 0), (2, -1), MID_GRAY),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT_GRAY]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("SPAN", (1, len(dose_rows)-1), (3, len(dose_rows)-1)),
            ]))
            drug_block.append(dt)

            # Preparation notes
            drug_block.append(Paragraph(
                f"<b>Preparation:</b> {d.preparation_notes}",
                ParagraphStyle("Prep", fontSize=7.5, textColor=MID_GRAY,
                    spaceAfter=2, spaceBefore=3, leftIndent=6),
            ))

            # Warnings
            for w in d.warnings:
                drug_block.append(Paragraph(f"⚠  {w}", warn_style))

            drug_block.append(Spacer(1, 6))
            story.append(KeepTogether(drug_block))

        # ── Interactions ─────────────────────────────────────────────────────
        if interactions:
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
            story.append(Paragraph("Drug Interactions", section_style))
            for ix in interactions:
                story.append(Paragraph(f"⚠  {ix}", warn_style))
            story.append(Spacer(1, 6))

        # ── Monitoring ───────────────────────────────────────────────────────
        if monitoring:
            story.append(Paragraph("Monitoring Schedule", section_style))
            mon_data = [[f"•  {m}"] for m in monitoring]
            mt = Table(mon_data, colWidths=[6.6*inch])
            mt.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT_GRAY]),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ]))
            story.append(mt)
            story.append(Spacer(1, 10))

        # ── Attestation ──────────────────────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=1, color=DARK))
        story.append(Paragraph("Attestation", section_style))
        story.append(Paragraph(
            "I verify that this prescription is accurate, complete, and appropriate "
            "for the patient named above. All doses have been independently verified "
            "against published canine TCC dosing protocols.",
            attest_style,
        ))
        attest_data = [
            ["Prescribing Vet Signature:", "_" * 30, "Date:", "____________"],
            ["Pharmacist Signature:", "_" * 30, "Date:", "____________"],
        ]
        at = Table(attest_data, colWidths=[1.8*inch, 3.0*inch, 0.5*inch, 1.3*inch])
        at.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(at)
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            "FOR VETERINARY USE ONLY — VetOnco Clinical Decision Support — "
            "Verify all doses before dispensing. Not for human use.",
            ParagraphStyle("Footer", fontSize=7, textColor=MID_GRAY, alignment=TA_CENTER),
        ))

        doc.build(story)
        return buf.getvalue()

    except Exception as e:
        # PDF generation failure is non-fatal — return empty bytes
        return b""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_recipe_card(
    pet_name: str,
    species: str,
    breed: str,
    weight_kg: float,
    prescribing_vet: str,
    dose_results: list[DoseResult],
    date_issued: str | None = None,
    patient_id: str | None = None,
    vet_license: str | None = None,
    clinic_name: str | None = None,
    clinic_address: str | None = None,
    refills: int = 0,
    generate_pdf: bool = True,
) -> RecipeCard:
    """
    Generate a compounding pharmacist recipe card from DoseResult objects.

    Parameters
    ----------
    dose_results    : list of DoseResult from tcc_dosing.compute_canine_dose()
    generate_pdf    : if True, build PDF bytes via reportlab
    """
    if date_issued is None:
        date_issued = date.today().isoformat()

    rx_number = f"VTO-{date_issued.replace('-', '')}-{uuid.uuid4().hex[:4].upper()}"
    bsa_m2 = dose_results[0].bsa_m2 if dose_results else 0.0
    drug_names = [d.drug for d in dose_results]

    # Build PharmacistDrugEntry for each drug
    pharmacist_drugs: list[PharmacistDrugEntry] = []
    for d in dose_results:
        comp = COMPOUNDING_DATA.get(d.drug)
        if comp:
            conc, vehicle, bud_days, storage, prep_notes = comp
        else:
            conc, vehicle, bud_days, storage, prep_notes = None, None, 0, "Per manufacturer", ""

        bud_date = (date.fromisoformat(date_issued) + timedelta(days=bud_days)).isoformat()
        qty = _quantity_to_dispense(d.drug, d.final_dose_mg, d.schedule)

        # Get ORR source from gene panels
        orr_src = None
        try:
            from services.tcc_gene_panels import DRUG_PANEL
            for entry in DRUG_PANEL:
                if entry.name == d.drug:
                    orr_src = entry.orr_source
                    pk_status = entry.pk_status
                    break
            else:
                pk_status = "UNVERIFIED"
        except Exception:
            pk_status = "UNVERIFIED"

        label = _label_text(
            drug=d.drug,
            pet_name=pet_name,
            breed=breed,
            weight_kg=weight_kg,
            final_dose_mg=d.final_dose_mg,
            schedule=d.schedule,
            route=d.route,
            prescribing_vet=prescribing_vet,
            rx_number=rx_number,
            date_issued=date_issued,
            bud=bud_date,
            storage=storage,
            warnings=d.warnings,
        )

        pharmacist_drugs.append(PharmacistDrugEntry(
            drug=d.drug,
            final_dose_mg=d.final_dose_mg,
            dose_per_kg=d.dose_per_kg,
            schedule=d.schedule,
            route=d.route,
            renal_adjustment=d.renal_adjustment,
            hepatic_adjustment=d.hepatic_adjustment,
            warnings=d.warnings,
            notes=d.notes,
            pk_status=pk_status,
            orr_source=orr_src,
            concentration_mg_ml=conc,
            vehicle=vehicle,
            quantity_to_dispense=qty,
            beyond_use_date_days=bud_days,
            beyond_use_date=bud_date,
            storage=storage,
            preparation_notes=prep_notes,
            label_text=label,
        ))

    interactions = _detect_interactions(drug_names)
    monitoring = _build_monitoring(drug_names)

    printable = _build_printable(
        rx_number=rx_number,
        pet_name=pet_name,
        species=species,
        breed=breed,
        weight_kg=weight_kg,
        bsa_m2=bsa_m2,
        prescribing_vet=prescribing_vet,
        vet_license=vet_license,
        clinic_name=clinic_name,
        date_issued=date_issued,
        drugs=pharmacist_drugs,
        interactions=interactions,
        monitoring=monitoring,
    )

    combined_label = "\n\n---\n\n".join(d.label_text for d in pharmacist_drugs)

    pdf_bytes = None
    if generate_pdf:
        pdf_bytes = _build_pdf(
            rx_number=rx_number,
            pet_name=pet_name,
            species=species,
            breed=breed,
            weight_kg=weight_kg,
            bsa_m2=bsa_m2,
            prescribing_vet=prescribing_vet,
            vet_license=vet_license,
            clinic_name=clinic_name,
            clinic_address=clinic_address,
            date_issued=date_issued,
            drugs=pharmacist_drugs,
            interactions=interactions,
            monitoring=monitoring,
        )

    return RecipeCard(
        rx_number=rx_number,
        pet_name=pet_name,
        species=species,
        breed=breed,
        weight_kg=weight_kg,
        bsa_m2=bsa_m2,
        patient_id=patient_id,
        prescribing_vet=prescribing_vet,
        vet_license=vet_license,
        clinic_name=clinic_name,
        clinic_address=clinic_address,
        date_issued=date_issued,
        refills=refills,
        drugs=pharmacist_drugs,
        interactions=interactions,
        monitoring=monitoring,
        printable_text=printable,
        label_text=combined_label,
        pdf_bytes=pdf_bytes,
    )
