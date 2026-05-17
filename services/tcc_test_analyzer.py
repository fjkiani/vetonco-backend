"""
VetOnco — TCC Test Analyzer
VCOG-CTCAE v1.1 grading for CBC, chemistry, urinalysis, BRAF urine, imaging.
Reference: Veterinary Cooperative Oncology Group CTCAE v1.1 (2011)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal

TestType = Literal["cbc", "chemistry", "urinalysis", "braf_urine", "imaging"]
Grade = Literal[0, 1, 2, 3, 4, 5]


@dataclass
class TestFinding:
    parameter: str
    value: Any
    unit: str
    grade: Grade
    interpretation: str
    action: str


@dataclass
class TestAnalysisResult:
    test_type: TestType
    pet_id: str
    findings: list[TestFinding]
    alerts: list[str]
    overall_grade: Grade
    clinical_action: str
    braf_positive: bool | None = None  # only for braf_urine
    imaging_response: str | None = None  # only for imaging


# ---------------------------------------------------------------------------
# CBC grading (VCOG-CTCAE v1.1)
# ---------------------------------------------------------------------------

def _grade_neutrophils(val: float) -> tuple[Grade, str, str]:
    """ANC in ×10³/µL"""
    if val >= 3.0:
        return 0, "Normal", "Continue therapy"
    if val >= 2.0:
        return 1, "Mild neutropenia", "Monitor; continue therapy"
    if val >= 1.0:
        return 2, "Moderate neutropenia", "Delay cytotoxic therapy; recheck in 5-7 days"
    if val >= 0.5:
        return 3, "Severe neutropenia", "HOLD toceranib/cytotoxics; prophylactic antibiotics; recheck in 3-5 days"
    return 4, "Life-threatening neutropenia", "EMERGENCY — hospitalize; IV antibiotics; G-CSF consider"


def _grade_platelets(val: float) -> tuple[Grade, str, str]:
    """Platelets in ×10³/µL"""
    if val >= 200:
        return 0, "Normal", "Continue therapy"
    if val >= 100:
        return 1, "Mild thrombocytopenia", "Monitor; continue therapy"
    if val >= 50:
        return 2, "Moderate thrombocytopenia", "Delay cytotoxic therapy"
    if val >= 25:
        return 3, "Severe thrombocytopenia", "HOLD all cytotoxics; bleeding precautions"
    return 4, "Life-threatening thrombocytopenia", "EMERGENCY — transfusion consider; hospitalize"


def _grade_hematocrit(val: float) -> tuple[Grade, str, str]:
    """Hematocrit in %"""
    if val >= 37:
        return 0, "Normal", "Continue therapy"
    if val >= 30:
        return 1, "Mild anemia", "Monitor; iron supplementation consider"
    if val >= 20:
        return 2, "Moderate anemia", "Delay cytotoxic therapy; supportive care"
    if val >= 14:
        return 3, "Severe anemia", "HOLD cytotoxics; transfusion consider"
    return 4, "Life-threatening anemia", "EMERGENCY — transfusion required"


def analyze_cbc(values: dict[str, float], pet_id: str) -> TestAnalysisResult:
    findings: list[TestFinding] = []
    alerts: list[str] = []
    max_grade: Grade = 0

    if "anc" in values:
        g, interp, action = _grade_neutrophils(values["anc"])
        findings.append(TestFinding("ANC", values["anc"], "×10³/µL", g, interp, action))
        if g >= 3:
            alerts.append(f"Grade {g} neutropenia — {action}")
        max_grade = max(max_grade, g)

    if "platelets" in values:
        g, interp, action = _grade_platelets(values["platelets"])
        findings.append(TestFinding("Platelets", values["platelets"], "×10³/µL", g, interp, action))
        if g >= 2:
            alerts.append(f"Grade {g} thrombocytopenia — {action}")
        max_grade = max(max_grade, g)

    if "hematocrit" in values:
        g, interp, action = _grade_hematocrit(values["hematocrit"])
        findings.append(TestFinding("Hematocrit", values["hematocrit"], "%", g, interp, action))
        if g >= 2:
            alerts.append(f"Grade {g} anemia — {action}")
        max_grade = max(max_grade, g)

    overall_action = "Continue current protocol" if max_grade <= 1 else alerts[0] if alerts else "Review findings"

    return TestAnalysisResult(
        test_type="cbc",
        pet_id=pet_id,
        findings=findings,
        alerts=alerts,
        overall_grade=max_grade,
        clinical_action=overall_action,
    )


# ---------------------------------------------------------------------------
# Chemistry grading
# ---------------------------------------------------------------------------

def _grade_creatinine(val: float) -> tuple[Grade, str, str]:
    """Creatinine in mg/dL"""
    if val <= 1.4:
        return 0, "Normal", "Continue therapy"
    if val <= 2.0:
        return 1, "Mild azotemia", "Increase hydration; recheck in 2 weeks"
    if val <= 3.0:
        return 2, "Moderate azotemia", "Reduce nephrotoxic drugs 25%; recheck in 1 week"
    if val <= 5.0:
        return 3, "Severe azotemia", "HOLD carboplatin/piroxicam; IV fluids; nephrology consult"
    return 4, "Life-threatening renal failure", "EMERGENCY — hospitalize; dialysis consider"


def _grade_alt(val: float) -> tuple[Grade, str, str]:
    """ALT in U/L (reference ~10-100 U/L)"""
    if val <= 100:
        return 0, "Normal", "Continue therapy"
    if val <= 250:
        return 1, "Mild hepatotoxicity", "Monitor; recheck in 2 weeks"
    if val <= 500:
        return 2, "Moderate hepatotoxicity", "Reduce hepatically-cleared drugs 25%"
    if val <= 1000:
        return 3, "Severe hepatotoxicity", "HOLD mitoxantrone/vinblastine/trametinib; hepatology consult"
    return 4, "Life-threatening hepatotoxicity", "EMERGENCY — hospitalize; discontinue all hepatotoxic drugs"


def analyze_chemistry(values: dict[str, float], pet_id: str) -> TestAnalysisResult:
    findings: list[TestFinding] = []
    alerts: list[str] = []
    max_grade: Grade = 0

    if "creatinine" in values:
        g, interp, action = _grade_creatinine(values["creatinine"])
        findings.append(TestFinding("Creatinine", values["creatinine"], "mg/dL", g, interp, action))
        if g >= 2:
            alerts.append(f"Grade {g} azotemia — {action}")
        max_grade = max(max_grade, g)

    if "alt" in values:
        g, interp, action = _grade_alt(values["alt"])
        findings.append(TestFinding("ALT", values["alt"], "U/L", g, interp, action))
        if g >= 2:
            alerts.append(f"Grade {g} hepatotoxicity — {action}")
        max_grade = max(max_grade, g)

    if "bun" in values:
        bun = values["bun"]
        if bun > 60:
            g = 2 if bun <= 100 else 3
            findings.append(TestFinding("BUN", bun, "mg/dL", g, f"Elevated BUN ({bun})", "Increase hydration; recheck"))
            alerts.append(f"Elevated BUN {bun} mg/dL")
            max_grade = max(max_grade, g)
        else:
            findings.append(TestFinding("BUN", bun, "mg/dL", 0, "Normal", "Continue therapy"))

    overall_action = "Continue current protocol" if max_grade <= 1 else alerts[0] if alerts else "Review findings"

    return TestAnalysisResult(
        test_type="chemistry",
        pet_id=pet_id,
        findings=findings,
        alerts=alerts,
        overall_grade=max_grade,
        clinical_action=overall_action,
    )


# ---------------------------------------------------------------------------
# Urinalysis
# ---------------------------------------------------------------------------

def analyze_urinalysis(values: dict[str, Any], pet_id: str) -> TestAnalysisResult:
    findings: list[TestFinding] = []
    alerts: list[str] = []
    max_grade: Grade = 0

    usg = values.get("usg", 1.025)
    if usg < 1.008:
        g = 2
        findings.append(TestFinding("USG", usg, "", g, "Isosthenuria — renal concentrating defect", "Renal function panel; hydration assessment"))
        alerts.append(f"Isosthenuria (USG {usg}) — possible renal dysfunction")
        max_grade = max(max_grade, g)
    else:
        findings.append(TestFinding("USG", usg, "", 0, "Adequate concentration", "Continue therapy"))

    protein = values.get("protein", "negative")
    if protein in ("3+", "4+"):
        g = 2
        findings.append(TestFinding("Protein", protein, "", g, "Significant proteinuria", "UPC ratio; nephrology consult"))
        alerts.append(f"Significant proteinuria ({protein}) — monitor UPC ratio")
        max_grade = max(max_grade, g)
    elif protein in ("1+", "2+"):
        findings.append(TestFinding("Protein", protein, "", 1, "Mild proteinuria", "Recheck in 4 weeks"))

    blood = values.get("blood", "negative")
    if blood in ("2+", "3+", "4+"):
        g = 2
        findings.append(TestFinding("Blood", blood, "", g, "Hematuria — TCC progression possible", "Cystoscopy or imaging; cytology"))
        alerts.append(f"Hematuria ({blood}) — evaluate for TCC progression")
        max_grade = max(max_grade, g)

    overall_action = "Continue current protocol" if max_grade <= 1 else alerts[0] if alerts else "Review findings"

    return TestAnalysisResult(
        test_type="urinalysis",
        pet_id=pet_id,
        findings=findings,
        alerts=alerts,
        overall_grade=max_grade,
        clinical_action=overall_action,
    )


# ---------------------------------------------------------------------------
# BRAF urine test
# ---------------------------------------------------------------------------

def analyze_braf_urine(values: dict[str, Any], pet_id: str) -> TestAnalysisResult:
    positive = values.get("braf_positive", False)
    mutation = values.get("mutation", "V595E")
    vaf = values.get("vaf", None)

    if positive:
        finding = TestFinding(
            "BRAF urine", f"POSITIVE ({mutation})", "",
            1, f"BRAF {mutation} detected in urine ctDNA",
            "Prioritize toceranib + trametinib in treatment plan; confirm with tissue biopsy if not done",
        )
        alerts = [f"BRAF {mutation} positive — targeted therapy indicated"]
        action = "Prioritize BRAF-targeted agents (toceranib, trametinib)"
    else:
        finding = TestFinding(
            "BRAF urine", "NEGATIVE", "",
            0, "No BRAF V595E detected in urine ctDNA",
            "BRAF-targeted therapy not indicated; proceed with BRAF-agnostic protocol",
        )
        alerts = []
        action = "BRAF-agnostic protocol (piroxicam ± mitoxantrone/vinblastine)"

    if vaf is not None:
        finding.value += f" | VAF {vaf:.1%}"

    return TestAnalysisResult(
        test_type="braf_urine",
        pet_id=pet_id,
        findings=[finding],
        alerts=alerts,
        overall_grade=1 if positive else 0,
        clinical_action=action,
        braf_positive=positive,
    )


# ---------------------------------------------------------------------------
# Imaging response (RECIST-like for veterinary oncology)
# ---------------------------------------------------------------------------

def analyze_imaging(values: dict[str, Any], pet_id: str) -> TestAnalysisResult:
    response = values.get("response", "stable_disease")
    tumor_size_cm = values.get("tumor_size_cm", None)
    change_pct = values.get("change_pct", 0.0)

    response_map = {
        "complete_response": (0, "Complete response — no measurable disease", "Continue current protocol; consider maintenance"),
        "partial_response": (0, f"Partial response ({change_pct:+.0f}%)", "Continue current protocol"),
        "stable_disease": (1, f"Stable disease ({change_pct:+.0f}%)", "Continue current protocol; reassess in 4-6 weeks"),
        "progressive_disease": (3, f"Progressive disease ({change_pct:+.0f}%)", "SWITCH protocol — consider second-line agents"),
    }

    grade, interp, action = response_map.get(response, (1, "Unknown response", "Clinical reassessment"))

    finding = TestFinding(
        "Imaging response",
        f"{response.replace('_', ' ').title()}" + (f" | {tumor_size_cm} cm" if tumor_size_cm else ""),
        "",
        grade, interp, action,
    )

    alerts = [f"Progressive disease — protocol change indicated"] if response == "progressive_disease" else []

    return TestAnalysisResult(
        test_type="imaging",
        pet_id=pet_id,
        findings=[finding],
        alerts=alerts,
        overall_grade=grade,
        clinical_action=action,
        imaging_response=response,
    )


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def analyze_test_results(
    test_type: TestType,
    values: dict[str, Any],
    pet_id: str,
) -> TestAnalysisResult:
    """Dispatch to the appropriate analyzer based on test_type."""
    dispatch = {
        "cbc": analyze_cbc,
        "chemistry": analyze_chemistry,
        "urinalysis": analyze_urinalysis,
        "braf_urine": analyze_braf_urine,
        "imaging": analyze_imaging,
    }
    fn = dispatch.get(test_type)
    if fn is None:
        raise ValueError(f"Unknown test_type: {test_type}")
    return fn(values, pet_id)
