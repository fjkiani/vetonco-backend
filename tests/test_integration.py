"""
W4 — Integration tests: 3 canonical patients end-to-end
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from services.tcc_scorer import score_tcc
from services.tcc_dosing import compute_canine_dose
from services.recipe_generator import generate_recipe_card, PharmacistDrugEntry

class TestAngus:
    def setup_method(self):
        self.score_result = score_tcc(braf_status="positive", breed="scottish terrier")
        self.dose = compute_canine_dose("toceranib", 12.0, creatinine_mg_dl=1.8)
        self.card = generate_recipe_card(
            pet_name="Angus", species="Canis lupus familiaris", breed="Scottish Terrier",
            weight_kg=12.0, prescribing_vet="Dr. Smith DVM DACVIM", dose_results=[self.dose],
            patient_id="test-angus", vet_license="VT-12345",
            clinic_name="Canine Oncology Center", generate_pdf=True,
        )
    def test_score_toceranib_rank1(self):
        top = self.score_result.recommendations[0]
        assert top.drug == "toceranib" and top.rank == 1 and top.score == 0.380
    def test_score_subtype(self):
        assert self.score_result.subtype == "BRAF-mutant TCC"
    def test_gate_pass(self):
        top = self.score_result.recommendations[0]
        assert top.feasibility_verdict == "PASS" and top.gap_ratio == 0.6494
    def test_dose_toceranib_no_renal(self):
        assert self.dose.renal_adjustment == "none" and self.dose.final_dose_mg == 33.0
    def test_recipe_schema(self):
        d = self.card.drugs[0]
        assert isinstance(d, PharmacistDrugEntry)
        assert d.drug == "toceranib" and d.final_dose_mg == 33.0
        assert d.concentration_mg_ml == 10.0 and d.vehicle == "Ora-Plus/Ora-Sweet 1:1"
        assert d.beyond_use_date_days == 60 and "Refrigerate" in d.storage
        assert d.pk_status == "VERIFIED" and "Bernabe" in (d.orr_source or "")
    def test_recipe_pdf_magic_bytes(self):
        assert self.card.pdf_bytes and self.card.pdf_bytes[:4] == b"%PDF"
        assert len(self.card.pdf_bytes) > 1000
    def test_recipe_label_text(self):
        assert "TOCERANIB" in self.card.label_text.upper()
        assert "33.0 mg" in self.card.label_text
        assert "FOR VETERINARY USE ONLY" in self.card.label_text
    def test_recipe_attestation(self):
        pt = self.card.printable_text
        assert "ATTESTATION" in pt and "Prescribing Vet Signature" in pt
    def test_recipe_bsa(self):
        assert self.card.bsa_m2 == 0.5298

class TestBella:
    def setup_method(self):
        self.score_result = score_tcc(braf_status="unknown", breed="other")
        self.dose = compute_canine_dose("toceranib", 25.0, creatinine_mg_dl=0.9)
        self.card = generate_recipe_card(
            pet_name="Bella", species="Canis lupus familiaris", breed="Labrador Retriever",
            weight_kg=25.0, prescribing_vet="Dr. Jones DVM", dose_results=[self.dose], generate_pdf=True,
        )
    def test_score(self):
        top = self.score_result.recommendations[0]
        assert top.drug == "toceranib" and top.score == 0.310
    def test_subtype_nos(self):
        assert self.score_result.subtype == "TCC-NOS"
    def test_dose_25kg(self):
        assert self.dose.dose_mg == 68.75 and self.dose.final_dose_mg == 68.75
    def test_pdf(self):
        assert self.card.pdf_bytes and self.card.pdf_bytes[:4] == b"%PDF"

class TestMaisie:
    def setup_method(self):
        self.score_result = score_tcc(braf_status="negative", breed="beagle", msh2_loss=True)
        self.dose = compute_canine_dose("toceranib", 8.0, creatinine_mg_dl=2.2)
        self.card = generate_recipe_card(
            pet_name="Maisie", species="Canis lupus familiaris", breed="Beagle",
            weight_kg=8.0, prescribing_vet="Dr. Lee DVM DACVIM", dose_results=[self.dose], generate_pdf=True,
        )
    def test_subtype_mmr(self):
        assert self.score_result.subtype == "MMR-deficient TCC"
    def test_score_braf_negative(self):
        top = self.score_result.recommendations[0]
        assert top.drug == "toceranib" and top.score == 0.260
    def test_dose_8kg(self):
        assert self.dose.dose_mg == 22.0 and self.dose.renal_adjustment == "none"
    def test_pdf(self):
        assert self.card.pdf_bytes and self.card.pdf_bytes[:4] == b"%PDF"

def test_pharmacist_drug_entry_schema():
    dose = compute_canine_dose("toceranib", 12.0)
    card = generate_recipe_card(
        pet_name="Test", species="Canis lupus familiaris", breed="Mixed",
        weight_kg=12.0, prescribing_vet="Dr. Test", dose_results=[dose], generate_pdf=False,
    )
    d = card.drugs[0]
    for field in ["drug","final_dose_mg","dose_per_kg","schedule","route",
                  "renal_adjustment","hepatic_adjustment","pk_status",
                  "concentration_mg_ml","vehicle","quantity_to_dispense",
                  "beyond_use_date_days","beyond_use_date","storage",
                  "preparation_notes","label_text"]:
        val = getattr(d, field, None)
        assert val is not None, f"PharmacistDrugEntry.{field} is None"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
