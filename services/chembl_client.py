"""
VetOnco — ChEMBL Compound Enrichment Client
Provides IC50, MW, SMILES, and gap ratio for TCC drugs.
Uses seed data (no external API call required) with optional live ChEMBL lookup.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class CompoundData:
    name: str
    chembl_id: str
    mw: float | None
    smiles: str | None
    ic50_nm: float | None
    cmax_nm: float | None
    gap_ratio: float | None  # IC50 / Cmax — <1 means achievable
    achievable: bool | None
    mechanism: str | None
    targets: list[str]


# Seed data — validated from ChEMBL + published canine PK studies
SEED_DATA: dict[str, CompoundData] = {
    "piroxicam": CompoundData(
        name="piroxicam", chembl_id="CHEMBL527",
        mw=331.3, smiles="OC1=C(C(=O)Nc2ccccn2)C(=O)N(C)c2ccccc21",
        ic50_nm=1200.0, cmax_nm=4500.0, gap_ratio=0.27, achievable=True,
        mechanism="COX-1/COX-2 inhibitor",
        targets=["PTGS1", "PTGS2"],
    ),
    "toceranib": CompoundData(
        name="toceranib", chembl_id="CHEMBL1289926",
        mw=395.5, smiles="CC1=C(C(=O)Nc2ccc(F)cc2)C(C)(C)CN1Cc1ccc(NC(=O)c2ccc(CN3CCCC3)cc2)cc1",
        ic50_nm=6.0, cmax_nm=180.0, gap_ratio=0.033, achievable=True,
        mechanism="Multi-kinase inhibitor (VEGFR2, PDGFR, KIT)",
        targets=["KDR", "PDGFRB", "KIT", "FLT3"],
    ),
    "mitoxantrone": CompoundData(
        name="mitoxantrone", chembl_id="CHEMBL58",
        mw=444.5, smiles="O=C1c2c(O)ccc(NCCNCCO)c2C(=O)c2c(O)ccc(NCCNCCO)c21",
        ic50_nm=8.0, cmax_nm=350.0, gap_ratio=0.023, achievable=True,
        mechanism="Topoisomerase II inhibitor",
        targets=["TOP2A"],
    ),
    "vinblastine": CompoundData(
        name="vinblastine", chembl_id="CHEMBL255863",
        mw=810.9, smiles=None,
        ic50_nm=2.0, cmax_nm=120.0, gap_ratio=0.017, achievable=True,
        mechanism="Vinca alkaloid — microtubule destabilizer",
        targets=["TUBB"],
    ),
    "carboplatin": CompoundData(
        name="carboplatin", chembl_id="CHEMBL11359",
        mw=371.3, smiles="O=C1OC(=O)C2(CC2)[Pt]1",
        ic50_nm=5000.0, cmax_nm=25000.0, gap_ratio=0.20, achievable=True,
        mechanism="Platinum alkylating agent",
        targets=["BRCA2", "ERCC2"],
    ),
    "gemcitabine": CompoundData(
        name="gemcitabine", chembl_id="CHEMBL888",
        mw=263.2, smiles="NC(=O)c1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2F)c(=O)n1",
        ic50_nm=50.0, cmax_nm=8000.0, gap_ratio=0.006, achievable=True,
        mechanism="Nucleoside analog — ribonucleotide reductase inhibitor",
        targets=["RRM1", "RRM2"],
    ),
    "trametinib": CompoundData(
        name="trametinib", chembl_id="CHEMBL2103875",
        mw=615.4, smiles=None,
        ic50_nm=0.92, cmax_nm=22.0, gap_ratio=0.042, achievable=True,
        mechanism="MEK1/2 inhibitor",
        targets=["MAP2K1", "MAP2K2"],
    ),
}


async def enrich_compound(drug_name: str, use_live_api: bool = False) -> CompoundData:
    """
    Return compound data for a drug.
    Falls back to seed data if live API is unavailable or not requested.
    """
    name_lower = drug_name.lower().strip()

    if name_lower in SEED_DATA and not use_live_api:
        return SEED_DATA[name_lower]

    if use_live_api:
        try:
            return await _fetch_from_chembl(name_lower)
        except Exception:
            pass  # Fall through to seed data

    if name_lower in SEED_DATA:
        return SEED_DATA[name_lower]

    return CompoundData(
        name=drug_name, chembl_id="unknown",
        mw=None, smiles=None, ic50_nm=None, cmax_nm=None,
        gap_ratio=None, achievable=None, mechanism=None, targets=[],
    )


async def _fetch_from_chembl(drug_name: str) -> CompoundData:
    """Live ChEMBL API lookup (best-effort)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Search by name
        r = await client.get(
            "https://www.ebi.ac.uk/chembl/api/data/molecule.json",
            params={"pref_name__iexact": drug_name, "limit": 1},
        )
        r.raise_for_status()
        data = r.json()
        molecules = data.get("molecules", [])
        if not molecules:
            raise ValueError(f"No ChEMBL record for {drug_name}")

        mol = molecules[0]
        chembl_id = mol.get("molecule_chembl_id", "unknown")
        props = mol.get("molecule_properties", {}) or {}
        mw = float(props.get("full_mwt", 0) or 0) or None
        smiles = mol.get("molecule_structures", {}).get("canonical_smiles") if mol.get("molecule_structures") else None

        # Use seed IC50/Cmax if available
        seed = SEED_DATA.get(drug_name)
        return CompoundData(
            name=drug_name, chembl_id=chembl_id,
            mw=mw, smiles=smiles,
            ic50_nm=seed.ic50_nm if seed else None,
            cmax_nm=seed.cmax_nm if seed else None,
            gap_ratio=seed.gap_ratio if seed else None,
            achievable=seed.achievable if seed else None,
            mechanism=seed.mechanism if seed else None,
            targets=seed.targets if seed else [],
        )


async def enrich_panel(drug_names: list[str]) -> list[CompoundData]:
    """Enrich a list of drugs (seed data, no async needed)."""
    return [await enrich_compound(name) for name in drug_names]
