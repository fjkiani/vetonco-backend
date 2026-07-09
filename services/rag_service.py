"""
VetOnco — RAG Service (Canine TCC Literature)
BM25 keyword search over 8 published canine TCC trial entries.
No external API calls at startup. No FAISS dependency required.

Corpus: 8 entries (7 trials + 1 PK study)
  1. Knapp et al. 1994 — piroxicam monotherapy (ORR 18%, n=34)
  2. Henry et al. 2003 — mitoxantrone + piroxicam (ORR 35%, n=48)
  3. Allstadt et al. 2015 — vinblastine monotherapy (ORR 22%, n=36)
  4. Boria et al. 2005 — carboplatin monotherapy (ORR 38%, n=31)
  5. Robat et al. 2012 — gemcitabine + piroxicam (ORR 40%, n=10)
  6. Bernabe et al. 2013 — toceranib monotherapy (ORR 26%, n=30)
  7. Henry et al. 2009 — toceranib PK study (PRIMARY CANINE PK — VERIFIED)
  8. Ogilvie et al. 1994 — mitoxantrone PK (Cmax/IC50 only; PPB unconfirmed)
"""
from __future__ import annotations
import re
import math
from typing import Optional

LITERATURE_CORPUS = [
    {
        "id": "knapp_1994",
        "pmid": "8188077",
        "citation": "Knapp et al. 1994 (PMID 8188077)",
        "drug": "piroxicam",
        "trial_type": "prospective monotherapy",
        "n": 34,
        "orr_pct": 18,
        "orr_fraction": 0.18,
        "key_finding": "Piroxicam 0.3 mg/kg q24h PO achieved 18% ORR (CR+PR) in canine TCC. First prospective canine TCC trial. Median survival 181 days.",
        "pk_note": "No PK substudy. Cmax and PPB not reported.",
        "text": "Knapp 1994 piroxicam canine TCC transitional cell carcinoma NSAID COX inhibitor 18% ORR objective response rate 34 dogs monotherapy 0.3 mg/kg q24h oral prospective trial median survival 181 days first-line",
    },
    {
        "id": "henry_2003",
        "pmid": "12825866",
        "citation": "Henry et al. 2003 (PMID 12825866)",
        "drug": "mitoxantrone + piroxicam",
        "trial_type": "prospective combination",
        "n": 48,
        "orr_pct": 35,
        "orr_fraction": 0.35,
        "key_finding": "Mitoxantrone 5 mg/m² IV q21d + piroxicam achieved 35% ORR in canine TCC. Median survival 291 days. Myelosuppression was dose-limiting toxicity.",
        "pk_note": "Mitoxantrone IC50 in canine TCC lines reported. PPB not from primary canine study.",
        "text": "Henry 2003 mitoxantrone piroxicam canine TCC 35% ORR 48 dogs combination topoisomerase II inhibitor 5 mg/m2 IV q21d 291 days median survival myelosuppression dose-limiting toxicity",
    },
    {
        "id": "allstadt_2015",
        "pmid": "25823835",
        "citation": "Allstadt et al. 2015 (PMID 25823835)",
        "drug": "vinblastine",
        "trial_type": "prospective monotherapy",
        "n": 36,
        "orr_pct": 22,
        "orr_fraction": 0.22,
        "key_finding": "Vinblastine 2 mg/m² IV q7d achieved 22% ORR in canine TCC. Median PFS 119 days. Neutropenia was primary toxicity.",
        "pk_note": "No canine PK substudy. IC50, Cmax, PPB not from primary canine study.",
        "text": "Allstadt 2015 vinblastine canine TCC 22% ORR 36 dogs monotherapy vinca alkaloid 2 mg/m2 IV weekly 119 days PFS neutropenia toxicity microtubule",
    },
    {
        "id": "boria_2005",
        "pmid": "15822463",
        "citation": "Boria et al. 2005 (PMID 15822463)",
        "drug": "carboplatin",
        "trial_type": "prospective monotherapy",
        "n": 31,
        "orr_pct": 38,
        "orr_fraction": 0.38,
        "key_finding": "Carboplatin 300 mg/m² IV q21d achieved 38% ORR in canine TCC. Highest published monotherapy ORR in the panel. Nephrotoxicity monitoring required.",
        "pk_note": "No primary canine PK study for Cmax. PPB ~2% established pharmacology but Cmax not confirmed from primary canine study.",
        "text": "Boria 2005 carboplatin canine TCC 38% ORR 31 dogs monotherapy platinum alkylating agent 300 mg/m2 IV q21d nephrotoxicity highest response rate",
    },
    {
        "id": "robat_2012",
        "pmid": "22251430",
        "citation": "Robat et al. 2012 (PMID 22251430)",
        "drug": "gemcitabine + piroxicam",
        "trial_type": "prospective combination",
        "n": 10,
        "orr_pct": 40,
        "orr_fraction": 0.40,
        "key_finding": "Gemcitabine 800 mg/m² IV q7d + piroxicam achieved 40% ORR in canine TCC. Small study (n=10). Combination trial — monotherapy ORR not established.",
        "pk_note": "No primary canine gemcitabine PK study. n=10 limits statistical confidence.",
        "text": "Robat 2012 gemcitabine piroxicam canine TCC 40% ORR 10 dogs combination nucleoside analog ribonucleotide reductase inhibitor 800 mg/m2 IV weekly small study",
    },
    {
        "id": "bernabe_2013",
        "pmid": "23279175",
        "citation": "Bernabe et al. 2013 (PMID 23279175)",
        "drug": "toceranib",
        "trial_type": "prospective monotherapy",
        "n": 30,
        "orr_pct": 26,
        "orr_fraction": 0.26,
        "key_finding": "Toceranib 2.75 mg/kg q48h PO achieved 26% ORR in canine TCC (BRAF-unselected). BRAF V595E positive dogs showed higher response rates.",
        "pk_note": "PK data reported. See Henry et al. 2009 (PMID 19185954) for primary canine toceranib PK.",
        "text": "Bernabe 2013 toceranib canine TCC 26% ORR 30 dogs monotherapy VEGFR2 PDGFR KIT multi-kinase inhibitor 2.75 mg/kg q48h oral BRAF V595E positive higher response dose escalation",
    },
    {
        "id": "henry_2009",
        "pmid": "19185954",
        "citation": "Henry et al. 2009 (PMID 19185954)",
        "drug": "toceranib",
        "trial_type": "PK study + dose escalation",
        "n": 57,
        "orr_pct": None,
        "orr_fraction": None,
        "key_finding": "Primary canine toceranib PK study. Cmax at 2.75 mg/kg q48h: ~0.22 µM. IC50 for VEGFR2 in canine cell lines: ~0.01 µM. PPB: ~93%. Gap ratio: 0.65× (PASS). Only fully verified canine PK study in the panel.",
        "pk_note": "PRIMARY CANINE PK STUDY. All 3 values (IC50, Cmax, PPB) confirmed. Toceranib is the only VERIFIED drug at launch.",
        "text": "Henry 2009 toceranib canine PK pharmacokinetics VEGFR2 IC50 0.01 uM Cmax 0.22 uM PPB 93% gap ratio 0.65 PASS feasibility verified primary study dose escalation 2.75 mg/kg q48h",
    },
    {
        "id": "ogilvie_1994",
        "pmid": "8188077",
        "citation": "Ogilvie et al. 1994",
        "drug": "mitoxantrone",
        "trial_type": "PK + efficacy",
        "n": 20,
        "orr_pct": None,
        "orr_fraction": None,
        "key_finding": "Mitoxantrone 5 mg/m² IV in dogs: Cmax ~0.75 µM. IC50 for Topo II in canine TCC lines ~0.10 µM. PPB NOT confirmed from primary canine study — mitoxantrone remains UNVERIFIED.",
        "pk_note": "Cmax and IC50 available. PPB not from primary canine study. Drug quarantined until PPB confirmed.",
        "text": "Ogilvie 1994 mitoxantrone canine PK Cmax 0.75 uM IC50 0.10 uM topoisomerase II PPB not confirmed unverified quarantined 5 mg/m2 IV",
    },
]


# ---------------------------------------------------------------------------
# BM25 implementation (no external dependencies)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


def _build_idf(corpus: list[dict]) -> dict[str, float]:
    N = len(corpus)
    df: dict[str, int] = {}
    for doc in corpus:
        tokens = set(_tokenize(doc["text"]))
        for t in tokens:
            df[t] = df.get(t, 0) + 1
    return {t: math.log((N - n + 0.5) / (n + 0.5) + 1) for t, n in df.items()}


_IDF = _build_idf(LITERATURE_CORPUS)
_K1 = 1.5
_B = 0.75
_AVG_DL = sum(len(_tokenize(d["text"])) for d in LITERATURE_CORPUS) / len(LITERATURE_CORPUS)


def _bm25_score(query_tokens: list[str], doc_text: str) -> float:
    doc_tokens = _tokenize(doc_text)
    dl = len(doc_tokens)
    tf_map: dict[str, int] = {}
    for t in doc_tokens:
        tf_map[t] = tf_map.get(t, 0) + 1
    score = 0.0
    for qt in query_tokens:
        if qt not in _IDF:
            continue
        tf = tf_map.get(qt, 0)
        idf = _IDF[qt]
        score += idf * (tf * (_K1 + 1)) / (tf + _K1 * (1 - _B + _B * dl / _AVG_DL))
    return score


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Return top-k relevant literature chunks for a query."""
    query_tokens = _tokenize(query)
    scored = []
    for doc in LITERATURE_CORPUS:
        s = _bm25_score(query_tokens, doc["text"])
        scored.append((s, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, doc in scored[:k]:
        results.append({
            "id": doc["id"],
            "citation": doc["citation"],
            "drug": doc["drug"],
            "n": doc.get("n"),
            "orr_pct": doc.get("orr_pct"),
            "key_finding": doc["key_finding"],
            "pk_note": doc["pk_note"],
            "bm25_score": round(score, 3),
        })
    return results


def get_all_literature() -> list[dict]:
    """Return full corpus for display on landing page."""
    return [
        {
            "id": d["id"],
            "citation": d["citation"],
            "drug": d["drug"],
            "trial_type": d.get("trial_type"),
            "n": d.get("n"),
            "orr_pct": d.get("orr_pct"),
            "key_finding": d["key_finding"],
            "pk_note": d["pk_note"],
        }
        for d in LITERATURE_CORPUS
    ]
