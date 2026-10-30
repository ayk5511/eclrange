"""Shared paths, file layouts and constants for Paper 10.

Raw Freddie Mac files and every loan-level derivative live under GSE_ROOT,
outside the repository: the dataset terms allow publishing results and
derived products that cannot recreate the data, never the loan records.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GSE_ROOT = Path(os.environ.get("GSE_ROOT", REPO / "data" / "gse"))
ZIPS = GSE_ROOT / "freddie" / "zips"
DERIVED = GSE_ROOT / "freddie" / "derived"
MACRO = GSE_ROOT / "macro"
RESULTS = REPO / "results"
SEED = 20261029

# Release 47 (July 2026) layouts, from Freddie Mac's "SFLLD Disclosure
# Changes, effective July 2026" (v1.2) and the July 2026 General User Guide,
# both saved in docs/. 01_ingest.py asserts these field counts on every file.
ORIG_COLS = [
    "fico", "first_pay", "fthb", "maturity", "msa", "mi_pct", "units", "occ",
    "cltv", "dti", "orig_upb", "ltv", "orig_rate", "channel", "ppm", "amort",
    "state", "prop_type", "zip3", "loan_id", "purpose", "orig_term",
    "n_borrowers", "seller", "super_conf", "pre_harp_id", "program", "harp",
    "valuation", "io", "vantage",
]
PERF_COLS = [
    "loan_id", "period", "upb", "dlq", "age", "rem_months", "defect_date",
    "mod_flag", "zb_code", "zb_date", "cur_rate", "nib_upb", "ddlpi",
    "mi_rec", "net_sale", "non_mi_rec", "expenses", "legal", "maint", "taxes",
    "misc", "actual_loss", "cum_mod_cost", "step_mod", "deferral", "eltv",
    "zb_upb", "dai", "disaster", "assist", "cur_mod_cost", "ib_upb",
    "mi_cancel", "servicer", "cramdown",
]

# Zero-balance codes (User Guide, July 2026). Credit events end the loan
# with a loss calculation (Actual Loss is computed for exactly these four);
# 01 is voluntary payoff; 16 (reperforming securitisation) and 96 (defect)
# remove the loan without either outcome and are treated as censoring.
ZB_PREPAY = "01"
ZB_CREDIT = ("02", "03", "09", "15")
# Delinquency status codes are "00".."99" (months delinquent, capped at 99),
# "RA" (REO acquisition) and "XX" (not available). Six months behind is
# 180+ days, the default threshold recorded in the design.
DEFAULT_MONTHS = 6

FIRST_YEAR, LAST_YEAR = 2005, 2025   # reporting year-ends t
LAST_OUTCOME_YEAR = 2025             # performance ends June 2026

# Judicial-foreclosure states: the 18 listed as "Judicial" in Table 2 of
# Gerardi, Lambie-Hanson and Willen (2013, J. Urban Econ. 73(1); NBER
# w17666, saved in docs/), whose assignment follows Rao and Walsh (2009,
# NCLC). Every other state and DC is power of sale.
JUDICIAL = frozenset({"CT", "DE", "FL", "IA", "IL", "IN", "KS", "KY", "LA", "ME", "ND",
                      "NJ", "NY", "OH", "PA", "SC", "VT", "WI"})

# Model inputs, in the order the design lists them. Sentinel codes for "not
# available" (9999, 999, 99) become missing values in 03_panel.py.
NUMERIC = ["fico", "ltv", "cltv", "dti", "orig_rate", "spread", "units", "n_borrowers",
           "mi_pct", "orig_term", "loan_age", "cur_ltv", "hpi_g", "ur", "ur_chg"]
CATEGORICAL = {
    "purpose": ["P", "C", "N", "R", "9"],
    "occ": ["P", "I", "S", "9"],
    "prop_type": ["SF", "PU", "CO", "MH", "CP", "99"],
    "fthb": ["Y", "N", "9"],
    "channel": ["R", "B", "C", "T", "9"],
}
FEATURES = NUMERIC + list(CATEGORICAL)
# Design: the default hazard must not fall as current LTV or DTI rises, or
# as the credit score falls.
MONOTONE = {"cur_ltv": 1, "dti": 1, "fico": -1}
