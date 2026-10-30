"""Unit tests for the arithmetic the allowance rests on."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").lstrip("0123456789_"), CODE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = load("05_allowance.py")
M = load("04_models.py")


def test_survival_el_matches_closed_form():
    # one loan, three full years, constant hazards
    hd, hp = 0.02, 0.10
    loan = np.array([0, 0, 0])
    k = np.array([0, 1, 2])
    bal = np.array([100.0, 90.0, 80.0])
    months = np.full(3, 12.0)
    lgd = np.full(3, 0.4)
    el, pd = A.survival_el(np.full(3, hd), np.full(3, hp), loan, k, bal, months, lgd, 1)
    s = 1.0
    exp_el = exp_pd = 0.0
    for b in bal:
        exp_el += b * s * hd * 0.4
        exp_pd += s * hd
        s *= (1 - hd) * (1 - hp)
    assert el[0] == pytest.approx(exp_el, rel=1e-12)
    assert pd[0] == pytest.approx(exp_pd, rel=1e-12)


def test_survival_resets_between_loans():
    loan = np.array([0, 0, 1, 1])
    k = np.array([0, 1, 0, 1])
    el, pd = A.survival_el(np.full(4, 0.5), np.zeros(4), loan, k, np.ones(4), np.full(4, 12.0), np.ones(4), 2)
    assert pd == pytest.approx([0.75, 0.75])


def test_partial_year_hazard():
    el, pd = A.survival_el(np.array([0.2]), np.array([0.0]), np.array([0]), np.array([0]), np.ones(1),
                           np.array([6.0]), np.ones(1), 1)
    assert pd[0] == pytest.approx(1 - 0.8 ** 0.5)


def base_frame():
    return pl.DataFrame({"state": ["AA"], "g0": [0.10], "u0": [8.0], "uc0": [2.0], "i0": [200.0],
                         "mg": [0.04], "mu": [5.0]})


@pytest.mark.parametrize("policy,expect_u", [
    ("sl1", [8.0, 8.0, 8.0, 5.0, 5.0]),
    ("immediate", [8.0, 8.0, 8.0, 5.0, 5.0]),   # same as sl1 at annual resolution
    ("sl3", [8.0, 8.0, 8.0, 7.0, 6.0, 5.0]),
])
def test_reversion_timing(policy, expect_u):
    g, u, uc, idx = A.path_values(base_frame(), {}, policy, len(expect_u), 2010)
    assert u[0] == pytest.approx(expect_u)
    assert uc[0, 0] == 2.0 and uc[0, 1:] == pytest.approx(np.diff(expect_u))
    gexp = 0.10 + (0.04 - 0.10) * (8.0 - np.array(expect_u)) / 3.0
    assert g[0] == pytest.approx(gexp)
    assert idx[0, 0] == 200.0
    assert idx[0, 1] == pytest.approx(200.0 * (1 + g[0, 1]))


def test_realized_path_uses_observed_then_mean():
    real = {"AA": {2011: (0.01, 9.0), 2012: (-0.02, 9.5)}}
    g, u, uc, idx = A.path_values(base_frame(), real, "realized", 5, 2010)
    assert u[0] == pytest.approx([8.0, 9.0, 9.5, 5.0, 5.0])
    assert g[0, :3] == pytest.approx([0.10, 0.01, -0.02])


def test_amortisation_schedule():
    from ecl import FEATURES
    row = {f: [0.0] for f in FEATURES}
    row.update({"rem_months": [360], "nib_upb": [0.0], "upb": [100_000.0], "cur_rate": [6.0],
                "loan_age": [0], "value": [125_000.0], "hpi_orig": [100.0], "orig_term": [360],
                "orig_upb": [100_000.0], "orig_rate": [6.0]})
    port = pl.DataFrame(row)
    paths = A.path_values(base_frame().with_columns(g0=pl.lit(0.0), mg=pl.lit(0.0)), {}, "sl1", 40, 2010)
    paths = (paths[0], paths[1], paths[2], np.full_like(paths[3], 100.0))
    X, loan, k, bal, months = A.build_rows(port, np.array([0]), paths)
    assert len(k) == 30 and months.sum() == 360
    r = 0.06 / 12
    pay = 100_000 * r / (1 - (1 + r) ** -360)
    b12 = 100_000 * (1 + r) ** 12 - pay * ((1 + r) ** 12 - 1) / r
    assert bal[1] == pytest.approx(b12, rel=1e-9)
    assert X[0, FEATURES.index("cur_ltv")] == pytest.approx(80.0)
    assert X[1, FEATURES.index("loan_age")] == 12


def test_weighted_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    s = np.round(rng.normal(size=5000), 2)
    y = (rng.random(5000) < 1 / (1 + np.exp(-s))).astype(float)
    assert M.weighted_auc(s, y, np.ones(5000)) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    w = rng.integers(0, 3, 5000).astype(float)
    assert M.weighted_auc(s, y, w) == pytest.approx(roc_auc_score(y, s, sample_weight=w), abs=1e-12)


def test_bootstrap_auc_fast_path_equals_naive():
    rng = np.random.default_rng(1)
    s = rng.normal(size=2000)
    y = (rng.random(2000) < 0.3).astype(float)
    loan = rng.integers(0, 500, 2000)
    counts = rng.multinomial(500, np.full(500, 1 / 500), size=20).astype(np.float32)
    naive = np.std([M.weighted_auc(s, y, c[loan].astype(float)) for c in counts], ddof=1)
    assert M.boot_auc_se(s, y, loan, counts) == pytest.approx(naive, rel=1e-10)


def test_logloss_se_is_clustered_ratio():
    terms = np.array([1.0, 3.0, 5.0])
    loan = np.array([0, 0, 1])
    counts = np.array([[2, 0], [0, 2], [1, 1]], dtype=np.float32)
    se = M.boot_logloss_se(terms, loan, counts)
    reps = [2.0, 5.0, 3.0]   # (2*4)/(2*2), (2*5)/(2*1), (4+5)/(2+1)
    assert se == pytest.approx(np.std(reps, ddof=1))


@pytest.mark.skipif(not (A.MACRO / "state_year.parquet").exists(), reason="macro tables not built")
def test_realized_path_reads_each_state():
    # regression: polars group_by keys are tuples; the realized path must
    # pick up each state's observed values, not fall back to the means
    sy = pl.read_parquet(A.MACRO / "state_year.parquet")
    states, base, real = A.macro_paths(2007)
    g, u, uc, idx = A.path_values(base, real, "realized", 4, 2007)
    i = states.index("CA")
    ca = sy.filter(pl.col("state") == "CA").sort("year")
    for k, yr in enumerate((2007, 2008, 2009, 2010)):
        row = ca.filter(pl.col("year") == yr)
        assert g[i, k] == pytest.approx(row["hpi_g"].item())
        assert u[i, k] == pytest.approx(row["ur"].item())
