"""Lifetime expected credit loss of the reporting-date portfolio under every
fitted default model, the bootstrap refits and the reversion policies.

Usage: python code/05_allowance.py T

Portfolio: loans active at December T, scored on the 200,000 with the
smallest u_port draw (all of them if fewer). The allowance rate is the
ratio estimator sum(EL) / sum(UPB) on the sample; the dollar allowance is
that rate times the full portfolio's balance.

Projection, per loan, one row per remaining contractual year k = 1..K:
  balance at the start of year k: the interest-bearing balance amortised
    at the current rate over the remaining term, plus any non-interest-
    bearing (deferred) balance, which is due at maturity;
  loan age + 12(k - 1); current LTV = balance / (origination value x HPI
    path / HPI at origination); state HPI growth, unemployment rate and its
    change from the macro path; origination inputs unchanged.
Hazards: h_d from the default model, h_p from the date's best prepayment
model; a final partial year of m months uses 1 - (1 - h)^(m/12).
  survival S_1 = 1, marginal default m_k = S_k h_d, S_{k+1} = S_k (1-h_d)(1-h_p)
  EL = sum_k balance_k x m_k x LGD(current-LTV band at k, judicial state)

Macro paths (point values at year-ends T, T+1, ...; a loan-year starting at
year-end p uses the values at p, as in training):
  forecast (primary): T observed; T+1 and T+2 hold T's HPI growth and
    unemployment rate (the two-year reasonable and supportable period);
    reversion to the long-run mean of each series over all years <= T
    starts at T+2: straight-line over one year (primary), immediately, or
    straight-line over three years (H2);
  realized (secondary): observed values through 2025, then the means.
LGD: balance-weighted severity (total actual loss / total balance at
default) of defaults resolved by December T, by current-LTV band at the start of the default year x
judicial state; a cell with fewer than 30 resolutions uses its band, a band
with fewer than 30 uses all resolutions.

Outputs: results/dates/T/allowance.csv, lgd.csv, portfolio.json;
DERIVED/alloc/T/ loan-level expected losses, lifetime PDs and stages.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from ecl import DERIVED, FEATURES, MACRO, RESULTS
from ecl.data import loan_draws, predict, scan_panel, to_matrix

THREADS = int(os.environ.get("ECL_THREADS", "18"))
N_PORT = 200_000
K_MAX = 40
LTV_BANDS = [0, 60, 80, 90, 100, 120, np.inf]
MIN_CELL = 30
STAGE_RATIO = 2.0
LAST_MACRO = 2025
SMOKE = bool(os.environ.get("ECL_SMOKE"))
if SMOKE:
    N_PORT = 5_000


def lgd_table(t: int) -> tuple[pl.DataFrame, dict]:
    loans = pl.scan_parquet(DERIVED / "loans.parquet")
    dflt = (scan_panel().filter(pl.col("outcome") == 1).select("loan_id", "cur_ltv")
            .join(loans.select("loan_id", "judicial", "zb_code", "zb_date", "actual_loss",
                               "defect_date", "default_upb", "default_period"), on="loan_id")
            .filter(pl.col("zb_date") <= t * 100 + 12)
            .with_columns(loss=pl.when(pl.col("zb_code") == "01").then(0.0)
                          .when(pl.col("zb_code").is_in(["02", "03", "09", "15"]) & pl.col("defect_date").is_null())
                          .then(pl.col("actual_loss")).otherwise(None))
            .filter(pl.col("loss").is_not_null() & (pl.col("default_upb") > 0))
            .with_columns(sev=pl.col("loss") / pl.col("default_upb"),
                          band=pl.col("cur_ltv").cut(LTV_BANDS[1:-1], left_closed=True, labels=[str(i) for i in range(6)])
                          .cast(pl.Utf8).cast(pl.Int8))
            .collect())
    # Severity is balance-weighted (total loss / total balance at default),
    # the rate a dollar allowance needs. An unweighted mean of per-loan
    # ratios is dominated by loans with a near-zero balance at default (one
    # loan with $0.19 at default and a $27,168 loss has a ratio of 143,000).
    overall = float(dflt["loss"].sum() / dflt["default_upb"].sum()) if len(dflt) else float("nan")
    band = dflt.group_by("band").agg(n_b=pl.len(), sev_b=pl.col("loss").sum() / pl.col("default_upb").sum())
    cell = dflt.group_by("band", "judicial").agg(n=pl.len(), sev_c=pl.col("loss").sum() / pl.col("default_upb").sum())
    grid = pl.DataFrame({"band": np.repeat(np.arange(6), 2).astype(np.int8), "judicial": [False, True] * 6})
    tab = (grid.join(cell, on=["band", "judicial"], how="left").join(band, on="band", how="left")
           .with_columns(pl.col("n").fill_null(0), pl.col("n_b").fill_null(0))
           .with_columns(lgd=pl.when(pl.col("n") >= MIN_CELL).then(pl.col("sev_c"))
                         .when(pl.col("n_b") >= MIN_CELL).then(pl.col("sev_b")).otherwise(overall),
                         source=pl.when(pl.col("n") >= MIN_CELL).then(pl.lit("cell"))
                         .when(pl.col("n_b") >= MIN_CELL).then(pl.lit("band")).otherwise(pl.lit("all")))
           .sort("band", "judicial"))
    meta = {"resolved_defaults": len(dflt), "overall_severity": overall,
            "cured_share": float((dflt["zb_code"] == "01").mean()) if len(dflt) else None}
    return tab, meta


def macro_paths(t: int):
    sy = pl.read_parquet(MACRO / "state_year.parquet")
    hist = sy.filter(pl.col("year") <= t)
    means = hist.group_by("state").agg(mg=pl.col("hpi_g").mean(), mu=pl.col("ur").mean())
    now = sy.filter(pl.col("year") == t).select("state", g0="hpi_g", u0="ur", uc0="ur_chg", i0="hpi_q4")
    states = sorted(now["state"].to_list())
    base = now.join(means, on="state").sort("state")
    # group_by yields tuple keys; unpack the state code.
    real = {s: dict(zip(d["year"].to_list(), zip(d["hpi_g"].to_list(), d["ur"].to_list())))
            for (s,), d in sy.filter(pl.col("year") > t).group_by("state")}
    return states, base, real


def path_values(base: pl.DataFrame, real: dict, policy: str, k_max: int, t: int):
    """Arrays [state, k] of (hpi_g, ur, ur_chg, hpi index) at point t + k - 1."""
    n = len(base)
    g = np.zeros((n, k_max))
    u = np.zeros((n, k_max))
    g0, u0, mg, mu = (base[c].to_numpy() for c in ("g0", "u0", "mg", "mu"))
    for k in range(k_max):
        p = k  # years after T
        if policy == "realized":
            if p == 0:
                g[:, k], u[:, k] = g0, u0
            else:
                for i, s in enumerate(base["state"].to_list()):
                    v = real.get(s, {}).get(t + p)
                    g[i, k], u[i, k] = (v if (v is not None and t + p <= LAST_MACRO) else (mg[i], mu[i]))
            continue
        # The value at year-end p is the trailing year's HPI growth and the
        # December unemployment rate, so points T+1 and T+2 describe the two
        # forecast years and must hold the forecast. Reversion starts after
        # T+2: immediate reversion reaches the mean at T+3, which at annual
        # resolution is the same path as one-year straight-line reversion.
        lag = {"sl1": 1, "immediate": 0, "sl3": 3}[policy]
        if p <= 2:
            w = 0.0
        elif lag == 0:
            w = 1.0
        else:
            w = min((p - 2) / lag, 1.0)
        g[:, k] = g0 + (mg - g0) * w
        u[:, k] = u0 + (mu - u0) * w
    uc = np.empty_like(u)
    uc[:, 0] = base["uc0"].to_numpy()
    uc[:, 1:] = u[:, 1:] - u[:, :-1]
    idx = np.empty_like(g)
    idx[:, 0] = base["i0"].to_numpy()
    for k in range(1, k_max):
        idx[:, k] = idx[:, k - 1] * (1 + g[:, k])
    return g, u, uc, idx


def build_rows(port: pl.DataFrame, path_row: np.ndarray, paths, origination=False):
    """Long projection matrix: one row per loan and remaining year. The macro
    arrays in `paths` are [row, k]; loan i reads row path_row[i] (its state
    for the reporting-date projection, itself for the origination one)."""
    g, u, uc, idx = paths
    if origination:
        R = port["orig_term"].to_numpy().astype(np.int64)
        ib0 = port["orig_upb"].to_numpy()
        nib = np.zeros(len(port))
        rate = port["orig_rate"].to_numpy() / 1200
        age0 = np.zeros(len(port))
    else:
        # a few dozen records a year have no remaining term: use the
        # original term less the months already paid
        R = (port["rem_months"].fill_null(port["orig_term"] - port["loan_age"] + 1)
             .to_numpy().astype(np.int64))
        nib = np.nan_to_num(port["nib_upb"].to_numpy())
        ib0 = np.maximum(port["upb"].to_numpy() - nib, 0)
        rate = port["cur_rate"].to_numpy() / 1200
        age0 = port["loan_age"].to_numpy().astype(np.float64)
    R = np.clip(R, 1, None)
    K = np.minimum(np.ceil(R / 12).astype(np.int64), K_MAX)
    loan = np.repeat(np.arange(len(port)), K)
    k = np.concatenate([np.arange(n) for n in K])
    m = 12 * k
    Rl, rl = R[loan], rate[loan]
    with np.errstate(over="ignore", invalid="ignore"):
        gR, gm = (1 + rl) ** Rl, (1 + rl) ** m
        frac = np.where(rl > 0, (gR - gm) / (gR - 1), 1 - m / Rl)
    bal = ib0[loan] * np.clip(frac, 0, 1) + nib[loan]
    months = np.minimum(12, Rl - m).astype(np.float64)
    X = to_matrix(port)[loan]
    col = {f: i for i, f in enumerate(FEATURES)}
    sp = path_row[loan]
    X[:, col["loan_age"]] = age0[loan] + m
    X[:, col["hpi_g"]] = g[sp, k]
    X[:, col["ur"]] = u[sp, k]
    X[:, col["ur_chg"]] = uc[sp, k]
    hpi = idx[sp, k]
    value = port["value"].to_numpy()[loan] * hpi / port["hpi_orig"].to_numpy()[loan]
    X[:, col["cur_ltv"]] = 100 * bal / value
    return X, loan, k, bal, months


def survival_el(hd, hp, loan, k, bal, months, lgd, n_loans):
    hd = 1 - (1 - hd) ** (months / 12)
    hp = 1 - (1 - hp) ** (months / 12)
    stay = (1 - hd) * (1 - hp)
    # cumulative product within loan of the survival factors before year k
    logs = np.log(np.clip(stay, 1e-300, None))
    csum = np.cumsum(logs)
    start = np.r_[0, np.flatnonzero(np.diff(loan)) + 1]
    first = np.repeat(csum[start] - logs[start], np.diff(np.r_[start, len(loan)]))
    S = np.exp(csum - logs - first)
    mk = S * hd
    el = np.bincount(loan, weights=bal * mk * lgd, minlength=n_loans)
    pd_life = np.bincount(loan, weights=mk, minlength=n_loans)
    return el, pd_life


def load_model(path):
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    return lgb.Booster(model_file=str(path))


def main() -> None:
    t = int(sys.argv[1])
    out = RESULTS / ("smoke" if SMOKE else "dates") / str(t)
    mdir = DERIVED / ("models_smoke" if SMOKE else "models") / str(t)
    adir = DERIVED / ("alloc_smoke" if SMOKE else "alloc") / str(t)
    adir.mkdir(parents=True, exist_ok=True)
    acc = json.loads((out / "accuracy.json").read_text())
    t0 = time.time()

    full = scan_panel().filter(pl.col("y") == t)
    n_active, upb_active = full.select(pl.len(), pl.col("upb").sum()).collect().row(0)
    port = (full.join(loan_draws().lazy(), on="loan_id").sort("u_port").head(N_PORT)
            .join(pl.scan_parquet(DERIVED / "loans.parquet").select("loan_id", "orig_upb", "first_pay"), on="loan_id")
            .collect())
    states, base, real = macro_paths(t)
    spos = {s: i for i, s in enumerate(states)}
    missing = set(port["state"].unique().to_list()) - set(spos)
    assert not missing, missing
    state_pos = np.array([spos[s] for s in port["state"].to_list()])
    tab, lgd_meta = lgd_table(t)
    tab.write_csv(out / "lgd.csv")
    lgd_lookup = {(int(b), bool(j)): float(v) for b, j, v in tab.select("band", "judicial", "lgd").iter_rows()}
    jud = port["judicial"].to_numpy()

    def lgd_of(X, loan):
        ltv = X[:, FEATURES.index("cur_ltv")]
        band = np.searchsorted(LTV_BANDS[1:-1], ltv, side="right")
        arr = np.array([[lgd_lookup[(b, False)], lgd_lookup[(b, True)]] for b in range(6)])
        return arr[band, jud[loan].astype(int)]

    # Origination-time projection for IFRS 9 staging: inputs as at
    # origination (age 0, original balance, rate and term, current LTV =
    # original LTV) and the macro values at the year-end before the first
    # payment year, held two years then reverted (one year, straight line)
    # to the same long-run means as at T.
    sy = pl.read_parquet(MACRO / "state_year.parquet")
    y0 = (port["first_pay"].to_numpy() // 100) - 1
    o_key = pl.DataFrame({"state": port["state"], "year": y0.astype(np.int32)})
    o_vals = o_key.join(sy, on=["state", "year"], how="left")
    mg = base.select("state", "mg", "mu")
    o_vals = o_vals.join(mg, on="state", how="left")
    og0, ou0, ouc0 = (o_vals[c].to_numpy() for c in ("hpi_g", "ur", "ur_chg"))
    omg, omu = o_vals["mg"].to_numpy(), o_vals["mu"].to_numpy()
    w = np.array([0.0, 0.0, 0.0] + [1.0] * (K_MAX - 3))  # same timing as the primary path
    oG = og0[:, None] + (omg - og0)[:, None] * w[None, :]
    oU = ou0[:, None] + (omu - ou0)[:, None] * w[None, :]
    oUC = np.empty_like(oU)
    oUC[:, 0] = ouc0
    oUC[:, 1:] = oU[:, 1:] - oU[:, :-1]
    oI = np.cumprod(np.concatenate([np.ones((len(port), 1)), 1 + oG[:, 1:]], axis=1), axis=1) * port["hpi_orig"].to_numpy()[:, None]
    port_o = port.with_columns(pl.col("ltv").cast(pl.Float64).alias("cur_ltv"))

    prepay_best = acc["prepay"]["best"]
    pmodel = load_model(next(mdir.glob(f"prepay_m{prepay_best:03d}.*")))
    d_paths = sorted(mdir.glob("default_m*.*"))
    r_paths = sorted(mdir.glob("refit_r*.*"))
    assert len(d_paths) == 65 and len(r_paths) == len(acc["refit"]["logloss"]), (len(d_paths), len(r_paths))
    prepay_set = acc["prepay"]["set_1se"]
    best = acc["default"]["best"]

    rows = []
    store = {}
    cells = []

    def run(policy: str, models: list, tag: str, origination=False, keep_pd=False, prepay_alt=None):
        if origination:
            X, loan, k, bal, months = build_rows(port_o, np.arange(len(port)), (oG, oU, oUC, oI),
                                                 origination=True)
        else:
            X, loan, k, bal, months = build_rows(port, state_pos, macro_paths_cache[policy])
        lg = lgd_of(X, loan)
        hp = predict(pmodel, X)
        hps = {pi: predict(load_model(next(mdir.glob(f"prepay_m{pi:03d}.*"))), X) for pi in (prepay_alt or [])}
        for name, path in models:
            m = load_model(path)
            hd = predict(m, X)
            el, pdl = survival_el(hd, hp, loan, k, bal, months, lg, len(port))
            if tag == "default" and not origination and name == f"m{best:03d}" and policy in ("sl1", "realized"):
                # exposure-weighted default mass by severity cell, so the
                # allowance can be recomputed offline with another LGD table
                ltv = X[:, FEATURES.index("cur_ltv")]
                cell = np.searchsorted(LTV_BANDS[1:-1], ltv, side="right") * 2 + jud[loan].astype(int)
                for c in range(12):
                    mass, _ = survival_el(hd, hp, loan, k, bal, months, (cell == c).astype(np.float64), len(port))
                    cells.append({"policy": policy, "band": c // 2, "judicial": bool(c % 2),
                                  "mass": float(mass.sum()), "upb_scored": float(port["upb"].sum())})
            rate = el.sum() / port["upb"].sum()
            rows.append({"policy": policy, "set": tag, "model": name, "prepay": prepay_best,
                         "origination": origination, "allowance_rate": rate,
                         "allowance_usd": rate * upb_active, "mean_pd_life": float(pdl.mean())})
            if keep_pd:
                store[(tag, name, origination)] = (el.astype(np.float32), pdl.astype(np.float32))
            for pi, hpa in hps.items():
                el2, _ = survival_el(hd, hpa, loan, k, bal, months, lg, len(port))
                rows.append({"policy": policy, "set": tag + "_prepay", "model": name, "prepay": pi,
                             "origination": origination, "allowance_rate": el2.sum() / port["upb"].sum(),
                             "allowance_usd": el2.sum() / port["upb"].sum() * upb_active, "mean_pd_life": None})

    macro_paths_cache = {p: path_values(base, real, p, K_MAX, t) for p in ("sl1", "immediate", "sl3", "realized")}
    d_models = [(f"m{int(p.stem[-3:]):03d}", p) for p in d_paths]
    r_models = [(f"r{p.stem[-2:]}", p) for p in r_paths]
    # post hoc: refits on independent same-size samples (04b_resample.py)
    s_models = [(f"s{p.stem[-2:]}", p) for p in sorted(mdir.glob("resample_r*.*"))]
    in_set = set(acc["default"]["sets"]["ll_1se"]) | {best}
    set_models = [(n, p) for n, p in d_models if int(n[1:]) in in_set]
    # Primary: every default model on the forecast path with one-year
    # reversion, and every prepayment model in the prepayment set as a
    # secondary; the refits; origination-time projections for staging (set
    # and refits); the other reversion policies and the realized path for
    # the set.
    run("sl1", d_models, "default", keep_pd=True, prepay_alt=[p for p in prepay_set if p != prepay_best])
    run("sl1", r_models, "refit", keep_pd=True)
    run("sl1", set_models, "default", origination=True, keep_pd=True)
    run("sl1", r_models, "refit", origination=True, keep_pd=True)
    if s_models:
        run("sl1", s_models, "resample", keep_pd=True)
        run("sl1", s_models, "resample", origination=True, keep_pd=True)
    for pol in ("immediate", "sl3", "realized"):
        run(pol, set_models, "default")
    print(t, "projections done", round(time.time() - t0), "s", flush=True)

    # Horizon-matched expected loss (defaults through December 2025) for
    # every default model, for H5's secondary comparison.
    X, loan, k, bal, months = build_rows(port, state_pos, macro_paths_cache["sl1"])
    lg = lgd_of(X, loan)
    hp = predict(pmodel, X)
    horizon = (t + k + 1) <= 2025  # row k covers year-end t+k to t+k+1
    trunc = []
    for name, path in d_models:
        hd = predict(load_model(path), X)
        hdz = np.where(horizon, hd, 0.0)
        el, _ = survival_el(hdz, hp, loan, k, bal, months, lg, len(port))
        trunc.append({"model": name, "allowance_rate_to2025": el.sum() / port["upb"].sum()})
    alw = pl.DataFrame(rows)
    pl.DataFrame(trunc).write_csv(out / "allowance_to2025.csv")
    alw.write_csv(out / "allowance.csv")
    pl.DataFrame(cells).write_csv(out / "el_mass.csv")

    # Loan-level arrays (outside the repository).
    for (tag, name, orig), (el, pdl) in store.items():
        np.save(adir / f"{tag}_{name}_{'orig' if orig else 't'}_pd.npy", pdl)
        if not orig:
            np.save(adir / f"{tag}_{name}_el.npy", el)
    port.select("loan_id", "upb", "state", "judicial", "cur_ltv", "fico", "loan_age", "vintage_q").write_parquet(adir / "port.parquet")
    meta = {"t": t, "n_active": n_active, "upb_active": upb_active, "n_scored": len(port),
            "upb_scored": float(port["upb"].sum()), "lgd": lgd_meta,
            "macro_now": base.select(pl.col("g0").mean(), pl.col("u0").mean(), pl.col("mg").mean(),
                                     pl.col("mu").mean()).to_dicts()[0],
            "seconds": round(time.time() - t0)}
    (out / "portfolio.json").write_text(json.dumps(meta, indent=1))
    print(t, "allowance done", meta["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
