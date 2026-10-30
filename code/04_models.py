"""Fit the default and prepayment model grids at one reporting date and
record held-out accuracy, the set of equally accurate default models, and
20 bootstrap refits of the best default configuration.

Usage: python code/04_models.py T      (T = reporting year-end, 2005..2025)

Training data: loan-years whose outcome year ends on or before T (snapshot
year y <= T - 1), censored loan-years dropped. Loans are split by a keyed
hash: 20% held out (accuracy only), 10% early stopping, 70% fitting. Loans
are kept with probability f = min(1, 6.25M / loan-years available), so the
fitting and early-stopping rows together are capped near the design's 5M.

Outputs:
  results/dates/T/models.csv    one row per model (default and prepayment):
                                held-out log loss and AUC, rounds, family
  results/dates/T/accuracy.json best models, bootstrap SEs, tolerances, sets
  DERIVED/models/T/             fitted models and held-out predictions
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import polars as pl

from ecl import DERIVED, RESULTS, SEED
from ecl.data import SplineLogit, fit_lgb, lgb_grid, loan_draws, predict, scan_panel, to_matrix

THREADS = int(os.environ.get("ECL_THREADS", "18"))
ROW_TARGET = 6_250_000   # 5M fit + early-stop rows / 0.8
HOLD, ES = 0.2, 0.3      # u_split < 0.2 held out; 0.2..0.3 early stop
N_BOOT = 1_000
N_REFIT = 20
EPS = 1e-15
# Smoke test: a tiny run of the whole script, written to a separate tree.
SMOKE = bool(os.environ.get("ECL_SMOKE"))
if SMOKE:
    ROW_TARGET, N_BOOT, N_REFIT = 300_000, 50, 2


def logloss_terms(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def weighted_auc(score, y, w):
    """AUC with case weights; ties count one half (Paper 9's estimator)."""
    order = np.argsort(score, kind="stable")
    s, yy, ww = score[order], y[order], w[order]
    _, start = np.unique(s, return_index=True)
    grp = np.repeat(np.arange(len(start)), np.diff(np.append(start, len(s))))
    pos = np.bincount(grp, weights=ww * yy)
    neg = np.bincount(grp, weights=ww * (1 - yy))
    neg_below = np.cumsum(neg) - neg
    return float((pos * (neg_below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def cluster_boot(loan_idx: np.ndarray, n_loans: int, seed: int) -> np.ndarray:
    """N_BOOT x n_loans multinomial resampling counts over held-out loans
    (each replicate draws n_loans loans with replacement)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_loans, np.full(n_loans, 1.0 / n_loans), size=N_BOOT).astype(np.float32)


def boot_logloss_se(terms, loan_idx, counts):
    per_loan = np.bincount(loan_idx, weights=terms)
    n_rows = np.bincount(loan_idx).astype(np.float64)
    reps = (counts @ per_loan) / (counts @ n_rows)
    return float(np.std(reps, ddof=1))


def boot_auc_se(score, y, loan_idx, counts):
    """Sort once, then each replicate is two weighted bincounts."""
    order = np.argsort(score, kind="stable")
    s, yy, li = score[order], y[order], loan_idx[order]
    _, start = np.unique(s, return_index=True)
    grp = np.repeat(np.arange(len(start)), np.diff(np.append(start, len(s))))
    reps = []
    for c in counts:
        ww = c[li].astype(np.float64)
        pos = np.bincount(grp, weights=ww * yy)
        neg = np.bincount(grp, weights=ww * (1 - yy))
        neg_below = np.cumsum(neg) - neg
        reps.append((pos * (neg_below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))
    return float(np.std(reps, ddof=1))


def load(t: int):
    draws = loan_draws()
    lf = scan_panel().filter((pl.col("y") <= t - 1) & (pl.col("outcome") >= 0))
    n_avail = lf.select(pl.len()).collect().item()
    frac = min(1.0, ROW_TARGET / n_avail)
    df = (lf.join(draws.lazy(), on="loan_id").filter(pl.col("u_cap") < frac)
          .sort(["loan_id", "y"]).collect())
    return df, n_avail, frac


def main() -> None:
    t = int(sys.argv[1])
    out = RESULTS / ("smoke" if SMOKE else "dates") / str(t)
    mdir = DERIVED / ("models_smoke" if SMOKE else "models") / str(t)
    out.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df, n_avail, frac = load(t)
    X = to_matrix(df)
    oc = df["outcome"].to_numpy()
    u = df["u_split"].to_numpy()
    hold, es = u < HOLD, (u >= HOLD) & (u < ES)
    fit = ~hold & ~es
    yd = (oc == 1).astype(np.float64)
    yp = (oc == 2).astype(np.float64)
    nd = oc != 1   # prepayment model rows: no default in the year
    _, hold_loan = np.unique(df["loan_id"].to_numpy()[hold], return_inverse=True)
    n_hold_loans = int(hold_loan.max()) + 1
    info = {"t": t, "loan_years_available": n_avail, "sample_fraction": frac,
            "rows": {"fit": int(fit.sum()), "early_stop": int(es.sum()), "held_out": int(hold.sum())},
            "held_out_loans": n_hold_loans,
            "defaults": {"fit": int(yd[fit].sum()), "early_stop": int(yd[es].sum()), "held_out": int(yd[hold].sum())},
            "prepays": {"fit": int(yp[fit].sum()), "held_out": int(yp[hold].sum())}}
    print(t, json.dumps(info), flush=True)

    counts = cluster_boot(hold_loan, n_hold_loans, SEED + t)
    # Prepayment rows are the held-out rows without a default; their loan
    # index is the same held-out loan index restricted to those rows.
    hold_nd = nd[hold]

    rows, d_pred, p_pred = [], [], []
    grid = list(lgb_grid())
    full = list(enumerate(grid + [{"family": "logit_spline"}]))
    # Deviation 1 (README): the prepayment model is chosen from the eight
    # unconstrained LightGBM settings at seed 1 plus the spline logit. The
    # monotone constraints are stated for default risk, and repeated seeds
    # only pick the luckiest draw of a nuisance model; a prepayment fit
    # takes ~100 s (about 2,700 rounds), so the full grid cost ~2 h a date.
    prepay_grid = [(i, c) for i, c in full if c["family"] == "logit_spline"
                   or (c["family"] == "lgbm" and c["seed"] == 1)]
    for target in ("default", "prepay"):
        if target == "default":
            m_fit, m_es, m_hold, y = fit, es, hold, yd
        else:
            m_fit, m_es, m_hold, y = fit & nd, es & nd, hold & nd, yp
        for i, cfg in (full if target == "default" else prepay_grid):
            if cfg["family"] == "logit_spline":
                m = SplineLogit().fit(X[m_fit], y[m_fit])
                rounds = None
                with open(mdir / f"{target}_m{i:03d}.pkl", "wb") as f:
                    pickle.dump(m, f)
            else:
                m = fit_lgb(cfg, X[m_fit], y[m_fit], None, X[m_es], y[m_es], THREADS)
                rounds = m.best_iteration
                m.save_model(str(mdir / f"{target}_m{i:03d}.txt"), num_iteration=m.best_iteration)
            ph = predict(m, X[m_hold])
            terms = logloss_terms(ph, y[m_hold])
            rows.append({"target": target, "model": i, **{k: cfg.get(k) for k in
                         ("family", "num_leaves", "feature_fraction", "bagging_fraction", "seed")},
                         "rounds": rounds, "logloss": float(terms.mean()),
                         "auc": weighted_auc(ph, y[m_hold], np.ones(len(ph)))})
            (d_pred if target == "default" else p_pred).append(ph.astype(np.float32))
            print(f"{t} {target} m{i:03d} {cfg['family']:12s} ll={rows[-1]['logloss']:.6f} "
                  f"auc={rows[-1]['auc']:.4f} rounds={rounds} {time.time() - t0:.0f}s", flush=True)
    np.save(mdir / "hold_pred_default.npy", np.stack(d_pred))
    np.save(mdir / "hold_pred_prepay.npy", np.stack(p_pred))
    np.save(mdir / "hold_y.npy", np.stack([yd[hold], yp[hold], hold_nd.astype(np.float64)]))
    mt = pl.DataFrame(rows)

    # Bootstrap standard errors of the best default model's held-out log loss
    # and AUC, and of each family's own best (for H4's family sets).
    dm = mt.filter(pl.col("target") == "default")
    acc = {"info": info}
    yh = yd[hold]

    def se_of(i, with_auc=True):
        ph = d_pred[i].astype(np.float64)
        return (boot_logloss_se(logloss_terms(ph, yh), hold_loan, counts),
                boot_auc_se(ph, yh, hold_loan, counts) if with_auc else None)

    best = int(dm.sort("logloss")["model"][0])
    se_ll, se_auc = se_of(best)
    best_auc_i = int(dm.sort("auc", descending=True)["model"][0])
    se_auc_b = se_of(best_auc_i)[1]
    ll = dm["logloss"].to_numpy()
    auc = dm["auc"].to_numpy()
    acc["default"] = {
        "best": best, "best_logloss": float(ll[best]), "se_logloss": se_ll, "se_auc_of_best_ll": se_auc,
        "best_auc_model": best_auc_i, "best_auc": float(auc[best_auc_i]), "se_auc": se_auc_b,
        "sets": {f"ll_{k}se": [int(i) for i in np.flatnonzero(ll <= ll[best] + k * se_ll)] for k in (0.5, 1, 2)}
        | {"auc_1se": [int(i) for i in np.flatnonzero(auc >= auc[best_auc_i] - se_auc_b)]},
        "family": {}}
    for fam in ("lgbm", "lgbm_mono"):
        idx = dm.filter(pl.col("family") == fam)["model"].to_numpy()
        fb = int(idx[np.argmin(ll[idx])])
        fse = se_of(fb, with_auc=False)[0]
        acc["default"]["family"][fam] = {"best": fb, "se_logloss": fse,
                                         "set": [int(i) for i in idx if ll[i] <= ll[fb] + fse]}

    pm = mt.filter(pl.col("target") == "prepay")
    p_ids = pm["model"].to_numpy()
    pll = pm["logloss"].to_numpy()
    pos = int(np.argmin(pll))
    pbest = int(p_ids[pos])
    _, hold_loan_nd = np.unique(df["loan_id"].to_numpy()[hold & nd], return_inverse=True)
    p_counts = cluster_boot(hold_loan_nd, int(hold_loan_nd.max()) + 1, SEED + 1000 + t)
    pph = p_pred[pos].astype(np.float64)
    p_se = boot_logloss_se(logloss_terms(pph, yp[hold & nd]), hold_loan_nd, p_counts)
    acc["prepay"] = {"best": pbest, "best_logloss": float(pll[pos]), "se_logloss": p_se,
                     "grid": [int(i) for i in p_ids],
                     "set_1se": [int(i) for i in p_ids[pll <= pll[pos] + p_se]]}

    # H1: 20 refits of the best default configuration, each on a loan-level
    # bootstrap resample of its fitting loans (resampling counts as case
    # weights); the early-stopping loans are unchanged.
    cfg = (grid + [{"family": "logit_spline"}])[best]
    _, fit_loan = np.unique(df["loan_id"].to_numpy()[fit], return_inverse=True)
    n_fit_loans = int(fit_loan.max()) + 1
    rng = np.random.default_rng(SEED + 2000 + t)
    refit_ll = []
    for r in range(N_REFIT):
        w = rng.multinomial(n_fit_loans, np.full(n_fit_loans, 1.0 / n_fit_loans))[fit_loan].astype(np.float64)
        keep = w > 0
        if cfg["family"] == "logit_spline":
            m = SplineLogit().fit(X[fit][keep], yd[fit][keep], w[keep])
            with open(mdir / f"refit_r{r:02d}.pkl", "wb") as f:
                pickle.dump(m, f)
        else:
            m = fit_lgb(cfg, X[fit][keep], yd[fit][keep], w[keep], X[es], yd[es], THREADS)
            m.save_model(str(mdir / f"refit_r{r:02d}.txt"), num_iteration=m.best_iteration)
        refit_ll.append(float(logloss_terms(predict(m, X[hold]), yh).mean()))
        print(f"{t} refit r{r:02d} ll={refit_ll[-1]:.6f} {time.time() - t0:.0f}s", flush=True)
    acc["refit"] = {"config": best, "logloss": refit_ll}
    acc["seconds"] = round(time.time() - t0)

    mt.write_csv(out / "models.csv")
    (out / "accuracy.json").write_text(json.dumps(acc, indent=1))
    print(t, "done", acc["seconds"], "s; set size", len(acc["default"]["sets"]["ll_1se"]), flush=True)


if __name__ == "__main__":
    main()
