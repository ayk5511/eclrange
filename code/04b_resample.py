"""Post hoc (added after the referee review of 2026-10-30): refits of the best
default configuration on independent random samples of the same size.

Usage: python code/04b_resample.py T

The recorded H1 refits use loan-level bootstrap resamples of the fitting
loans, which keep about 63% of distinct loans and so fit with less
information than the best model (their held-out log loss is worse at every
date). Here each refit instead fits on a fresh random sample of the panel of
the same size: the loans whose cap draw falls in the window
[(r + 1) / 21, (r + 1) / 21 + f) (mod 1), where f is the date's sampling
fraction, so each sample is a uniform random draw of the same expected size.
For f below 1/21 (every date from 2018) the 20 samples are disjoint from each
other and from the original; at earlier dates neighbouring samples overlap. Fitting and early
stopping use the same hash split as 04_models.py; accuracy is measured on the
original held-out loans, which no sample fits on.

Outputs: DERIVED/models/T/resample_rNN.{txt,pkl};
results/dates/T/resample.json (held-out log loss of each refit).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import polars as pl

from ecl import DERIVED, RESULTS
from ecl.data import SplineLogit, fit_lgb, lgb_grid, loan_draws, predict, scan_panel, to_matrix

THREADS = int(os.environ.get("ECL_THREADS", "18"))
N_REFIT = 20
HOLD, ES = 0.2, 0.3


def logloss(p, y):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean())


def main() -> None:
    t = int(sys.argv[1])
    out = RESULTS / "dates" / str(t)
    mdir = DERIVED / "models" / str(t)
    acc = json.loads((out / "accuracy.json").read_text())
    frac = acc["info"]["sample_fraction"]
    best = acc["default"]["best"]
    cfg = (list(lgb_grid()) + [{"family": "logit_spline"}])[best]
    t0 = time.time()
    lf = (scan_panel().filter((pl.col("y") <= t - 1) & (pl.col("outcome") >= 0))
          .join(loan_draws().lazy(), on="loan_id"))
    hold = lf.filter((pl.col("u_cap") < frac) & (pl.col("u_split") < HOLD)).sort(["loan_id", "y"]).collect()
    Xh, yh = to_matrix(hold), (hold["outcome"].to_numpy() == 1).astype(np.float64)
    assert len(hold) == acc["info"]["rows"]["held_out"], (len(hold), acc["info"]["rows"]["held_out"])
    res = {"t": t, "fraction": frac, "config": best, "logloss": [], "rows": [], "overlap_original": []}
    for r in range(N_REFIT):
        lo = (r + 1) / (N_REFIT + 1)
        d = pl.col("u_cap") - lo
        win = pl.when(d < 0).then(d + 1.0).otherwise(d) < frac  # wrap the window around 1
        df = lf.filter(win & (pl.col("u_split") >= HOLD)).sort(["loan_id", "y"]).collect()
        X, y = to_matrix(df), (df["outcome"].to_numpy() == 1).astype(np.float64)
        u = df["u_split"].to_numpy()
        fit, es = u >= ES, u < ES
        if cfg["family"] == "logit_spline":
            m = SplineLogit().fit(X[fit], y[fit])
            with open(mdir / f"resample_r{r:02d}.pkl", "wb") as f:
                pickle.dump(m, f)
        else:
            m = fit_lgb(cfg, X[fit], y[fit], None, X[es], y[es], THREADS)
            m.save_model(str(mdir / f"resample_r{r:02d}.txt"), num_iteration=m.best_iteration)
        res["logloss"].append(logloss(predict(m, Xh), yh))
        res["rows"].append(int(fit.sum() + es.sum()))
        res["overlap_original"].append(float((df["u_cap"].to_numpy() < frac).mean()))
        print(f"{t} resample r{r:02d} rows={res['rows'][-1]} ll={res['logloss'][-1]:.6f} {time.time() - t0:.0f}s",
              flush=True)
    (out / "resample.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
