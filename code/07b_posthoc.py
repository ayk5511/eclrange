"""Post hoc checks added after the second referee review (2026-10-30).

Usage: python code/07b_posthoc.py   (after 07_analysis.py)

Reads only results/; writes results/posthoc2.csv (one row per date):
- where the most accurate model sits in the set's range, and the set median;
- Spearman correlation of held-out log loss with allowance across the set,
  and across the 20 independent-sample refits of the best configuration
  (does choosing the lowest log loss pick a low allowance?);
- H4 with equal-size sets: the mean range of 2,000 random subsets of each
  family's set, both of the smaller family's size;
- the H5 decomposition with the severity table of December 2019 (defaults
  resolved before pandemic forbearance), and the best model's mean lifetime
  PD on the realized path next to the realized default share;
- whether realized losses fall inside the range of each kind of refit.
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from ecl import RESULTS

PRE_PANDEMIC_LGD = 2019  # last table built only from resolutions before March 2020
N_SUB = 2000


def main() -> None:
    bd = pl.read_csv(RESULTS / "by_date.csv").sort("t")
    rz = {r["t"]: r for r in pl.read_csv(RESULTS / "realized.csv").iter_rows(named=True)}
    lgd19 = pl.read_csv(RESULTS / "dates" / str(PRE_PANDEMIC_LGD) / "lgd.csv").select("band", "judicial", "lgd")
    rows = []
    for t in bd["t"].to_list():
        d = RESULTS / "dates" / str(t)
        acc = json.loads((d / "accuracy.json").read_text())
        best = acc["default"]["best"]
        al = pl.read_csv(d / "allowance.csv").filter(~pl.col("origination"))
        prim = dict(al.filter((pl.col("set") == "default") & (pl.col("policy") == "sl1"))
                    .select("model", "allowance_rate").iter_rows())
        ll = dict(pl.read_csv(d / "models.csv").filter(pl.col("target") == "default")
                  .select("model", "logloss").iter_rows())
        S = sorted(acc["default"]["sets"]["ll_1se"])
        v = np.array([prim[f"m{m:03d}"] for m in S])
        b = prim[f"m{best:03d}"]
        rsm = al.filter((pl.col("set") == "resample") & (pl.col("policy") == "sl1")).sort("model")
        rj = json.loads((d / "resample.json").read_text())
        assert rsm["model"].to_list() == [f"s{r:02d}" for r in range(20)]
        ref = al.filter((pl.col("set") == "refit") & (pl.col("policy") == "sl1"))["allowance_rate"].to_numpy()
        # H4 on equal-size sets
        g = np.random.default_rng(20_000 + t)
        fam = {}
        for f in ("lgbm", "lgbm_mono"):
            fs = acc["default"]["family"][f]
            fam[f] = (np.array([prim[f"m{m:03d}"] for m in fs["set"]]), prim[f"m{fs['best']:03d}"])
        k = min(len(fam["lgbm"][0]), len(fam["lgbm_mono"][0]))
        sub = {f: float(np.mean([np.ptp(g.choice(x, k, replace=False)) for _ in range(N_SUB)])) / fb
               for f, (x, fb) in fam.items()}
        # decomposition with the pre-pandemic severity table
        mass = pl.read_csv(d / "el_mass.csv").filter(pl.col("policy") == "realized").join(lgd19, on=["band", "judicial"])
        assert len(mass) == 12
        pd_real = al.filter((pl.col("set") == "default") & (pl.col("policy") == "realized")
                            & (pl.col("model") == f"m{best:03d}"))["mean_pd_life"]
        R = rz[t]["realized_rate"]
        rows.append({
            "t": t, "best_pos": float((b - v.min()) / (v.max() - v.min())),
            "set_median": float(np.median(v)), "best_rate": b,
            "rho_set": float(spearmanr([ll[m] for m in S], v).statistic),
            "rho_rsm": float(spearmanr(rj["logloss"], rsm["allowance_rate"].to_numpy()).statistic),
            "h4_k": k, "h4_lgbm_sub": sub["lgbm"], "h4_mono_sub": sub["lgbm_mono"],
            "dec_real_lgd19": float((mass["mass"] * mass["lgd"]).sum() / mass["upb_scored"][0]),
            "pd_real_best": float(pd_real.item()) if len(pd_real) == 1 else None,
            "default_share": rz[t]["default_share"],
            "realized_in_refit": bool(ref.min() <= R <= ref.max()),
            "realized_in_rsm": bool(rsm["allowance_rate"].min() <= R <= rsm["allowance_rate"].max()),
        })
    out = pl.DataFrame(rows)
    out.write_csv(RESULTS / "posthoc2.csv")
    print(out)


if __name__ == "__main__":
    main()
