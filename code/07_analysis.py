"""Evaluate H1-H6 and the secondary analyses from the per-date outputs.

Usage: python code/07_analysis.py [--smoke]

Reads results/dates/T/{accuracy.json, allowance.csv, allowance_to2025.csv,
portfolio.json}, results/realized.csv and the loan-level arrays under
DERIVED/alloc/T (lifetime PDs and expected losses).

Outputs:
  results/by_date.csv           one row per reporting date, every quantity
                                the hypotheses and figures use
  results/models_long.csv       one row per date x default model: held-out
                                log loss, AUC, allowance rate, set flags
  results/loan_dispersion.csv   loan-level lifetime-PD spread across the set,
                                by current-LTV band and credit-score band
  results/summary.json          the six predictions, each held or failed
"""
from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl

from ecl import DERIVED, FIRST_YEAR, LAST_YEAR, REPO, RESULTS, SEED

REPO_CODE = REPO / "code"

SMOKE = "--smoke" in sys.argv
DATES_DIR = RESULTS / ("smoke" if SMOKE else "dates")
ALLOC = DERIVED / ("alloc_smoke" if SMOKE else "alloc")
STAGE_RATIO = 2.0
N_SAMPLING_BOOT = 200
H5_MIN_YEARS = 8
LTV_BANDS = [60, 80, 90, 100, 120]
FICO_BANDS = [620, 680, 720, 760]


def p95_width(x: np.ndarray) -> float:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(hi - lo)


def stages(tag: str, names: list[str], adir) -> np.ndarray:
    """[model, loan] booleans: lifetime PD at T more than twice the lifetime PD
    at origination (IFRS 9 analogue of a significant increase)."""
    out = []
    for n in names:
        pt = np.load(adir / f"{tag}_{n}_t_pd.npy").astype(np.float64)
        po = np.load(adir / f"{tag}_{n}_orig_pd.npy").astype(np.float64)
        out.append(pt > STAGE_RATIO * po)
    return np.array(out)


def flip_share(st: np.ndarray) -> float:
    return float((st.any(axis=0) & ~st.all(axis=0)).mean())


def paired_sets(t: int, acc: dict) -> dict:
    """Post-hoc (added after the first date's results): a stricter set. A
    model is in it if its held-out log loss exceeds the best's by at most k
    loan-clustered bootstrap SEs of the paired difference (same resamples as
    04_models.py), k = 1 or 2. The recorded set uses the SE of the best
    model's own loss, which is far larger than the SE of a difference."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("m04", REPO_CODE / "04_models.py")
    m04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m04)
    df, _, _ = m04.load(t)
    hold = df["u_split"].to_numpy() < m04.HOLD
    _, hl = np.unique(df["loan_id"].to_numpy()[hold], return_inverse=True)
    mdir = DERIVED / "models" / str(t)
    P = np.load(mdir / "hold_pred_default.npy").astype(np.float64)
    y = np.load(mdir / "hold_y.npy")[0]
    assert len(y) == hold.sum() == P.shape[1]
    b = acc["default"]["best"]
    L = np.array([m04.logloss_terms(p, y) for p in P])
    # saved predictions are float32; the fit-time log loss used float64
    assert abs(L[b].mean() / acc["default"]["best_logloss"] - 1) < 1e-5
    counts = m04.cluster_boot(hl, int(hl.max()) + 1, SEED + t)
    nrow = np.bincount(hl).astype(np.float64)
    den = counts @ nrow
    out = {1: [], 2: []}
    for m in range(len(P)):
        dlt = (L[m] - L[b]).mean()
        se = 0.0 if m == b else float(np.std((counts @ np.bincount(hl, weights=L[m] - L[b])) / den, ddof=1))
        for k in out:
            if dlt <= k * se:
                out[k].append(m)
    return out


def main() -> None:
    realized = pl.read_csv(RESULTS / "realized.csv") if (RESULTS / "realized.csv").exists() and not SMOKE else None
    rows, longs, disp = [], [], []
    for t in range(FIRST_YEAR, LAST_YEAR + 1):
        d = DATES_DIR / str(t)
        if not (d / "allowance.csv").exists():
            continue
        acc = json.loads((d / "accuracy.json").read_text())
        port = json.loads((d / "portfolio.json").read_text())
        alw = pl.read_csv(d / "allowance.csv")
        to25 = pl.read_csv(d / "allowance_to2025.csv")
        mt = pl.read_csv(d / "models.csv").filter(pl.col("target") == "default").sort("model")
        dfl = acc["default"]
        best = dfl["best"]
        bname = f"m{best:03d}"
        prim = alw.filter((pl.col("set") == "default") & (pl.col("policy") == "sl1") & ~pl.col("origination"))
        rate = dict(zip(prim["model"].to_list(), prim["allowance_rate"].to_list()))
        assert len(rate) == 65, (t, len(rate))
        names = lambda ids: [f"m{i:03d}" for i in ids]  # noqa: E731
        S = names(dfl["sets"]["ll_1se"])
        r_set = np.array([rate[n] for n in S])
        refit = alw.filter((pl.col("set") == "refit") & (pl.col("policy") == "sl1") & ~pl.col("origination"))
        r_ref = refit["allowance_rate"].to_numpy()
        pol = {p: alw.filter((pl.col("set") == "default") & (pl.col("policy") == p) & ~pl.col("origination")
                             & (pl.col("model") == bname))["allowance_rate"].item()
               for p in ("sl1", "immediate", "sl3")}
        real_path = alw.filter((pl.col("set") == "default") & (pl.col("policy") == "realized"))
        rp = dict(zip(real_path["model"].to_list(), real_path["allowance_rate"].to_list()))
        r_real = np.array([rp[n] for n in S])
        # prepayment model varied too: every (default in set, prepay in set)
        pp = alw.filter((pl.col("set") == "default_prepay") & pl.col("model").is_in(S))["allowance_rate"].to_numpy()
        r_both = np.concatenate([r_set, pp])

        def rng_of(ids):
            v = np.array([rate[n] for n in names(ids)])
            return float(v.max() - v.min())

        fam = {}
        for f in ("lgbm", "lgbm_mono"):
            fs = dfl["family"][f]
            v = np.array([rate[n] for n in names(fs["set"])])
            fam[f] = {"size": len(v), "rel_range": float((v.max() - v.min()) / rate[f"m{fs['best']:03d}"])}

        # IFRS 9 staging flips
        adir = ALLOC / str(t)
        st_set = stages("default", S, adir)
        st_ref = stages("refit", [f"r{i:02d}" for i in range(len(r_ref))], adir)

        # Post hoc: refits on independent same-size samples (04b_resample.py)
        rsm = alw.filter((pl.col("set") == "resample") & (pl.col("policy") == "sl1") & ~pl.col("origination"))
        r_rsm = rsm["allowance_rate"].to_numpy()
        rj = json.loads((d / "resample.json").read_text()) if (d / "resample.json").exists() else None
        st_rsm = stages("resample", [f"s{i:02d}" for i in range(len(r_rsm))], adir) if len(r_rsm) else None
        ll_best, se_best = acc["default"]["best_logloss"], acc["default"]["se_logloss"]
        refit_gap = (np.array(acc["refit"]["logloss"]) - ll_best) / se_best

        # H5 decomposition for the best model: forecast path vs realized path,
        # each with the date-T severity table, and the realized path with the
        # severity table of all defaults resolved by December 2025.
        mass = pl.read_csv(d / "el_mass.csv")
        final_lgd = pl.read_csv(DATES_DIR / "2025" / "lgd.csv").select("band", "judicial", "lgd")
        tlgd = pl.read_csv(d / "lgd.csv").select("band", "judicial", "lgd")

        def el_with(policy, table):
            mm = mass.filter(pl.col("policy") == policy).join(table, on=["band", "judicial"])
            return float((mm["mass"] * mm["lgd"]).sum() / mm["upb_scored"][0])

        # sampling error of the 200,000-loan sample: bootstrap over loans
        upb = pl.read_parquet(adir / "port.parquet")["upb"].to_numpy()
        el = np.array([np.load(adir / f"default_{n}_el.npy").astype(np.float64) for n in S])
        g = np.random.default_rng(t)
        bs_best, bs_range = [], []
        bi = S.index(bname)
        for _ in range(N_SAMPLING_BOOT):
            ix = g.integers(0, len(upb), len(upb))
            rr = el[:, ix].sum(axis=1) / upb[ix].sum()
            bs_best.append(rr[bi])
            bs_range.append(rr.max() - rr.min())

        # loan-level lifetime-PD spread across the set
        pdt = np.array([np.load(adir / f"default_{n}_t_pd.npy").astype(np.float64) for n in S])
        pf = pl.read_parquet(adir / "port.parquet").with_columns(
            pd_lo=pdt.min(axis=0), pd_hi=pdt.max(axis=0), pd_best=pdt[bi],
            ltv_band=pl.col("cur_ltv").cut(LTV_BANDS, left_closed=True),
            fico_band=pl.col("fico").cut(FICO_BANDS, left_closed=True))
        for key in ("ltv_band", "fico_band"):
            agg = (pf.group_by(key).agg(n=pl.len(), pd_best=pl.col("pd_best").mean(),
                                        spread=(pl.col("pd_hi") - pl.col("pd_lo")).mean(),
                                        ratio_med=(pl.col("pd_hi") / pl.col("pd_lo")).median())
                   .with_columns(t=pl.lit(t), by=pl.lit(key), band=pl.col(key).cast(pl.Utf8)).drop(key))
            disp.append(agg)

        ll = mt["logloss"].to_numpy()
        auc = mt["auc"].to_numpy()
        for i in range(65):
            n = f"m{i:03d}"
            longs.append({"t": t, "model": i, "family": mt["family"][i], "logloss": ll[i], "auc": auc[i],
                          "d_logloss_se": (ll[i] - ll[best]) / dfl["se_logloss"],
                          "allowance_rate": rate[n], "rel_to_best": rate[n] / rate[bname] - 1,
                          "in_set": i in dfl["sets"]["ll_1se"]})
        ps = paired_sets(t, acc) if not SMOKE else {1: [best], 2: [best]}
        # Post hoc, like-for-like versions of H1 and H6: the recorded tests
        # compare the full range of ~50 set models with the central 95%
        # width of 20 refits (H1), and disagreement among ~50 models with
        # disagreement among 20 (H6), both of which favour the set. Here 20
        # set models are drawn at random (2,000 / 200 draws) and compared
        # with the full range / disagreement of the 20 refits.
        g2 = np.random.default_rng(10_000 + t)
        k20 = min(len(r_ref), len(S))
        lfl_range = float(np.mean([np.ptp(g2.choice(r_set, k20, replace=False)) for _ in range(2000)]))
        lfl_flip = float(np.mean([flip_share(st_set[g2.choice(len(S), k20, replace=False)]) for _ in range(200)]))
        to25r = dict(zip(to25["model"].to_list(), to25["allowance_rate_to2025"].to_list()))
        row = {"t": t, "n_active": port["n_active"], "upb_active": port["upb_active"],
               "n_scored": port["n_scored"], "best": best, "best_family": mt["family"][best],
               "best_logloss": ll[best], "se_logloss": dfl["se_logloss"], "best_auc": auc[best],
               "set_size": len(S), "set_lgbm": sum(mt["family"][int(n[1:])] == "lgbm" for n in S),
               "set_mono": sum(mt["family"][int(n[1:])] == "lgbm_mono" for n in S),
               "set_logit": sum(mt["family"][int(n[1:])] == "logit_spline" for n in S),
               "best_rate": rate[bname], "set_min": float(r_set.min()), "set_max": float(r_set.max()),
               "set_range": float(r_set.max() - r_set.min()),
               "rel_range": float((r_set.max() - r_set.min()) / rate[bname]),
               "usd_best": rate[bname] * port["upb_active"],
               "usd_range": float(r_set.max() - r_set.min()) * port["upb_active"],
               "all65_min": float(min(rate.values())), "all65_max": float(max(rate.values())),
               "refit_mean": float(r_ref.mean()), "refit_w95": p95_width(r_ref),
               "refit_min": float(r_ref.min()), "refit_max": float(r_ref.max()),
               "refit_p025": float(np.percentile(r_ref, 2.5)), "refit_p975": float(np.percentile(r_ref, 97.5)),
               "policy_sl1": pol["sl1"], "policy_imm": pol["immediate"], "policy_sl3": pol["sl3"],
               "policy_range": float(max(pol.values()) - min(pol.values())),
               "range_0.5se": rng_of(dfl["sets"]["ll_0.5se"]), "size_0.5se": len(dfl["sets"]["ll_0.5se"]),
               "range_2se": rng_of(dfl["sets"]["ll_2se"]), "size_2se": len(dfl["sets"]["ll_2se"]),
               "range_auc": rng_of(dfl["sets"]["auc_1se"]), "size_auc": len(dfl["sets"]["auc_1se"]),
               "range_paired1": rng_of(ps[1]), "size_paired1": len(ps[1]),
               "range_paired2": rng_of(ps[2]), "size_paired2": len(ps[2]),
               "realpath_best": rp[bname], "realpath_min": float(r_real.min()), "realpath_max": float(r_real.max()),
               "both_min": float(r_both.min()), "both_max": float(r_both.max()),
               "prepay_set_size": len(acc["prepay"]["set_1se"]),
               "fam_lgbm_rel": fam["lgbm"]["rel_range"], "fam_lgbm_size": fam["lgbm"]["size"],
               "fam_mono_rel": fam["lgbm_mono"]["rel_range"], "fam_mono_size": fam["lgbm_mono"]["size"],
               "lfl_range20": lfl_range, "refit_range": float(np.ptp(r_ref)), "lfl_flip20": lfl_flip,
               "refit_gap_se_median": float(np.median(refit_gap)), "refit_gt1se": int((refit_gap > 1).sum()),
               "refit_bias": float(r_ref.mean() / rate[bname] - 1),
               "rsm_range": float(np.ptp(r_rsm)) if len(r_rsm) else None,
               "rsm_w95": p95_width(r_rsm) if len(r_rsm) else None,
               "rsm_bias": float(r_rsm.mean() / rate[bname] - 1) if len(r_rsm) else None,
               "rsm_gap_se_median": float(np.median((np.array(rj["logloss"]) - ll_best) / se_best)) if rj else None,
               "rsm_flip": flip_share(st_rsm) if st_rsm is not None else None,
               "dec_sl1_tlgd": el_with("sl1", tlgd), "dec_real_tlgd": el_with("realized", tlgd),
               "dec_real_flgd": el_with("realized", final_lgd),
               "stage2_best": float(st_set[bi].mean()), "flip_set": flip_share(st_set),
               "flip_refit": flip_share(st_ref),
               "sampling_se_best": float(np.std(bs_best, ddof=1)),
               "sampling_range_lo": float(np.percentile(bs_range, 2.5)),
               "sampling_range_hi": float(np.percentile(bs_range, 97.5)),
               "to25_min": min(to25r[n] for n in S), "to25_max": max(to25r[n] for n in S),
               "lgd_resolved": port["lgd"]["resolved_defaults"], "lgd_overall": port["lgd"]["overall_severity"]}
        if realized is not None:
            rz = realized.filter(pl.col("t") == t)
            row |= {"realized": rz["realized_rate"].item(), "realized_to25": rz["realized_rate_to2025"].item(),
                    "realized_lb": rz["realized_rate_lb"].item(),
                    "years_observed": rz["years_observed"].item(), "imputed_share": rz["imputed_share_of_loss"].item()}
        rows.append(row)
        print(t, f"set={len(S)} best={rate[bname]:.4%} range={row['set_range']:.4%} "
              f"refitW={row['refit_w95']:.4%} policy={row['policy_range']:.4%} flips {row['flip_set']:.3f}/{row['flip_refit']:.3f}",
              flush=True)

    bd = pl.DataFrame(rows)
    out = RESULTS / ("smoke_by_date.csv" if SMOKE else "by_date.csv")
    bd.write_csv(out)
    if SMOKE:
        return
    pl.DataFrame(longs).write_csv(RESULTS / "models_long.csv")
    pl.concat(disp, how="diagonal").write_csv(RESULTS / "loan_dispersion.csv")

    n = len(bd)
    h1 = (bd["set_range"] > bd["refit_w95"]).to_numpy()
    h2 = (bd["set_range"] >= bd["policy_range"]).to_numpy()
    others = bd.filter(~pl.col("t").is_in([2006, 2007]))["rel_range"].median()
    r06 = bd.filter(pl.col("t") == 2006)["rel_range"].item()
    r07 = bd.filter(pl.col("t") == 2007)["rel_range"].item()
    h4 = (bd["fam_mono_rel"] < bd["fam_lgbm_rel"]).to_numpy()
    elig = bd.filter(pl.col("years_observed") >= H5_MIN_YEARS)
    inside = ((elig["realized"] >= elig["set_min"]) & (elig["realized"] <= elig["set_max"])).to_numpy()
    inside_lb = ((elig["realized_lb"] >= elig["set_min"]) & (elig["realized_lb"] <= elig["set_max"])).to_numpy()
    inside25 = ((elig["realized_to25"] >= elig["to25_min"]) & (elig["realized_to25"] <= elig["to25_max"])).to_numpy()
    h6 = (bd["flip_set"] > bd["flip_refit"]).to_numpy()
    summ = {
        "n_dates": n,
        "H1": {"dates_holding": int(h1.sum()), "holds": bool(h1.sum() > n / 2),
               "median_ratio_range_to_refit": float((bd["set_range"] / bd["refit_w95"]).median())},
        "H2": {"dates_holding": int(h2.sum()), "holds": bool(h2.sum() > n / 2),
               "median_ratio_range_to_policy": float((bd["set_range"] / bd["policy_range"]).median())},
        "H3": {"rel_2006": r06, "rel_2007": r07, "median_others": float(others),
               "holds": bool(r06 > others and r07 > others)},
        "H4": {"dates_holding": int(h4.sum()), "holds": bool(h4.sum() > n / 2)},
        "H5": {"eligible_dates": [int(x) for x in elig["t"]], "inside": int(inside.sum()),
               "holds": bool(inside.sum() < len(elig) / 2), "inside_horizon_matched": int(inside25.sum()),
               "inside_lower_bound": int(inside_lb.sum()),
               "above_range": int((elig["realized"] > elig["set_max"]).sum()),
               "below_range": int((elig["realized"] < elig["set_min"]).sum()),
               "inside_dates": [int(x) for x in elig["t"].filter(pl.Series(inside))]},
        "H6": {"dates_holding": int(h6.sum()), "holds": bool(h6.sum() > n / 2)},
    }
    lr = (bd["lfl_range20"] / bd["refit_range"]).to_numpy()
    summ["post_hoc"] = {
        "H1_like_for_like": {"dates_set_larger": int((lr > 1).sum()), "median_ratio": float(np.median(lr))},
        "H6_like_for_like": {"dates_set_larger": int((bd["lfl_flip20"] > bd["flip_refit"]).sum()),
                             "median_flip_set20": float(bd["lfl_flip20"].median()),
                             "median_flip_refit": float(bd["flip_refit"].median())},
        "paired2_rel_range_median": float((bd["range_paired2"] / bd["best_rate"]).median()),
        "paired2_size_median": float(bd["size_paired2"].median()),
        "H5_lower_bound_inside": int(inside_lb.sum()),
        "refit_quality": {"median_gap_se": float(bd["refit_gap_se_median"].median()),
                          "refits_gt1se": int(bd["refit_gt1se"].sum()),
                          "dates_mean_refit_above_best": int((bd["refit_bias"] > 0).sum()),
                          "median_bias": float(bd["refit_bias"].median())},
    }
    if bd["rsm_range"].null_count() == 0:
        rr = (bd["lfl_range20"] / bd["rsm_range"]).to_numpy()
        summ["post_hoc"]["resample"] = {
            "dates_set20_larger_than_resample_range": int((rr > 1).sum()), "median_ratio": float(np.median(rr)),
            "dates_set_range_gt_resample_w95": int((bd["set_range"] > bd["rsm_w95"]).sum()),
            "median_gap_se": float(bd["rsm_gap_se_median"].median()),
            "median_bias": float(bd["rsm_bias"].median()),
            "dates_flip_set20_gt_resample": int((bd["lfl_flip20"] > bd["rsm_flip"]).sum()),
            "median_flip_resample": float(bd["rsm_flip"].median())}
    (RESULTS / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
