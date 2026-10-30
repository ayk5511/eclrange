"""Write every number and table the paper prints, from results/ only.

Usage: python code/09_paper.py
Writes paper/numbers.tex (\\num{key} macros), paper/tables/*.tex and
paper/submission-ssrn/abstract.txt. The paper never types a result by hand.
"""
from __future__ import annotations

import json
import re

import numpy as np
import polars as pl

from ecl import FEATURES, MACRO, REPO, RESULTS
from ecl.data import lgb_grid

PAPER = REPO / "paper"
TAB = PAPER / "tables"
NUM: dict[str, str] = {}


def put(key: str, value, fmt: str | None = None) -> None:
    if fmt == "int":
        s = f"{int(round(value)):,}".replace(",", "{,}")
    elif fmt:
        s = format(value, fmt)
    else:
        s = str(value)
    NUM[key] = s


def bp(x: float) -> float:
    return x * 1e4


def table(name: str, body: str) -> None:
    (TAB / f"{name}.tex").write_text(body)


def data_numbers(flow, logs_summary, stats):
    put("data.quarters", logs_summary["quarters"])
    put("data.orig_rows", logs_summary["orig_rows"], "int")
    put("data.perf_rows", logs_summary["perf_rows"] / 1e9, ".2f")
    put("data.excl_hpi", flow["excluded_no_state_hpi"], "int")
    put("data.excl_ltv", flow["excluded_no_ltv"], "int")
    ly = flow["loan_years"]
    put("data.active_2025", sum(ly["2025"].values()) / 1e6, ".1f")
    rates = {int(y) + 1: v.get("1", 0) / sum(v[k] for k in ("0", "1", "2") if k in v)
             for y, v in ly.items() if "None" not in v}
    gfc = {y: r for y, r in rates.items() if y <= 2015}
    yg = max(gfc, key=gfc.get)
    put("data.dr_peak_gfc", gfc[yg] * 100, ".2f")
    put("data.dr_peak_gfc_year", yg)
    put("data.dr_2020", rates[2020] * 100, ".2f")
    put("data.ce_2009", stats["2009"]["credit_event"] * 100, ".0f")
    put("data.ce_2020", stats["2020"]["credit_event"] * 100, ".1f")
    put("data.po_2020", stats["2020"]["paid_off"] * 100, ".0f")
    put("data.act_2020", stats["2020"]["still_active"] * 100, ".0f")
    n_ly = sum(sum(v.get(k, 0) for k in ("0", "1", "2")) for v in ly.values())
    n_cens = sum(v.get("-1", 0) + v.get("-2", 0) for v in ly.values())
    put("data.loan_years", n_ly / 1e6, ".1f")
    put("data.panel_loans", flow["panel_loans"] / 1e6, ".1f")
    rows = [
        ("Loans in the 110 origination files", f"{logs_summary['orig_rows']:,}"),
        ("\\quad without any performance record", f"{logs_summary['orig_rows'] - flow['loans_ingested']:,}"),
        ("\\quad outside the 50 states and DC (one with no state)", f"{flow['excluded_no_state_hpi']:,}"),
        ("\\quad no original LTV", f"{flow['excluded_no_ltv']:,}"),
        ("\\quad no original interest rate", f"{flow['excluded_no_rate']:,}"),
        ("Loans kept", f"{flow['loans_kept']:,}"),
        ("\\quad active at one or more year-ends (in the panel)", f"{flow['panel_loans']:,}"),
        ("Loan-years with an outcome, December 1999--2024", f"{n_ly:,}"),
        ("Loan-years censored and dropped (defect or reperforming-sale removal)", f"{n_cens:,}"),
        ("Loans active at December 2025 (last portfolio)", f"{sum(ly['2025'].values()):,}"),
    ]
    body = ("\\begin{table}[t]\\centering\\small\n\\caption{Sample flow.}\\label{tab:flow}\n"
            "\\begin{tabular}{lr}\\toprule\n" + "\n".join(f"{a} & {b} \\\\" for a, b in rows)
            + "\n\\bottomrule\\end{tabular}\n\\end{table}\n")
    table("flow", body.replace(",", "{,}"))


def constants(bd):
    put("const.n_dates", len(bd))
    put("const.n_port", 200_000, "int")
    put("const.n_models", len(list(lgb_grid())) + 1)
    put("const.n_boot", 1000, "int")
    put("const.n_features", len(FEATURES))


def results_numbers(bd, summ, rz):
    b = bd.sort("t")
    rel = b["rel_range"].to_numpy() * 100
    put("res.rel.median", np.median(rel), ".0f")
    put("res.rel.min", rel.min(), ".0f")
    put("res.rel.max", rel.max(), ".0f")
    put("res.rel.max_year", int(b["t"][int(rel.argmax())]))
    put("res.rel.min_year", int(b["t"][int(rel.argmin())]))
    put("res.range_bp.median", np.median(bp(b["set_range"].to_numpy())), ".1f")
    put("res.usd.max", b["usd_range"].max() / 1e9, ".1f")
    put("res.usd.max_year", int(b["t"][int(np.argmax(b["usd_range"]))]))
    put("res.usd.median", np.median(b["usd_range"]) / 1e9, ".1f")
    put("res.set.size.median", int(np.median(b["set_size"])))
    put("res.set.size.min", int(b["set_size"].min()))
    put("res.set.size.max", int(b["set_size"].max()))
    put("res.ratio_refit.median", summ["H1"]["median_ratio_range_to_refit"], ".1f")
    put("res.ratio_policy.median", summ["H2"]["median_ratio_range_to_policy"], ".1f")
    for h in ("H1", "H2", "H4", "H6"):
        put(f"{h}.n", summ[h]["dates_holding"])
        put(f"{h}.verdict", "holds" if summ[h]["holds"] else "fails")
    put("H3.rel2006", summ["H3"]["rel_2006"] * 100, ".0f")
    put("H3.rel2007", summ["H3"]["rel_2007"] * 100, ".0f")
    put("H3.median", summ["H3"]["median_others"] * 100, ".0f")
    put("H3.verdict", "holds" if summ["H3"]["holds"] else "fails")
    put("H5.n_elig", len(summ["H5"]["eligible_dates"]))
    put("H5.inside", summ["H5"]["inside"])
    put("H5.inside25", summ["H5"]["inside_horizon_matched"])
    put("H5.verdict", "holds" if summ["H5"]["holds"] else "fails")
    put("H5.first", summ["H5"]["eligible_dates"][0])
    put("H5.last", summ["H5"]["eligible_dates"][-1])
    put("res.samp_se.median_bp", np.median(bp(b["sampling_se_best"].to_numpy())), ".2f")
    put("res.best_rate.min", b["best_rate"].min() * 100, ".2f")
    put("res.best_rate.max", b["best_rate"].max() * 100, ".2f")
    put("res.best_rate.max_year", int(b["t"][int(b["best_rate"].arg_max())]))
    put("res.flip.median", np.median(b["flip_set"]) * 100, ".1f")
    put("res.flip_refit.median", np.median(b["flip_refit"]) * 100, ".1f")
    put("res.auc.min", b["best_auc"].min(), ".3f")
    put("res.auc.max", b["best_auc"].max(), ".3f")


def more_numbers(bd, summ, ld, runtime):
    b = bd.sort("t")
    best = b["best_rate"].to_numpy()

    def rel(col):
        return b[col].to_numpy() / best * 100

    put("const.hours", runtime["wall_hours_estimate"], ".0f")
    put("res.mono.share", float(np.median(b["set_mono"] / b["set_size"])) * 100, ".0f")
    put("res.rel05.median", np.median(rel("range_0.5se")), ".0f")
    put("res.rel2.median", np.median(rel("range_2se")), ".0f")
    put("res.relauc.median", np.median(rel("range_auc")), ".0f")
    put("res.paired1.size", int(np.median(b["size_paired1"])))
    put("res.paired1.rel", np.median(rel("range_paired1")), ".0f")
    put("res.both.rel.median", np.median((b["both_max"] - b["both_min"]).to_numpy() / best * 100), ".0f")
    put("res.realpath.rel.median",
        np.median(((b["realpath_max"] - b["realpath_min"]) / b["realpath_best"]).to_numpy() * 100), ".0f")
    pol = rel("policy_range")
    put("res.policy.rel.median", np.median(pol), ".0f")
    gfc = pol[np.isin(b["t"].to_numpy(), [2008, 2009, 2010, 2011])]
    put("res.policy.rel.gfc.min", gfc.min(), ".0f")
    put("res.policy.rel.gfc.max", gfc.max(), ".0f")
    put("res.stage2.min", b["stage2_best"].min() * 100, ".0f")
    put("res.stage2.max", b["stage2_best"].max() * 100, ".0f")
    r06 = b.filter(pl.col("t") == 2006)
    put("res.miss2006", (r06["realized"] / r06["set_max"]).item(), ".0f")
    put("H5.above", summ["H5"]["above_range"])
    put("H5.below", summ["H5"]["below_range"])
    ph = summ["post_hoc"]
    put("ph.H1.n", ph["H1_like_for_like"]["dates_set_larger"])
    put("ph.H1.refitmore", len(bd) - ph["H1_like_for_like"]["dates_set_larger"])
    put("ph.H1.ratio", ph["H1_like_for_like"]["median_ratio"], ".2f")
    put("ph.H6.flipset", ph["H6_like_for_like"]["median_flip_set20"] * 100, ".1f")
    put("ph.H6.fliprefit", ph["H6_like_for_like"]["median_flip_refit"] * 100, ".1f")
    put("ph.paired2.size", int(ph["paired2_size_median"]))
    put("ph.paired2.rel", ph["paired2_rel_range_median"] * 100, ".0f")
    m = (ld.filter(pl.col("band").is_not_null() & (pl.col("band") != "null"))
         .group_by("by", "band").agg(pl.col("ratio_med").median()))
    put("res.loan.ratio.min", m["ratio_med"].min(), ".1f")
    put("res.loan.ratio.max", m["ratio_med"].max(), ".1f")


def review_numbers(bd, summ):
    """Numbers for the analyses added after the referee review: refit quality,
    independent-sample refits, delinquency and the macro values the forecast held."""
    b = bd.sort("t")
    ph = summ["post_hoc"]
    rq, rs = ph["refit_quality"], ph["resample"]
    put("ph.rq.gap", rq["median_gap_se"], ".1f")
    put("ph.rq.gt1", rq["refits_gt1se"])
    put("ph.rq.total", 20 * len(b))
    put("ph.rq.gt1.share", rq["refits_gt1se"] / (20 * len(b)) * 100, ".0f")
    put("ph.rq.bias", rq["median_bias"] * 100, ".0f")
    put("ph.rq.above", rq["dates_mean_refit_above_best"])
    put("ph.rs.gap", rs["median_gap_se"], ".2f")
    put("ph.rs.bias", rs["median_bias"] * 100, ".0f")
    put("ph.rs.bias.pos", int((b["rsm_bias"] > 0).sum()))
    put("ph.rs.n", rs["dates_set20_larger_than_resample_range"])
    put("ph.rs.ratio", rs["median_ratio"], ".2f")
    put("ph.rs.w95", rs["dates_set_range_gt_resample_w95"])
    put("ph.rs.flipmore", rs["dates_flip_set20_gt_resample"])
    put("ph.rs.flip", rs["median_flip_resample"] * 100, ".1f")
    put("ph.rs.rel.median", float(np.median((b["rsm_range"] / b["best_rate"]).to_numpy())) * 100, ".0f")
    put("res.fam.mono.median", float(np.median(b["fam_mono_rel"].to_numpy())) * 100, ".0f")
    put("res.fam.lgbm.median", float(np.median(b["fam_lgbm_rel"].to_numpy())) * 100, ".0f")
    put("ph.lfl.rel.median", float(np.median((b["lfl_range20"] / b["best_rate"]).to_numpy())) * 100, ".0f")
    # overlap of the independent samples with the original fitting sample
    ov = {t: json.loads((RESULTS / "dates" / str(t) / "resample.json").read_text())["overlap_original"]
          for t in b["t"].to_list()}
    disj = [t for t, o in ov.items() if max(o) == 0]
    assert disj == list(range(min(disj), 2026)), disj
    put("ph.rs.disj.n", len(disj))
    put("ph.rs.disj.first", min(disj))
    mean_ov = {t: float(np.mean(o)) for t, o in ov.items()}
    put("ph.rs.ov.max", max(mean_ov.values()) * 100, ".0f")
    put("ph.rs.ov.max_year", max(mean_ov, key=mean_ov.get))
    dj = b.filter(pl.col("t").is_in(disj))
    put("ph.rs.disj.refitmore", int((dj["rsm_range"] > dj["lfl_range20"]).sum()))
    # current delinquency of the portfolio (every active loan, not only the scored sample)
    dq = pl.read_csv(RESULTS / "delinquency.csv")
    tot = dq.group_by("y").agg(pl.col("upb").sum().alias("tot"))
    sh = (dq.filter(pl.col("g") != "current").group_by("y").agg(pl.col("upb").sum())
          .join(tot, on="y").with_columns((pl.col("upb") / pl.col("tot")).alias("s")))
    put("dq.share.min", sh["s"].min() * 100, ".1f")
    put("dq.share.max", sh["s"].max() * 100, ".1f")
    put("dq.share.max_year", int(sh.sort("s")["y"][-1]))
    d09 = dq.filter(pl.col("y") == 2009)
    for g, k in (("current", "cur"), ("30-89", "d30"), ("90-179", "d90")):
        r = d09.filter(pl.col("g") == g).row(0, named=True)
        put(f"dq.2009.{k}", r["dflt"] / r["known"] * 100, ".1f" if k != "cur" else ".2f")
    # the macro values the primary forecast held for two years (median state)
    sy = pl.read_parquet(MACRO / "state_year.parquet")
    for y in (2005, 2006):
        put(f"mac.hpi{y}", float(sy.filter(pl.col("year") == y)["hpi_g"].median()) * 100, ".1f")


def review2_numbers(bd, summ):
    """Numbers for the checks added after the second referee review (07b_posthoc.py)."""
    p2 = pl.read_csv(RESULTS / "posthoc2.csv").sort("t")
    b = bd.sort("t")
    put("p2.pos.median", p2["best_pos"].median() * 100, ".0f")
    put("p2.below_median", int((p2["best_rate"] < p2["set_median"]).sum()))
    put("p2.median_gap", float(((p2["set_median"] / p2["best_rate"]) - 1).median()) * 100, ".0f")
    put("p2.rho_set.pos", int((p2["rho_set"] > 0).sum()))
    put("p2.rho_set.median", p2["rho_set"].median(), ".2f")
    put("p2.rho_rsm.median", p2["rho_rsm"].median(), ".2f")
    put("p2.rho_rsm.pos", int((p2["rho_rsm"] > 0).sum()))
    put("p2.h4.n", int((p2["h4_mono_sub"] < p2["h4_lgbm_sub"]).sum()))
    put("p2.h4.mono", p2["h4_mono_sub"].median() * 100, ".0f")
    put("p2.h4.lgbm", p2["h4_lgbm_sub"].median() * 100, ".0f")
    put("p2.h4.smaller", int((p2["h4_k"] < 32).sum()))
    e = p2.filter(pl.col("t") <= 2017)
    put("p2.in_refit", int(e["realized_in_refit"].sum()))
    put("p2.in_rsm", int(e["realized_in_rsm"].sum()))
    dj = b.filter(pl.col("t") >= 2018)
    put("p2.disj.rsm", float((dj["rsm_range"] / dj["best_rate"]).median()) * 100, ".0f")
    put("p2.disj.lfl", float((dj["lfl_range20"] / dj["best_rate"]).median()) * 100, ".0f")
    put("p2.boot.rel", float((b["refit_range"] / b["best_rate"]).median()) * 100, ".0f")
    put("p2.boot_gt_rsm", int((b["refit_range"] > b["rsm_range"]).sum()))
    put("p2.paired1.single", int((b["size_paired1"] == 1).sum()))
    put("res.policy.rel.median1", float((b["policy_range"] / b["best_rate"]).median()) * 100, ".1f")
    for y in (2007, 2021):
        r = b.filter(pl.col("t") == y)
        put(f"res.policy.rel.{y}", (r["policy_range"] / r["best_rate"]).item() * 100, ".0f")
    pre = p2.join(b.select("t", "dec_sl1_tlgd", "dec_real_tlgd", "realized"), on="t").filter(pl.col("t") <= 2008)
    f19 = (pre["dec_real_lgd19"] / pre["dec_real_tlgd"]).to_numpy()
    put("dec.lgd19.min", f19.min(), ".1f")
    put("dec.lgd19.max", f19.max(), ".1f")
    gap = (pre["dec_real_lgd19"] / pre["realized"] - 1).to_numpy()
    over = pre.filter(pl.col("dec_real_lgd19") > pl.col("realized"))["t"].to_list()
    assert over == [2006], over
    put("dec.lgd19.short.max", float(-gap[gap < 0].min()) * 100, ".0f")
    put("dec.lgd19.over2006", float(gap[pre["t"].to_list().index(2006)]) * 100, ".0f")
    post = p2.join(b.select("t", "realized"), on="t").filter((pl.col("t") >= 2010) & (pl.col("t") <= 2017))
    assert (post["dec_real_lgd19"] > post["realized"]).all()
    pdp = p2.filter(pl.col("t") <= 2008)
    put("pd.pre.min", float((pdp["pd_real_best"] / pdp["default_share"]).min()), ".2f")
    put("pd.pre.max", float((pdp["pd_real_best"] / pdp["default_share"]).max()), ".2f")
    pdl = p2.filter((pl.col("t") >= 2013) & (pl.col("t") <= 2017))
    put("pd.late.min", float((pdl["pd_real_best"] / pdl["default_share"]).min()), ".2f")
    put("pd.late.max", float((pdl["pd_real_best"] / pdl["default_share"]).max()), ".2f")
    put("data.kept", flow_kept(), "int")
    fr = {t: json.loads((RESULTS / "dates" / str(t) / "accuracy.json").read_text())["info"]["sample_fraction"]
          for t in b["t"].to_list()}
    mut = {t: max(0.0, (f - 1 / 21) / f) for t, f in fr.items()}
    put("ph.rs.mut.max", max(mut.values()) * 100, ".0f")
    put("ph.rs.mut.max_year", max(mut, key=mut.get))
    put("ph.frac.min", min(fr.values()) * 100, ".1f")
    put("ph.frac.min_year", min(fr, key=fr.get))


def flow_kept() -> int:
    return json.loads((RESULTS / "flow.json").read_text())["loans_kept"]


def decomp_table(bd):
    """H5 decomposition (post hoc) for the dates with eight or more years observed."""
    e = bd.filter(pl.col("years_observed") >= 8).sort("t")
    pre, post = e.filter(pl.col("t") <= 2008), e.filter(pl.col("t") >= 2011)
    f_path = (pre["dec_real_tlgd"] / pre["dec_sl1_tlgd"]).to_numpy()
    f_lgd = (pre["dec_real_flgd"] / pre["dec_real_tlgd"]).to_numpy()
    assert (pre["dec_real_flgd"] < pre["realized"]).all()
    put("dec.path.min", f_path.min(), ".1f")
    put("dec.path.max", f_path.max(), ".0f")
    put("dec.lgd.min", f_lgd.min(), ".1f")
    put("dec.lgd.max", f_lgd.max(), ".1f")
    put("dec.pre.below_lb", int((pre["dec_real_flgd"] < pre["realized_lb"]).sum()))
    put("dec.pre.n", len(pre))
    # from 2011 each step lowers the allowance
    assert ((post["dec_real_tlgd"] < post["dec_sl1_tlgd"]) & (post["dec_real_flgd"] < post["dec_real_tlgd"])).all()
    late = e.filter(pl.col("t") >= 2013)
    put("dec.late.maxdev", float((late["dec_real_flgd"] / late["realized"] - 1).abs().max()) * 100, ".0f")
    p2 = {r["t"]: r for r in pl.read_csv(RESULTS / "posthoc2.csv").iter_rows(named=True)}
    rows = []
    for r in bd.sort("t").iter_rows(named=True):
        if r["years_observed"] < 8:
            continue
        q = p2[r["t"]]
        rows.append(" & ".join([
            str(r["t"]), f"{r['dec_sl1_tlgd'] * 100:.2f}", f"{r['dec_real_tlgd'] * 100:.2f}",
            f"{q['dec_real_lgd19'] * 100:.2f}", f"{r['dec_real_flgd'] * 100:.2f}",
            f"{r['realized'] * 100:.2f}", f"{r['realized_lb'] * 100:.2f}",
            f"{r['imputed_share'] * 100:.0f}", f"{q['pd_real_best'] * 100:.1f}",
            f"{q['default_share'] * 100:.1f}"]) + " \\\\")
    body = ("\\begin{table}[t]\\centering\\footnotesize\n\\caption{Where the miss comes from (added after the "
            "results). Lifetime allowance of the most accurate model, \\% of balance: (1) primary forecast and the "
            "severity table at $T$; (2) realized macro path, severity at $T$; (3) realized path and the severity "
            "table of defaults resolved by December 2019, before pandemic forbearance; (4) realized path and the "
            "table of defaults resolved by December 2025, which counts pandemic-era payoffs after default as "
            "zero-loss resolutions. Realized loss: recorded estimate, lower bound with unresolved defaults at zero, "
            "and the imputed share, valued at the severity of all resolutions through June 2026. Defaults: the "
            "model's mean lifetime PD on the realized path, and the share of loans that defaulted by June 2026.}"
            "\\label{tab:decomp}\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{rrrrrrrrrr}\\toprule\n"
            "& \\multicolumn{4}{c}{Allowance} & \\multicolumn{3}{c}{Realized loss} & \\multicolumn{2}{c}{Defaults (\\%)} \\\\\n"
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}\n"
            "Dec. & (1) & (2) & (3) & (4) & Est. & Lower & Imp.\\,\\% & PD & Actual \\\\ \\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n\\end{table}\n")
    table("decomp", body)


def judicial_numbers():
    """Band-and-date comparisons where both cells have 30+ resolutions:
    how often judicial-state severity exceeds power-of-sale severity."""
    n = hi = 0
    for t in range(2005, 2026):
        d = pl.read_csv(RESULTS / "dates" / str(t) / "lgd.csv")
        for band in range(6):
            p = d.filter((pl.col("band") == band) & ~pl.col("judicial")).row(0, named=True)
            j = d.filter((pl.col("band") == band) & pl.col("judicial")).row(0, named=True)
            if p["source"] == "cell" and j["source"] == "cell":
                n += 1
                hi += j["lgd"] > p["lgd"]
    put("data.jud.n", n)
    put("data.jud.higher", hi)


def per_date_table(bd):
    lines = []
    for r in bd.sort("t").iter_rows(named=True):
        real = (f"{r['realized'] * 100:.2f}" if r.get("years_observed", 0) >= 8 else
                f"({r['realized'] * 100:.2f})") if r.get("realized") is not None else "--"
        lines.append(" & ".join([
            str(r["t"]), f"{r['set_size']}", f"{r['best_auc']:.3f}", f"{r['best_rate'] * 100:.2f}",
            f"{r['set_min'] * 100:.2f}--{r['set_max'] * 100:.2f}", f"{r['rel_range'] * 100:.0f}",
            f"{bp(r['refit_w95']):.1f}", f"{bp(r['policy_range']):.1f}", f"{r['usd_range'] / 1e9:.1f}", real]) + " \\\\")
    body = ("\\begin{table}[H]\\centering\\footnotesize\n\\caption{Allowance by reporting date. "
            "Rates are lifetime expected loss as a percentage of balance on the scored sample; "
            "set range in \\% of balance; refit and policy widths in basis points of balance; \\$ range scales the set's range to the whole "
            "active book. Realized loss in parentheses where fewer than eight years are observed.}"
            "\\label{tab:bydate}\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{rrrrrrrrrr}\\toprule\n"
            "& Set & Best & Best & Set range & Rel. & Refit & Policy & \\$ range & Realized \\\\\n"
            "Dec. & size & AUC & rate (\\%) & (\\%) & range (\\%) & 95\\% (bp) & range (bp) & (bn) & (\\%) \\\\ \\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\n\\end{table}\n")
    table("bydate", body)


def hypotheses_table(bd, summ):
    n = summ["n_dates"]
    rows = [
        ("H1", "Model choice $>$ refit noise", f"{summ['H1']['dates_holding']} of {n} dates", summ["H1"]["holds"]),
        ("H2", "Model choice $\\ge$ reversion range (1 vs 3 years)", f"{summ['H2']['dates_holding']} of {n} dates", summ["H2"]["holds"]),
        ("H3", "Relative range larger at 2006 and 2007",
         f"{summ['H3']['rel_2006'] * 100:.0f}\\%, {summ['H3']['rel_2007'] * 100:.0f}\\% vs {summ['H3']['median_others'] * 100:.0f}\\%",
         summ["H3"]["holds"]),
        ("H4", "Monotone family narrower", f"{summ['H4']['dates_holding']} of {n} dates", summ["H4"]["holds"]),
        ("H5", "Realized loss inside the range at $<$ half",
         f"{summ['H5']['inside']} of {len(summ['H5']['eligible_dates'])} dates", summ["H5"]["holds"]),
        ("H6", "Stage flips $>$ refit flips", f"{summ['H6']['dates_holding']} of {n} dates", summ["H6"]["holds"]),
    ]
    body = ("\\begin{table}[t]\\centering\\small\n\\caption{The six predictions recorded before the data were "
            "opened, as tested.}\\label{tab:hyp}\n\\begin{tabular}{llll}\\toprule\n"
            "& Prediction & Evidence & Verdict \\\\ \\midrule\n"
            + "\n".join(f"{a} & {b} & {c} & {'Holds' if d else '\\textbf{Fails}'} \\\\" for a, b, c, d in rows)
            + "\n\\bottomrule\\end{tabular}\n\\end{table}\n")
    table("hypotheses", body)


def lgd_table():
    rows = []
    for t in (2006, 2008, 2012, 2017, 2025):
        f = RESULTS / "dates" / str(t) / "lgd.csv"
        if not f.exists():
            continue
        d = pl.read_csv(f).sort("band", "judicial")
        cells = []
        for b in range(6):
            for j in (False, True):
                r = d.filter((pl.col("band") == b) & (pl.col("judicial") == j)).row(0, named=True)
                mark = "" if r["source"] == "cell" else "$^\\dagger$"
                cells.append(f"{r['lgd'] * 100:.0f}{mark}")
        rows.append(f"{t} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for lab in
                      ("$<$60", "60--80", "80--90", "90--100", "100--120", "$\\ge$120"))
    body = ("\\begin{table}[H]\\centering\\footnotesize\n\\caption{Loss given default (\\% of balance at default) "
            "by current-LTV band at the start of the default year; P = power of sale, J = judicial. "
            "$^\\dagger$ fewer than 30 resolutions in the cell, so the band or overall mean is used.}"
            "\\label{tab:lgd}\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{r" + "rr" * 6 + "}\\toprule\n"
            f"Dec. & {head} \\\\\n & " + " & ".join(["P", "J"] * 6) + " \\\\ \\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n\\end{table}\n")
    table("lgd", body)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    flow = json.loads((RESULTS / "flow.json").read_text())
    logs_summary = json.loads((RESULTS / "ingest_summary.json").read_text())
    stats = json.loads((RESULTS / "data_stats.json").read_text())
    data_numbers(flow, logs_summary, stats)
    bd = pl.read_csv(RESULTS / "by_date.csv")
    summ = json.loads((RESULTS / "summary.json").read_text())
    rz = pl.read_csv(RESULTS / "realized.csv")
    constants(bd)
    results_numbers(bd, summ, rz)
    more_numbers(bd, summ, pl.read_csv(RESULTS / "loan_dispersion.csv"),
                 json.loads((RESULTS / "runtime.json").read_text()))
    review_numbers(bd, summ)
    review2_numbers(bd, summ)
    judicial_numbers()
    per_date_table(bd)
    decomp_table(bd)
    hypotheses_table(bd, summ)
    lgd_table()
    lines = ["% Generated by code/09_paper.py. Do not edit.", "\\makeatletter",
             "\\newcommand{\\num}[1]{\\@ifundefined{num@#1}{\\errmessage{Undefined number: #1}}{\\@nameuse{num@#1}}}"]
    lines += [f"\\@namedef{{num@{k}}}{{{v}}}" for k, v in sorted(NUM.items())]
    lines.append("\\makeatother")
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")
    used = set()
    for f in (PAPER / "sections").glob("*.tex"):
        used |= set(re.findall(r"\\num\{([^}]+)\}", f.read_text()))
    missing = sorted(used - set(NUM))
    assert not missing, f"numbers used but not defined: {missing}"
    print(f"{len(NUM)} numbers, {len(used)} used")
    plain_abstract()


def plain_abstract() -> None:
    """The abstract as plain text for the SSRN form, numbers substituted."""
    tex = (PAPER / "sections" / "00_abstract.tex").read_text()
    txt = re.sub(r"\\num\{([^}]+)\}", lambda m: NUM[m.group(1)].replace("{,}", ","), tex)
    txt = txt.replace("\\%", "%").replace("~", " ").replace("--", "\u2013")
    assert "\\" not in txt and "{" not in txt, "LaTeX left in the abstract"
    txt = " ".join(txt.split())
    out = PAPER / "submission-ssrn"
    out.mkdir(exist_ok=True)
    (out / "abstract.txt").write_text(txt + "\n")
    print(f"abstract: {len(txt.split())} words")


if __name__ == "__main__":
    main()
