"""Independent audit of Paper 10.

Usage: python paper/audit.py

Recomputes, from the per-date raw outputs (results/dates/T/*.csv and *.json)
and with code that does not import the analysis scripts, the sets of equally
accurate models, every allowance range, the six verdicts, the post-hoc
checks and every number printed in the paper (paper/numbers.tex), and checks
the text's verdict words, tables, file hygiene and the firewall. Exits
non-zero on any failure.
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PAPER = ROOT / "paper"
DATES = list(range(2005, 2026))
FAIL: list[str] = []
N_OK = 0


def check(cond: bool, msg: str) -> None:
    global N_OK
    if cond:
        N_OK += 1
    else:
        FAIL.append(msg)


def rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def pctl(x: list[float], q: float) -> float:
    """numpy's default (linear) percentile, written out."""
    s = sorted(x)
    h = (len(s) - 1) * q / 100
    lo = math.floor(h)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (h - lo)


# ---------------------------------------------------------------- numbers.tex
NUM = dict(re.findall(r"^\\@namedef\{num@([^}]+)\}\{(.*)\}$", (PAPER / "numbers.tex").read_text(), re.M))


def num(key: str) -> float:
    return float(NUM[key].replace("{,}", ""))


def printed(key: str, value: float, digits: int) -> None:
    check(key in NUM, f"{key} not printed")
    if key in NUM:
        want = round(value, digits)
        check(abs(num(key) - want) <= 0.5 * 10 ** -digits + 1e-9,
              f"{key}: printed {NUM[key]}, recomputed {value:.6g}")


# ---------------------------------------------------------------- per date, from raw outputs
by = {int(r["t"]): r for r in rows(RES / "by_date.csv")}
check(sorted(by) == DATES, "by_date.csv does not hold exactly the 21 dates")
rec = {}
for t in DATES:
    d = RES / "dates" / str(t)
    acc = json.loads((d / "accuracy.json").read_text())
    models = [r for r in rows(d / "models.csv") if r["target"] == "default"]
    check(len(models) == 65, f"{t}: {len(models)} default models")
    ll = {int(r["model"]): float(r["logloss"]) for r in models}
    auc = {int(r["model"]): float(r["auc"]) for r in models}
    best = min(ll, key=ll.get)
    check(best == acc["default"]["best"], f"{t}: best model differs")
    se = acc["default"]["se_logloss"]
    s1 = sorted(m for m in ll if ll[m] <= ll[best] + se)
    check(s1 == sorted(acc["default"]["sets"]["ll_1se"]), f"{t}: recorded set differs from recomputation")
    for k, key in ((0.5, "ll_0.5se"), (2, "ll_2se")):
        check(sorted(m for m in ll if ll[m] <= ll[best] + k * se) == sorted(acc["default"]["sets"][key]), f"{t}: {key}")
    ba = max(auc, key=auc.get)
    check(sorted(m for m in auc if auc[m] >= auc[ba] - acc["default"]["se_auc"]) == sorted(acc["default"]["sets"]["auc_1se"]),
          f"{t}: AUC set")
    fams = {int(r["model"]): r["family"] for r in models}
    check(fams[best] == "lgbm", f"{t}: best is not unconstrained LightGBM")
    check(sum(f == "lgbm" for f in fams.values()) == 32 and sum(f == "lgbm_mono" for f in fams.values()) == 32,
          f"{t}: family counts")

    al = rows(d / "allowance.csv")
    prim = {r["model"]: float(r["allowance_rate"]) for r in al
            if r["set"] == "default" and r["policy"] == "sl1" and r["origination"] == "false"}
    check(len(prim) == 65, f"{t}: {len(prim)} primary allowances")
    S = [f"m{m:03d}" for m in s1]
    rs = [prim[m] for m in S]
    b_rate = prim[f"m{best:03d}"]
    ref = [float(r["allowance_rate"]) for r in al if r["set"] == "refit" and r["policy"] == "sl1" and r["origination"] == "false"]
    check(len(ref) == 20, f"{t}: {len(ref)} refits")
    pol = [float(r["allowance_rate"]) for r in al if r["set"] == "default" and r["model"] == f"m{best:03d}"
           and r["policy"] in ("sl1", "immediate", "sl3") and r["origination"] == "false"]
    check(len(pol) == 3, f"{t}: policy rows")
    real_p = {r["model"]: float(r["allowance_rate"]) for r in al if r["set"] == "default" and r["policy"] == "realized"}
    check(set(real_p) == set(S), f"{t}: realized-path rows are not exactly the set")
    rp = [real_p[m] for m in S]
    rec[t] = {"best_rate": b_rate, "set_min": min(rs), "set_max": max(rs), "set_range": max(rs) - min(rs),
              "refit_w95": pctl(ref, 97.5) - pctl(ref, 2.5), "policy_range": max(pol) - min(pol),
              "set_size": len(S), "rel": (max(rs) - min(rs)) / b_rate,
              "realpath_rel": (max(rp) - min(rp)) / real_p[f"m{best:03d}"], "auc": auc[best]}
    for k in ("best_rate", "set_min", "set_max", "set_range", "refit_w95", "policy_range"):
        check(close(rec[t][k], float(by[t][k])), f"{t}: {k} differs from by_date.csv")
    check(int(by[t]["set_size"]) == len(S), f"{t}: set size")
    for f in ("lgbm", "lgbm_mono"):
        fs = acc["default"]["family"][f]
        idx = [m for m in ll if fams[m] == f]
        fb = min(idx, key=ll.get)
        check(fb == fs["best"], f"{t}: family best {f}")
        fset = [m for m in idx if ll[m] <= ll[fb] + fs["se_logloss"]]
        check(sorted(fset) == sorted(fs["set"]), f"{t}: family set {f}")
        v = [prim[f"m{m:03d}"] for m in fset]
        key = "fam_lgbm_rel" if f == "lgbm" else "fam_mono_rel"
        rec[t][key] = (max(v) - min(v)) / prim[f"m{fb:03d}"]
        check(close(rec[t][key], float(by[t][key])), f"{t}: {key}")
    port = json.loads((d / "portfolio.json").read_text())
    check(close(float(by[t]["usd_range"]), rec[t]["set_range"] * port["upb_active"]), f"{t}: dollar range")
    check(port["n_scored"] == 200_000, f"{t}: scored loans")
    # sampling and prepay files sane
    check(float(by[t]["sampling_se_best"]) < 0.05 * rec[t]["set_range"] or rec[t]["set_range"] == 0,
          f"{t}: sampling error not small next to the range")
    lgd = rows(d / "lgd.csv")
    check(len(lgd) == 12 and all(0 <= float(r["lgd"]) <= 1 for r in lgd), f"{t}: LGD table out of [0, 1]")

# ---------------------------------------------------------------- verdicts
rz = {int(r["t"]): r for r in rows(RES / "realized.csv")}
summ = json.loads((RES / "summary.json").read_text())
n = len(DATES)
h1 = sum(rec[t]["set_range"] > rec[t]["refit_w95"] for t in DATES)
h2 = sum(rec[t]["set_range"] >= rec[t]["policy_range"] for t in DATES)
others = st.median([rec[t]["rel"] for t in DATES if t not in (2006, 2007)])
h3 = rec[2006]["rel"] > others and rec[2007]["rel"] > others
h4 = sum(rec[t]["fam_mono_rel"] < rec[t]["fam_lgbm_rel"] for t in DATES)
elig = [t for t in DATES if float(rz[t]["years_observed"]) >= 8]
check(elig == list(range(2005, 2018)), f"H5 eligible dates {elig}")
inside = [t for t in elig if rec[t]["set_min"] <= float(rz[t]["realized_rate"]) <= rec[t]["set_max"]]
inside_lb = [t for t in elig if rec[t]["set_min"] <= float(rz[t]["realized_rate_lb"]) <= rec[t]["set_max"]]
above = sum(float(rz[t]["realized_rate"]) > rec[t]["set_max"] for t in elig)
below = sum(float(rz[t]["realized_rate"]) < rec[t]["set_min"] for t in elig)
h6 = sum(float(by[t]["flip_set"]) > float(by[t]["flip_refit"]) for t in DATES)
want = {"H1": (h1, h1 > n / 2), "H2": (h2, h2 > n / 2), "H4": (h4, h4 > n / 2), "H6": (h6, h6 > n / 2)}
for h, (cnt, holds) in want.items():
    check(summ[h]["dates_holding"] == cnt and summ[h]["holds"] == holds, f"{h} verdict differs")
    check(int(num(f"{h}.n")) == cnt and NUM[f"{h}.verdict"] == ("holds" if holds else "fails"), f"{h} printed")
check(summ["H3"]["holds"] == h3 and NUM["H3.verdict"] == ("holds" if h3 else "fails"), "H3 verdict")
check(summ["H5"]["inside"] == len(inside) and summ["H5"]["holds"] == (len(inside) < len(elig) / 2), "H5 verdict")
check(NUM["H5.verdict"] == ("holds" if len(inside) < len(elig) / 2 else "fails"), "H5 printed verdict")
check(int(num("H5.inside")) == len(inside) and inside == [2009], f"H5 inside dates {inside}")
check(len(inside_lb) == 0 and summ["post_hoc"]["H5_lower_bound_inside"] == 0, "H5 lower bound")
check(int(num("H5.above")) == above == 4 and int(num("H5.below")) == below == 8, "H5 above/below")
check(above + below + len(inside) == len(elig), "H5 partition")
printed("H3.rel2006", rec[2006]["rel"] * 100, 0)
printed("H3.rel2007", rec[2007]["rel"] * 100, 0)
printed("H3.median", others * 100, 0)

# like-for-like post hoc: values must agree with by_date and the summary
lr = [float(by[t]["lfl_range20"]) / float(by[t]["refit_range"]) for t in DATES]
check(summ["post_hoc"]["H1_like_for_like"]["dates_set_larger"] == sum(x > 1 for x in lr), "H1 like-for-like")
check(int(num("ph.H1.n")) == sum(x > 1 for x in lr), "H1 like-for-like printed")
check(int(num("ph.H1.refitmore")) == n - sum(x > 1 for x in lr), "refits-span-more printed")
printed("ph.H1.ratio", st.median(lr), 2)
lf = sum(float(by[t]["lfl_flip20"]) > float(by[t]["flip_refit"]) for t in DATES)
check(lf == summ["post_hoc"]["H6_like_for_like"]["dates_set_larger"] == 0, "H6 like-for-like: set never larger")
printed("ph.H6.flipset", st.median(float(by[t]["lfl_flip20"]) for t in DATES) * 100, 1)
printed("ph.H6.fliprefit", st.median(float(by[t]["flip_refit"]) for t in DATES) * 100, 1)
printed("res.flip.median", st.median(float(by[t]["flip_set"]) for t in DATES) * 100, 1)
printed("res.flip_refit.median", st.median(float(by[t]["flip_refit"]) for t in DATES) * 100, 1)

# ---------------------------------------------------------------- headline numbers
rel = [rec[t]["rel"] * 100 for t in DATES]
printed("res.rel.median", st.median(rel), 0)
printed("res.rel.min", min(rel), 0)
printed("res.rel.max", max(rel), 0)
check(int(num("res.rel.max_year")) == DATES[rel.index(max(rel))], "max year")
check(int(num("res.rel.min_year")) == DATES[rel.index(min(rel))], "min year")
usd = [float(by[t]["usd_range"]) / 1e9 for t in DATES]
printed("res.usd.max", max(usd), 1)
check(int(num("res.usd.max_year")) == DATES[usd.index(max(usd))], "usd max year")
printed("res.usd.median", st.median(usd), 1)
printed("res.range_bp.median", st.median(rec[t]["set_range"] * 1e4 for t in DATES), 1)
printed("res.ratio_refit.median", st.median(rec[t]["set_range"] / rec[t]["refit_w95"] for t in DATES), 1)
printed("res.ratio_policy.median", st.median(rec[t]["set_range"] / rec[t]["policy_range"] for t in DATES), 1)
printed("res.best_rate.min", min(rec[t]["best_rate"] for t in DATES) * 100, 2)
printed("res.best_rate.max", max(rec[t]["best_rate"] for t in DATES) * 100, 2)
printed("res.auc.min", min(rec[t]["auc"] for t in DATES), 3)
printed("res.auc.max", max(rec[t]["auc"] for t in DATES), 3)
sizes = [rec[t]["set_size"] for t in DATES]
check(int(num("res.set.size.median")) == int(st.median(sizes)) and int(num("res.set.size.min")) == min(sizes)
      and int(num("res.set.size.max")) == max(sizes), "set sizes printed")
printed("res.realpath.rel.median", st.median(rec[t]["realpath_rel"] for t in DATES) * 100, 0)
printed("res.policy.rel.median", st.median(rec[t]["policy_range"] / rec[t]["best_rate"] for t in DATES) * 100, 0)
gfc = [rec[t]["policy_range"] / rec[t]["best_rate"] * 100 for t in (2008, 2009, 2010, 2011)]
printed("res.policy.rel.gfc.min", min(gfc), 0)
printed("res.policy.rel.gfc.max", max(gfc), 0)
printed("res.miss2006", float(rz[2006]["realized_rate"]) / rec[2006]["set_max"], 0)
check(float(rz[2006]["realized_rate"]) / rec[2006]["set_max"] > 10, "2006 miss below ten-fold")
# claims in the text
check(sum(rec[t]["refit_w95"] / rec[t]["best_rate"] > 0.5 for t in DATES) >= 3,
      "'at several dates' refit width exceeds half the allowance")
logit_dates = [t for t in DATES if int(by[t]["set_logit"]) > 0]
check(logit_dates == [2005, 2006, 2007, 2008, 2009], f"spline logit dates {logit_dates}")
check(h4 == (n // 2) + 2, "H4 'one more than a majority'")
# sensitivity
for key, col in (("res.rel05.median", "range_0.5se"), ("res.rel2.median", "range_2se"),
                 ("res.relauc.median", "range_auc"), ("res.paired1.rel", "range_paired1")):
    printed(key, st.median(float(by[t][col]) / rec[t]["best_rate"] for t in DATES) * 100, 0)
printed("ph.paired2.rel", st.median(float(by[t]["range_paired2"]) / rec[t]["best_rate"] for t in DATES) * 100, 0)
check(int(num("ph.paired2.size")) == int(st.median(int(by[t]["size_paired2"]) for t in DATES)), "paired2 size")
check(int(num("res.paired1.size")) == int(st.median(int(by[t]["size_paired1"]) for t in DATES)), "paired1 size")

# ---------------------------------------------------------------- refits (post hoc, after the referee review)
gap_b, gap_r, bias_b, bias_r, rr, gt1, ov_mean, disj = [], [], [], [], {}, 0, {}, []
for t in DATES:
    d = RES / "dates" / str(t)
    acc = json.loads((d / "accuracy.json").read_text())
    rj = json.loads((d / "resample.json").read_text())
    llb, se = acc["default"]["best_logloss"], acc["default"]["se_logloss"]
    check(rj["config"] == acc["default"]["best"] and len(rj["logloss"]) == 20, f"{t}: resample config/count")
    gb = [(x - llb) / se for x in acc["refit"]["logloss"]]
    gr = [(x - llb) / se for x in rj["logloss"]]
    check(len(gb) == 20, f"{t}: bootstrap refit log losses")
    gap_b.append(st.median(gb))
    gap_r.append(st.median(gr))
    gt1 += sum(g > 1 for g in gb)
    al = rows(d / "allowance.csv")
    ref = [float(r["allowance_rate"]) for r in al if r["set"] == "refit" and r["policy"] == "sl1" and r["origination"] == "false"]
    rsm = [float(r["allowance_rate"]) for r in al if r["set"] == "resample" and r["policy"] == "sl1" and r["origination"] == "false"]
    check(len(rsm) == 20, f"{t}: {len(rsm)} resample allowances")
    bias_b.append(st.mean(ref) / rec[t]["best_rate"] - 1)
    bias_r.append(st.mean(rsm) / rec[t]["best_rate"] - 1)
    rr[t] = max(rsm) - min(rsm)
    check(close(rr[t], float(by[t]["rsm_range"])), f"{t}: resample range")
    ov_mean[t] = st.mean(rj["overlap_original"])
    if max(rj["overlap_original"]) == 0:
        disj.append(t)
    # H5 decomposition from the cell masses and the two severity tables
    mass = rows(d / "el_mass.csv")
    lg_t = {(r["band"], r["judicial"]): float(r["lgd"]) for r in rows(d / "lgd.csv")}
    lg_f = {(r["band"], r["judicial"]): float(r["lgd"]) for r in rows(RES / "dates" / "2025" / "lgd.csv")}
    for pol, tab, col in (("sl1", lg_t, "dec_sl1_tlgd"), ("realized", lg_t, "dec_real_tlgd"), ("realized", lg_f, "dec_real_flgd")):
        mm = [r for r in mass if r["policy"] == pol]
        check(len(mm) == 12, f"{t}: el_mass cells {pol}")
        v = sum(float(r["mass"]) * tab[(r["band"], r["judicial"])] for r in mm) / float(mm[0]["upb_scored"])
        rec[t][col] = v
        check(close(v, float(by[t][col]), 1e-6), f"{t}: {col}")
    check(close(rec[t]["dec_sl1_tlgd"], rec[t]["best_rate"], 1e-6), f"{t}: decomposition base equals best allowance")
printed("ph.rq.gap", st.median(gap_b), 1)
printed("ph.rs.gap", st.median(gap_r), 2)
check(int(num("ph.rq.gt1")) == gt1 and int(num("ph.rq.total")) == 20 * n, "bootstrap refits outside tolerance")
printed("ph.rq.gt1.share", gt1 / (20 * n) * 100, 0)
printed("ph.rq.bias", st.median(bias_b) * 100, 0)
check(int(num("ph.rq.above")) == sum(x > 0 for x in bias_b), "bootstrap refit mean above best")
printed("ph.rs.bias", st.median(bias_r) * 100, 0)
check(int(num("ph.rs.bias.pos")) == sum(x > 0 for x in bias_r), "resample mean above best")
check(all(g > 0 for g in gap_b), "text: bootstrap refits worse than the best at every date")
lfl = {t: float(by[t]["lfl_range20"]) for t in DATES}
ratio_rs = [lfl[t] / rr[t] for t in DATES]
check(int(num("ph.rs.n")) == sum(x > 1 for x in ratio_rs), "set20 vs resample range")
printed("ph.rs.ratio", st.median(ratio_rs), 2)
check(int(num("ph.rs.w95")) == sum(rec[t]["set_range"] > float(by[t]["rsm_w95"]) for t in DATES), "set vs resample w95")
printed("ph.rs.rel.median", st.median(rr[t] / rec[t]["best_rate"] for t in DATES) * 100, 0)
printed("ph.lfl.rel.median", st.median(lfl[t] / rec[t]["best_rate"] for t in DATES) * 100, 0)
check(int(num("ph.rs.flipmore")) == sum(float(by[t]["lfl_flip20"]) > float(by[t]["rsm_flip"]) for t in DATES),
      "staging: set20 vs resample")
printed("ph.rs.flip", st.median(float(by[t]["rsm_flip"]) for t in DATES) * 100, 1)
check(disj == list(range(2018, 2026)) and int(num("ph.rs.disj.n")) == len(disj)
      and int(num("ph.rs.disj.first")) == disj[0], f"disjoint resample dates {disj}")
check(int(num("ph.rs.disj.refitmore")) == sum(rr[t] > lfl[t] for t in disj), "disjoint dates: refits span more")
printed("ph.rs.ov.max", max(ov_mean.values()) * 100, 0)
check(int(num("ph.rs.ov.max_year")) == max(ov_mean, key=ov_mean.get), "max overlap year")
fm = st.median(rec[t]["fam_mono_rel"] for t in DATES) * 100
fl = st.median(rec[t]["fam_lgbm_rel"] for t in DATES) * 100
printed("res.fam.mono.median", fm, 0)
printed("res.fam.lgbm.median", fl, 0)
check(fm >= fl, "text: constraints do not narrow the range on average")
pre = [t for t in elig if t <= 2008]
fp = [rec[t]["dec_real_tlgd"] / rec[t]["dec_sl1_tlgd"] for t in pre]
fg = [rec[t]["dec_real_flgd"] / rec[t]["dec_real_tlgd"] for t in pre]
printed("dec.path.min", min(fp), 1)
printed("dec.path.max", max(fp), 0)
printed("dec.lgd.min", min(fg), 1)
printed("dec.lgd.max", max(fg), 1)
check(all(rec[t]["dec_real_flgd"] < float(rz[t]["realized_rate"]) for t in pre) and int(num("dec.pre.n")) == len(pre),
      "text: pre-crisis decomposition still below realized")
check(int(num("dec.pre.below_lb")) == sum(rec[t]["dec_real_flgd"] < float(rz[t]["realized_rate_lb"]) for t in pre),
      "pre-crisis below lower bound")
check(all(rec[t]["dec_real_tlgd"] < rec[t]["dec_sl1_tlgd"] and rec[t]["dec_real_flgd"] < rec[t]["dec_real_tlgd"]
          for t in elig if t >= 2011), "text: from 2011 each replacement lowers the allowance")
check(not all(rec[t]["dec_real_tlgd"] < rec[t]["dec_sl1_tlgd"] and rec[t]["dec_real_flgd"] < rec[t]["dec_real_tlgd"]
              for t in (2009, 2010)), "text: 'from December 2011' is the first such date")
printed("dec.late.maxdev", max(abs(rec[t]["dec_real_flgd"] / float(rz[t]["realized_rate"]) - 1)
                               for t in elig if t >= 2013) * 100, 0)
check(all(float(rz[t]["imputed_share_of_loss"]) > 0.5 for t in elig if t >= 2015), "text: imputed is most of loss from 2015")
check(float(rz[2014]["imputed_share_of_loss"]) <= 0.5005, "text: 'from 2015'")
for t in DATES:
    check(close(float(by[t]["policy_imm"]), float(by[t]["policy_sl1"]), 1e-12), f"{t}: immediate equals sl1 at annual resolution")
check(int(num("p2.boot_gt_rsm")) == sum(float(by[t]["refit_range"]) > rr[t] for t in DATES), "bootstrap vs independent spread")
printed("p2.boot.rel", st.median(float(by[t]["refit_range"]) / rec[t]["best_rate"] for t in DATES) * 100, 0)
printed("p2.disj.rsm", st.median(rr[t] / rec[t]["best_rate"] for t in disj) * 100, 0)
printed("p2.disj.lfl", st.median(lfl[t] / rec[t]["best_rate"] for t in disj) * 100, 0)
check(num("p2.disj.rsm") > num("p2.disj.lfl"), "text: refits exceed model choice at disjoint dates")


def ranks(x):
    o = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y):
    a, b = ranks(x), ranks(y)
    ma, mb = st.mean(a), st.mean(b)
    num_ = sum((u - ma) * (v - mb) for u, v in zip(a, b))
    return num_ / math.sqrt(sum((u - ma) ** 2 for u in a) * sum((v - mb) ** 2 for v in b))


p2 = {int(r["t"]): r for r in rows(RES / "posthoc2.csv")}
pos, rho_s, rho_r, in_ref, in_rsm, below = [], [], [], 0, 0, 0
lg19 = {(r["band"], r["judicial"]): float(r["lgd"]) for r in rows(RES / "dates" / "2019" / "lgd.csv")}
for t in DATES:
    d = RES / "dates" / str(t)
    acc = json.loads((d / "accuracy.json").read_text())
    models = {int(r["model"]): float(r["logloss"]) for r in rows(d / "models.csv") if r["target"] == "default"}
    al = rows(d / "allowance.csv")
    prim = {r["model"]: float(r["allowance_rate"]) for r in al
            if r["set"] == "default" and r["policy"] == "sl1" and r["origination"] == "false"}
    S = sorted(acc["default"]["sets"]["ll_1se"])
    v = [prim[f"m{m:03d}"] for m in S]
    b = rec[t]["best_rate"]
    pos.append((b - min(v)) / (max(v) - min(v)))
    below += b < st.median(v)
    rho_s.append(spearman([models[m] for m in S], v))
    rsm = sorted((r["model"], float(r["allowance_rate"])) for r in al
                 if r["set"] == "resample" and r["policy"] == "sl1" and r["origination"] == "false")
    rho_r.append(spearman(json.loads((d / "resample.json").read_text())["logloss"], [x for _, x in rsm]))
    check(close(rho_s[-1], float(p2[t]["rho_set"]), 1e-9) and close(rho_r[-1], float(p2[t]["rho_rsm"]), 1e-9), f"{t}: rank correlations")
    mass = [r for r in rows(d / "el_mass.csv") if r["policy"] == "realized"]
    rec[t]["dec19"] = sum(float(r["mass"]) * lg19[(r["band"], r["judicial"])] for r in mass) / float(mass[0]["upb_scored"])
    check(close(rec[t]["dec19"], float(p2[t]["dec_real_lgd19"]), 1e-9), f"{t}: 2019-severity decomposition")
    pdr = [float(r["mean_pd_life"]) for r in al if r["set"] == "default" and r["policy"] == "realized"
           and r["model"] == f"m{acc['default']['best']:03d}"]
    check(len(pdr) == 1 and close(pdr[0], float(p2[t]["pd_real_best"]), 1e-9), f"{t}: realized-path PD")
    check(close(float(rz[t]["default_share"]), float(p2[t]["default_share"]), 1e-12), f"{t}: default share")
    if t in elig:
        R = float(rz[t]["realized_rate"])
        ref = [float(r["allowance_rate"]) for r in al if r["set"] == "refit" and r["policy"] == "sl1" and r["origination"] == "false"]
        in_ref += min(ref) <= R <= max(ref)
        in_rsm += min(x for _, x in rsm) <= R <= max(x for _, x in rsm)
printed("p2.pos.median", st.median(pos) * 100, 0)
check(int(num("p2.below_median")) == below, "best below set median")
check(int(num("p2.rho_set.pos")) == sum(x > 0 for x in rho_s), "rho_set positive count")
printed("p2.rho_set.median", st.median(rho_s), 2)
printed("p2.rho_rsm.median", st.median(rho_r), 2)
check(int(num("p2.in_refit")) == in_ref and int(num("p2.in_rsm")) == in_rsm, "refit ranges vs realized")
check(int(num("p2.h4.n")) == sum(float(p2[t]["h4_mono_sub"]) < float(p2[t]["h4_lgbm_sub"]) for t in DATES), "H4 equal-size")
check(int(num("p2.h4.n")) > n / 2, "text: equal-size H4 verdict unchanged")
pre4 = [t for t in elig if t <= 2008]
printed("dec.lgd19.min", min(rec[t]["dec19"] / rec[t]["dec_real_tlgd"] for t in pre4), 1)
printed("dec.lgd19.max", max(rec[t]["dec19"] / rec[t]["dec_real_tlgd"] for t in pre4), 1)
short = {t: rec[t]["dec19"] / float(rz[t]["realized_rate"]) - 1 for t in pre4}
check([t for t in pre4 if short[t] > 0] == [2006], "text: only 2006 above realized with 2019 severity")
printed("dec.lgd19.short.max", -min(short.values()) * 100, 0)
printed("dec.lgd19.over2006", short[2006] * 100, 0)
pr = {t: float(p2[t]["pd_real_best"]) / float(rz[t]["default_share"]) for t in elig}
printed("pd.pre.min", min(pr[t] for t in pre4), 2)
printed("pd.pre.max", max(pr[t] for t in pre4), 2)
printed("pd.late.min", min(pr[t] for t in range(2013, 2018)), 2)
printed("pd.late.max", max(pr[t] for t in range(2013, 2018)), 2)
check(all(rec[t]["dec19"] > float(rz[t]["realized_rate"]) for t in range(2010, 2018)), "text: col 3 above realized 2010-2017")
check(all(rec[t]["dec19"] < rec[t]["dec_real_tlgd"] for t in range(2011, 2018)) and rec[2010]["dec19"] > rec[2010]["dec_real_tlgd"],
      "text: from 2011 the 2019 severities lower the allowance")
check(all(abs(rec[t]["dec_real_flgd"] - float(rz[t]["realized_rate"])) < abs(rec[t]["dec19"] - float(rz[t]["realized_rate"]))
          for t in range(2010, 2016)), "text: column 4 closer 2010-2015")
check(not all(abs(rec[t]["dec_real_flgd"] - float(rz[t]["realized_rate"])) < abs(rec[t]["dec19"] - float(rz[t]["realized_rate"]))
              for t in (2016, 2017)), "text: 'to 2015' boundary")
pol = {t: rec[t]["policy_range"] / rec[t]["best_rate"] * 100 for t in DATES}
printed("res.policy.rel.median1", st.median(pol.values()), 1)
printed("res.policy.rel.2007", pol[2007], 0)
printed("res.policy.rel.2021", pol[2021], 0)
check(int(num("p2.paired1.single")) == sum(int(by[t]["size_paired1"]) == 1 for t in DATES), "paired1 singletons")
check(int(num("data.kept")) == flow_kept if (flow_kept := json.loads((RES / "flow.json").read_text())["loans_kept"]) else False, "kept loans")
fr = {t: json.loads((RES / "dates" / str(t) / "accuracy.json").read_text())["info"]["sample_fraction"] for t in DATES}
printed("ph.frac.min", min(fr.values()) * 100, 1)
printed("ph.rs.mut.max", max(max(0, (f - 1 / 21) / f) for f in fr.values()) * 100, 0)
check([t for t in elig if float(rz[t]["realized_rate"]) < rec[t]["set_min"]] == list(range(2010, 2018)), "text: below from 2010 to 2017")
check(all((rec[t]["dec_real_flgd"] - rec[t]["dec_sl1_tlgd"]) / (float(rz[t]["realized_rate"]) - rec[t]["dec_sl1_tlgd"]) >= 0.5
          for t in pre), "text: much of the pre-crisis miss is forecast and severity")
# delinquency
dq: dict[int, dict[str, dict]] = {}
for r in rows(RES / "delinquency.csv"):
    dq.setdefault(int(r["y"]), {})[r["g"]] = r
share = {y: sum(float(v["upb"]) for g, v in d.items() if g != "current") / sum(float(v["upb"]) for v in d.values())
         for y, d in dq.items() if y in DATES}
printed("dq.share.min", min(share.values()) * 100, 1)
printed("dq.share.max", max(share.values()) * 100, 1)
check(int(num("dq.share.max_year")) == max(share, key=share.get), "delinquency max year")
for g, k, dg in (("current", "cur", 2), ("30-89", "d30", 1), ("90-179", "d90", 1)):
    r = dq[2009][g]
    printed(f"dq.2009.{k}", int(r["dflt"]) / int(r["known"]) * 100, dg)
for y in DATES:
    check(int(dq[y]["current"]["n"]) + int(dq[y]["30-89"]["n"]) + int(dq[y]["90-179"]["n"]) == int(by[y]["n_active"]),
          f"{y}: delinquency groups sum to the active book")

# ---------------------------------------------------------------- data section
flow = json.loads((RES / "flow.json").read_text())
ing = json.loads((RES / "ingest_summary.json").read_text())
stats = json.loads((RES / "data_stats.json").read_text())
check(ing["quarters"] == 110 and all(v == 0 for k, v in ing["checks"].items()
                                     if k not in ("loans_without_perf", "orig_id_quarter_mismatch")), "ingest checks")
check(ing["orig_rows"] - ing["checks"]["loans_without_perf"] == flow["loans_ingested"], "flow: ingested")
check(flow["loans_ingested"] - flow["excluded_no_state_hpi"] - flow["excluded_no_ltv"] - flow["excluded_no_rate"]
      == flow["loans_kept"], "flow: kept arithmetic")
printed("data.orig_rows", ing["orig_rows"], 0)
printed("data.perf_rows", ing["perf_rows"] / 1e9, 2)
printed("data.ce_2009", stats["2009"]["credit_event"] * 100, 0)
printed("data.ce_2020", stats["2020"]["credit_event"] * 100, 1)
printed("data.po_2020", stats["2020"]["paid_off"] * 100, 0)
printed("data.act_2020", stats["2020"]["still_active"] * 100, 0)
for yr in ("2009", "2020"):
    check(abs(sum(v for k, v in stats[yr].items() if k != "defaults") - 1) < 1e-9, f"{yr} default endings sum to 1")
ly = flow["loan_years"]
printed("data.active_2025", sum(ly["2025"].values()) / 1e6, 1)
dr = {int(y) + 1: v["1"] / (v["0"] + v["1"] + v["2"]) for y, v in ly.items() if "1" in v}
printed("data.dr_2020", dr[2020] * 100, 2)
check(int(num("data.dr_peak_gfc_year")) == max((y for y in dr if y <= 2015), key=dr.get), "GFC peak year")
check(1.0 <= sum(ly["1999"].values()) / 1e6 < 1.01, "'about one million loans at the end of 1999'")
n_cmp = hi = 0
for t in DATES:
    lg = {(r["band"], r["judicial"]): r for r in rows(RES / "dates" / str(t) / "lgd.csv")}
    for b in map(str, range(6)):
        p, j = lg[(b, "false")], lg[(b, "true")]
        if p["source"] == "cell" and j["source"] == "cell":
            n_cmp += 1
            hi += float(j["lgd"]) > float(p["lgd"])
check(int(num("data.jud.n")) == n_cmp and int(num("data.jud.higher")) == hi, "judicial severity comparison")

# ---------------------------------------------------------------- text and tables
sec = {p.name: p.read_text() for p in (PAPER / "sections").glob("*.tex")}
res = sec["05_results.tex"]
for h in ("H1", "H2", "H3", "H4", "H5", "H6"):
    check(f"\\num{{{h}.verdict}}" in res, f"{h} verdict word not taken from numbers.tex")
hyp = (PAPER / "tables" / "hypotheses.tex").read_text()
check(hyp.count("Holds") == 5 and hyp.count("Fails") == 1, "hypothesis table verdicts")
bt = (PAPER / "tables" / "bydate.tex").read_text()
for t in DATES:
    m = re.search(rf"^{t} & (\d+) & ([\d.]+) & ([\d.]+) & ([\d.]+)--([\d.]+) & (\d+)", bt, re.M)
    check(m is not None, f"bydate row {t}")
    if m:
        check(int(m.group(1)) == rec[t]["set_size"], f"bydate {t} size")
        check(abs(float(m.group(3)) - rec[t]["best_rate"] * 100) <= 0.0051, f"bydate {t} best rate")
        check(abs(float(m.group(6)) - rec[t]["rel"] * 100) <= 0.51, f"bydate {t} rel")
undef = set(re.findall(r"\\num\{([^}]+)\}", "".join(sec.values()))) - set(NUM)
check(not undef, f"undefined numbers {undef}")
check("\\@namedef{num@res.rel.median}" in (PAPER / "numbers.tex").read_text(), "numbers.tex generated")

readme = (ROOT / "README.md").read_text()
check(readme.splitlines()[147] == "Each prediction is reported whether it holds or fails.", "README design prefix moved")
for txt in (f"median {NUM['res.set.size.median']} of 65 ({NUM['res.set.size.min']}–{NUM['res.set.size.max']})",
            f"median {NUM['res.rel.median']}% of the best model's allowance ({NUM['res.rel.min']}%–{NUM['res.rel.max']}%)",
            f"up to ${NUM['res.usd.max']}bn (Dec. {NUM['res.usd.max_year']})",
            f"median {NUM['ph.paired2.size']} models, median relative range {NUM['ph.paired2.rel']}%",
            f"holds, {NUM['H1.n']} of 21 dates", f"holds, {NUM['H2.n']} of 21 dates (median ratio {NUM['res.ratio_policy.median']})",
            f"({NUM['H3.rel2006']}% and {NUM['H3.rel2007']}% vs {NUM['H3.median']}%)",
            f"holds narrowly, {NUM['H4.n']} of 21 dates; medians {NUM['res.fam.mono.median']}% vs {NUM['res.fam.lgbm.median']}%",
            f"holds, {NUM['H5.inside']} of {NUM['H5.n_elig']} dates",
            f"holds, {NUM['H6.n']} of 21 dates ({NUM['res.flip.median']}% vs {NUM['res.flip_refit.median']}%)",
            f"span {NUM['ph.rs.rel.median']}% vs {NUM['ph.lfl.rel.median']}% for 20 set models ({NUM['p2.disj.rsm']}% vs {NUM['p2.disj.lfl']}% at the {NUM['ph.rs.disj.n']} disjoint dates); flip {NUM['ph.rs.flip']}% vs {NUM['ph.H6.flipset']}%"):
    check(txt in readme, f"README results summary: '{txt}' not found")
abstract = (PAPER / "submission-ssrn" / "abstract.txt").read_text()
check(NUM["res.rel.median"] + "%" in abstract and "{" not in abstract, "plain abstract")

# ---------------------------------------------------------------- hygiene and firewall
for p in RES.rglob("*"):
    if p.is_file() and p.suffix in (".csv", ".json"):
        head = p.read_text()[:2000]
        check("loan_id" not in head and not re.search(r"F\d{2}Q[1-4]\d{7}", head), f"loan identifier in {p}")
    if p.is_file():
        check(p.stat().st_size < 5_000_000, f"large file in results: {p}")
terms_file = PAPER / ".firewall_terms"
if terms_file.exists():
    terms = [t.strip().lower() for t in terms_file.read_text().splitlines() if t.strip() and not t.startswith("#")]
    corpus = " ".join(sec.values()).lower() + (ROOT / "README.md").read_text().lower()
    for code in (ROOT / "code").rglob("*.py"):
        # matplotlib's transparency keyword is not a firewall term
        corpus += re.sub(r"\balpha=", "", code.read_text().lower())
    hits = [t for t in terms if t in corpus]
    check(not hits, f"firewall terms found: {hits}")
else:
    FAIL.append("paper/.firewall_terms missing; firewall scan not run")

print(f"{N_OK} checks passed, {len(FAIL)} failed")
for f in FAIL:
    print("FAIL:", f)
sys.exit(1 if FAIL else 0)
