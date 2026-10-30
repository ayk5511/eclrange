"""Every figure in the paper, from results/ only (no loan-level file).

Usage: python code/08_figures.py
Writes paper/figures/*.pdf.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ecl import REPO, RESULTS
from ecl.style import (ALT, FAMILY, FAMILY_LABEL, INK, INK2, INK3, MODEL, MODEL_LIGHT, NOISE,
                       NOISE_LIGHT, POLICY, save, setup)

FIG = REPO / "paper" / "figures"
FULL_W, HALF_W = 6.5, 3.15
STRESS = (2006, 2007)


def pct(ax, axis="y", digits=1):
    fmt = plt.FuncFormatter(lambda v, _: f"{v:.{digits}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def years_axis(ax, years, step=2):
    ax.set_xticks([y for y in years if (y - 2001) % step == 0])
    ax.set_xlim(years[0] - 0.6, years[-1] + 0.6)


def fig_data(flow):
    ly = flow["loan_years"]
    ys = sorted(int(y) for y in ly)
    active = np.array([sum(ly[str(y)].values()) for y in ys]) / 1e6
    oy = [y for y in ys if "None" not in ly[str(y)]]
    tot = np.array([sum(v for k, v in ly[str(y)].items() if k in ("0", "1", "2")) for y in oy])
    dr = np.array([ly[str(y)].get("1", 0) for y in oy]) / tot * 100
    pr = np.array([ly[str(y)].get("2", 0) for y in oy]) / tot * 100
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.0))
    ax = axes[0]
    ax.bar(ys, active, width=0.72, color=MODEL, linewidth=0)
    ax.set_title("Active loans at year-end (millions)")
    years_axis(ax, ys, step=6)
    ax = axes[1]
    x = np.array(oy) + 1
    ax.plot(x, dr, color=MODEL, marker="o", ms=3)
    ax.set_title("Annual default rate")
    pct(ax)
    ax.annotate("2020:\npandemic\nforbearance", xy=(2020.2, dr[oy.index(2019)]), xytext=(2021.3, dr.max() * 0.62),
                fontsize=6.5, color=INK2, arrowprops={"arrowstyle": "-", "color": INK3, "lw": 0.6})
    years_axis(ax, list(x), step=6)
    ax = axes[2]
    ax.plot(x, pr, color=POLICY, marker="o", ms=3)
    ax.set_title("Annual prepayment rate")
    pct(ax, digits=0)
    years_axis(ax, list(x), step=6)
    fig.tight_layout(w_pad=1.2)
    save(fig, FIG / "data.pdf")


def fig_allowance(bd, rz):
    t = bd["t"].to_numpy()
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(FULL_W, 5.0), height_ratios=[1.35, 1], sharex=True)
    ax.fill_between(t, bd["set_min"] * 100, bd["set_max"] * 100, color=MODEL_LIGHT, lw=0, step=None,
                    label="Range across the equally accurate set")
    ax.plot(t, bd["best_rate"] * 100, color=MODEL, lw=1.6, label="Most accurate model", zorder=3)
    ax.fill_between(t, bd["refit_p025"] * 100, bd["refit_p975"] * 100, facecolor="none", edgecolor=NOISE,
                    hatch="////", lw=0, label="95% band, 20 bootstrap refits of that model")
    el = rz.filter(pl.col("years_observed") >= 8)
    ax.vlines(el["t"], el["realized_rate_lb"] * 100, el["realized_rate"] * 100, color=INK, lw=1.0, zorder=4)
    ax.scatter(el["t"], el["realized_rate"] * 100, s=26, color=INK, marker="D", zorder=5,
               edgecolor="white", linewidth=1.0,
               label="Realized loss on the same loans (line: lower bound,\nunresolved defaults at zero)")
    ax.set_title("A. Lifetime allowance as a share of balance, by reporting date")
    pct(ax, digits=2)
    ax.legend(loc="upper right", ncol=1, handlelength=1.6)
    years_axis(ax, list(t))

    # B: the three ranges on one log scale (all in basis points of balance)
    w = 0.22
    bx.scatter(t - 1.5 * w, bd["set_range"] * 1e4, color=MODEL, s=18, label="Model choice (range across the set)", zorder=3)
    bx.scatter(t - 0.5 * w, bd["refit_w95"] * 1e4, color=NOISE, s=18, marker="s",
               label="Bootstrap refits (95% width)", zorder=3)
    bx.scatter(t + 0.5 * w, bd["rsm_w95"] * 1e4, color=ALT, s=18, marker="D",
               label="Independent-sample refits (95% width, post hoc)", zorder=3)
    bx.scatter(t + 1.5 * w, bd["policy_range"] * 1e4, color=POLICY, s=18, marker="^", label="Reversion policy (range)", zorder=3)
    bx.set_yscale("log")
    bx.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    bx.set_ylabel("basis points of balance")
    bx.set_title("B. Size of each source of variation")
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, handletextpad=0.3, columnspacing=1.2)
    years_axis(bx, list(t))
    fig.tight_layout(h_pad=1.0)
    save(fig, FIG / "allowance.pdf")


def fig_cloud(ml, bd, dates):
    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 4.2))
    for ax, t in zip(axes.flat, dates):
        d = ml.filter(pl.col("t") == t)
        ax.axvspan(-0.3, 1.0, color=MODEL_LIGHT, alpha=0.45, lw=0)
        ax.axhline(0, color=INK3, lw=0.6)
        for fam in ("lgbm", "lgbm_mono", "logit_spline"):
            f = d.filter(pl.col("family") == fam)
            ax.scatter(f["d_logloss_se"], f["rel_to_best"] * 100, s=12 if fam != "logit_spline" else 30,
                       color=FAMILY[fam], edgecolor="white", linewidth=0.5, zorder=3,
                       marker="o" if fam != "logit_spline" else "D")
        row = bd.filter(pl.col("t") == t)
        ax.set_title(f"December {t}  ·  set of {row['set_size'].item()}", fontsize=8)
        ax.set_xlim(-0.3, float(d["d_logloss_se"].max()) * 1.08 + 0.3)
        pct(ax, digits=0)
    fig.supxlabel("held-out log loss above the most accurate model, in bootstrap standard errors", fontsize=8,
                  color=INK2, y=0.085)
    fig.supylabel("allowance relative to the most accurate model", fontsize=8, color=INK2, x=0.005)
    handles = [Line2D([], [], ls="", marker="o", color=FAMILY[f], ms=4, label=FAMILY_LABEL[f]) for f in ("lgbm", "lgbm_mono")]
    handles.append(Line2D([], [], ls="", marker="D", color=ALT, ms=4, label=FAMILY_LABEL["logit_spline"]))
    handles.append(Patch(color=MODEL_LIGHT, alpha=0.6, label="within one SE (the set)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0.015, 0.075, 1, 1))
    save(fig, FIG / "cloud.pdf")


def fig_relative(bd):
    t = bd["t"].to_numpy()
    rel = bd["rel_range"].to_numpy() * 100
    others = np.median(rel[~np.isin(t, STRESS)])
    fig, ax = plt.subplots(figsize=(FULL_W, 2.3))
    ax.bar(t, rel, width=0.7, color=[MODEL if y in STRESS else NOISE_LIGHT for y in t], lw=0)
    ax.axhline(others, color=INK2, lw=0.8)
    ax.text(t[-1] + 0.4, others, f"median of other dates {others:.0f}%", ha="right", va="bottom", fontsize=7, color=INK2)
    for y in STRESS:
        v = rel[list(t).index(y)]
        ax.text(y, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_title("Range across the set as a share of the most accurate model's allowance")
    pct(ax, digits=0)
    years_axis(ax, list(t))
    save(fig, FIG / "relative.pdf")


def fig_families(bd):
    t = bd["t"].to_numpy()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(FULL_W, 2.3))
    ax.plot(t, bd["fam_lgbm_rel"] * 100, color=MODEL, marker="o", ms=3, label="LightGBM")
    ax.plot(t, bd["fam_mono_rel"] * 100, color=POLICY, marker="^", ms=3, label="LightGBM, monotone")
    ax.set_title("A. Relative range within each family's own set")
    pct(ax, digits=0)
    ax.legend(loc="upper left")
    bx.plot(t, bd["flip_set"] * 100, color=MODEL, marker="o", ms=3, label="across the set")
    bx.plot(t, bd["flip_refit"] * 100, color=NOISE, marker="s", ms=3, label="across 20 refits")
    bx.set_title("B. Loans whose IFRS 9 stage flips")
    pct(bx, digits=0)
    bx.legend(loc="upper right")
    for a in (ax, bx):
        years_axis(a, list(t), step=4)
    fig.tight_layout(w_pad=1.5)
    save(fig, FIG / "families_stages.pdf")


def fig_sensitivity(bd):
    t = bd["t"].to_numpy()
    fig, ax = plt.subplots(figsize=(FULL_W, 2.4))
    b = bd["best_rate"].to_numpy()
    for col, lab, c, mk in (("range_0.5se", "0.5 SE", MODEL_LIGHT, "o"), ("set_range", "1 SE (primary)", MODEL, "o"),
                            ("range_2se", "2 SE", "#1c5cab", "o"), ("range_auc", "AUC within 1 SE", POLICY, "^"),
                            ("range_paired2", "paired, 2 SE (post hoc)", NOISE, "s")):
        ax.plot(t, bd[col].to_numpy() / b * 100, color=c, marker=mk, ms=3, lw=1.3, label=lab)
    ax.set_title("Relative range under other definitions of \u201cequally accurate\u201d")
    pct(ax, digits=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5)
    years_axis(ax, list(t))
    save(fig, FIG / "sensitivity.pdf")


def fig_dollars(bd):
    t = bd["t"].to_numpy()
    fig, ax = plt.subplots(figsize=(FULL_W, 2.2))
    ax.bar(t, bd["usd_range"] / 1e9, width=0.7, color=MODEL, lw=0)
    i = int(np.argmax(bd["usd_range"]))
    ax.text(t[i], bd["usd_range"][i] / 1e9, f"${bd['usd_range'][i] / 1e9:.1f}bn", ha="center", va="bottom", fontsize=7)
    ax.set_title("Range across the set, scaled to Freddie Mac's whole active standard book ($ billion)")
    years_axis(ax, list(t))
    save(fig, FIG / "dollars.pdf")


def band_label(b: str) -> str:
    """'[60, 80)' -> '60\u201380', '[-inf, 60)' -> '<60', '[120, inf)' -> '\u2265120'."""
    lo, hi = (v.strip() for v in b.strip("[()]").split(","))
    if lo == "-inf":
        return f"<{hi}"
    if hi == "inf":
        return f"\u2265{lo}"
    return f"{lo}\u2013{hi}"


def fig_loans(ld):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4), sharey=True)
    for ax, key, title in ((axes[0], "ltv_band", "A. By current LTV (%)"), (axes[1], "fico_band", "B. By credit score")):
        d = ld.filter((pl.col("by") == key) & pl.col("band").is_not_null() & (pl.col("band") != "null"))
        bands = d.select("band").unique(maintain_order=True)["band"].to_list()
        order = sorted(bands, key=lambda s: float(s.strip("[(").split(",")[0].replace("-inf", "-1e9")))
        x = np.arange(len(order))
        med = [d.filter(pl.col("band") == b)["ratio_med"].median() for b in order]
        for xi, b in enumerate(order):
            v = d.filter(pl.col("band") == b)["ratio_med"].to_numpy()
            ax.scatter(np.full(len(v), xi) + np.random.default_rng(xi).uniform(-0.12, 0.12, len(v)), v,
                       s=6, color=NOISE, alpha=0.7, lw=0, zorder=2)
        ax.scatter(x, med, s=40, color=MODEL, marker="_", linewidth=2.2, zorder=3)
        ax.set_xticks(x, [band_label(b) for b in order], fontsize=7)
        ax.set_title(title)
    axes[0].set_ylabel("median of max/min lifetime PD\nacross the set (per loan)")
    fig.tight_layout(w_pad=1.0)
    save(fig, FIG / "loans.pdf")


def main() -> None:
    setup()
    FIG.mkdir(parents=True, exist_ok=True)
    flow = json.loads((RESULTS / "flow.json").read_text())
    fig_data(flow)
    if not (RESULTS / "by_date.csv").exists():
        return
    bd = pl.read_csv(RESULTS / "by_date.csv").sort("t")
    rz = pl.read_csv(RESULTS / "realized.csv").sort("t")
    ml = pl.read_csv(RESULTS / "models_long.csv")
    ld = pl.read_csv(RESULTS / "loan_dispersion.csv")
    fig_allowance(bd, rz)
    dates = [d for d in (2006, 2008, 2012, 2017, 2020, 2025) if d in bd["t"].to_list()]
    fig_cloud(ml, bd, dates)
    fig_relative(bd)
    fig_families(bd)
    fig_sensitivity(bd)
    fig_dollars(bd)
    fig_loans(ld)


if __name__ == "__main__":
    main()
