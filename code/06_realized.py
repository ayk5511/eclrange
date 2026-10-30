"""Realized credit losses on each reporting date's scored portfolio.

Usage: python code/06_realized.py

For every loan in the scored sample at December T that later defaults
(default month after December T, observed through June 2026):
  - resolved by a voluntary payoff after the default: loss 0;
  - resolved by a credit event (02, 03, 09, 15) with no defect settlement
    and a disclosed actual loss: that loss;
  - otherwise (still active at the end of the data, removed by code 16 or
    96, a defect settlement, or a loss not yet disclosed): balance at
    default x the balance-weighted severity of all resolved defaults in the same
    current-LTV band x judicial cell (over the whole data), and flagged as
    imputed.
The realized loss rate is sum(loss) / sum(UPB at T) on the same sample the
allowance is computed on. A second figure counts only defaults through
December 2025, matching allowance_to2025.csv.

Output: results/realized.csv (one row per reporting date).
"""
from __future__ import annotations

import json

import polars as pl

from ecl import DERIVED, FIRST_YEAR, LAST_YEAR, RESULTS, ZB_CREDIT
from ecl.data import scan_panel

BANDS = [60, 80, 90, 100, 120]


def main() -> None:
    loans = pl.read_parquet(DERIVED / "loans.parquet", columns=[
        "loan_id", "judicial", "zb_code", "zb_date", "actual_loss", "defect_date", "default_upb", "default_period"])
    loans = loans.filter(pl.col("default_period").is_not_null())
    # current LTV at the start of the default year: the December row before it
    pre = (scan_panel().select("loan_id", "y", "cur_ltv")
           .join(loans.lazy().select("loan_id", "default_period"), on="loan_id")
           .filter(pl.col("y") == pl.col("default_period") // 100 - 1)
           .select("loan_id", "cur_ltv").collect())
    d = (loans.join(pre, on="loan_id", how="left")
         .with_columns(band=pl.col("cur_ltv").cut(BANDS, left_closed=True, labels=[str(i) for i in range(6)])
                       .cast(pl.Utf8).cast(pl.Int8),
                       loss=pl.when(pl.col("zb_code") == "01").then(0.0)
                       .when(pl.col("zb_code").is_in(list(ZB_CREDIT)) & pl.col("defect_date").is_null())
                       .then(pl.col("actual_loss")).otherwise(None)))
    resolved = d.filter(pl.col("loss").is_not_null() & (pl.col("default_upb") > 0))
    # balance-weighted severity, as in 05_allowance.py
    cell = resolved.group_by("band", "judicial").agg(sev=pl.col("loss").sum() / pl.col("default_upb").sum(), n=pl.len())
    overall = float(resolved["loss"].sum() / resolved["default_upb"].sum())
    d = (d.join(cell, on=["band", "judicial"], how="left")
         .with_columns(imputed=pl.col("loss").is_null(),
                       loss_r=pl.coalesce(pl.col("loss"), pl.col("default_upb") * pl.coalesce(pl.col("sev"), pl.lit(overall)))))
    rows = []
    for t in range(FIRST_YEAR, LAST_YEAR + 1):
        f = DERIVED / "alloc" / str(t) / "port.parquet"
        if not f.exists():
            continue
        port = pl.read_parquet(f, columns=["loan_id", "upb"])
        j = port.join(d.filter(pl.col("default_period") > t * 100 + 12), on="loan_id", how="left")
        upb = float(port["upb"].sum())
        dl = j.filter(pl.col("default_period").is_not_null())
        to25 = dl.filter(pl.col("default_period") <= 202512)
        rows.append({"t": t, "n_scored": len(port), "upb_scored": upb,
                     "n_default_after": len(dl), "default_share": len(dl) / len(port),
                     "realized_rate": float(dl["loss_r"].sum()) / upb,
                     "realized_rate_to2025": float(to25["loss_r"].sum()) / upb,
                     # lower bound: unresolved defaults (mostly still-active,
                     # largely reperforming loans) contribute no loss
                     "realized_rate_lb": float(dl.filter(~pl.col("imputed"))["loss_r"].sum()) / upb,
                     "imputed_share_of_loss": float(dl.filter(pl.col("imputed"))["loss_r"].sum() / max(dl["loss_r"].sum(), 1e-9)),
                     # outcomes run from the end of December T to June 2026
                     "years_observed": 2025.5 - t})
    pl.DataFrame(rows).write_csv(RESULTS / "realized.csv")

    # How the defaults of a crisis year and of the pandemic year ended
    # (terminal zero-balance code by June 2026), for the data section.
    stats = {}
    for yr in (2009, 2020):
        dd = d.filter(pl.col("default_period").is_between(yr * 100 + 1, yr * 100 + 12))
        n = len(dd)
        zb = dd["zb_code"].fill_null("active")  # a null comparison would drop out of a mean
        stats[str(yr)] = {"defaults": n,
                          "credit_event": int(zb.is_in(list(ZB_CREDIT)).sum()) / n,
                          "paid_off": int((zb == "01").sum()) / n,
                          "still_active": int((zb == "active").sum()) / n,
                          "other_removal": int(zb.is_in(["16", "96"]).sum()) / n}
        assert abs(sum(v for k, v in stats[str(yr)].items() if k != "defaults") - 1) < 1e-12
    stats["severity_all_resolved"] = {"n": len(resolved), "mean": overall}

    # Current delinquency is not a model input (design). How much of each
    # portfolio is 30-150 days past due, and how those loans default.
    dq = (scan_panel().filter(pl.col("y") >= FIRST_YEAR)
          .with_columns(g=pl.when(pl.col("dlq_m").fill_null(0) == 0).then(pl.lit("current"))
                        .when(pl.col("dlq_m") <= 2).then(pl.lit("30-89"))
                        .otherwise(pl.lit("90-179")))
          .group_by("y", "g").agg(n=pl.len(), upb=pl.col("upb").sum(),
                                  dflt=(pl.col("outcome") == 1).sum(), known=(pl.col("outcome") >= 0).sum())
          .sort("y", "g").collect())
    dq.write_csv(RESULTS / "delinquency.csv")
    (RESULTS / "data_stats.json").write_text(json.dumps(stats, indent=1))
    print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
