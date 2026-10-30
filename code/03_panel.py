"""Build the loan table and the loan-year panel from the ingested quarters.

Usage: python code/03_panel.py

A loan-year is a loan active at December of year y (balance above zero, no
zero-balance event, never yet 180+ days delinquent). Its features are
measured at that December; its outcome is what happened in year y + 1:
default (first month 180+ days delinquent, or a credit-event zero balance,
whichever is first), prepayment (zero-balance code 01), survival, or
censoring (a defect or reperforming-sale removal, or records that stop).
Rows for December 2025 have no outcome yet; they are only scored as the
last portfolio.

Outputs (DERIVED/):
  loans.parquet          one row per loan: inputs fixed at origination, the
                         default month and balance, the terminal event
  panel/y=YYYY/*.parquet loan-years by snapshot year
  results/flow.json      sample flow and outcome counts (aggregate only)
"""
from __future__ import annotations

import json
import os

import duckdb

from ecl import CATEGORICAL, DEFAULT_MONTHS, DERIVED, JUDICIAL, LAST_OUTCOME_YEAR, MACRO, RESULTS, ZB_CREDIT

THREADS = int(os.environ.get("ECL_THREADS", "18"))
MEM = os.environ.get("ECL_MEM", "90GB")


def main() -> None:
    con = duckdb.connect()
    con.execute(f"SET threads={THREADS}; SET memory_limit='{MEM}'; SET preserve_insertion_order=false; "
                f"SET temp_directory='{DERIVED / 'duck_tmp'}'")
    credit = ", ".join(f"'{c}'" for c in ZB_CREDIT)
    judicial = ", ".join(f"'{s}'" for s in sorted(JUDICIAL))

    def code(col: str) -> str:
        cats = CATEGORICAL[col]
        whens = " ".join(f"WHEN '{v}' THEN {i}" for i, v in enumerate(cats))
        return f"CAST(CASE {col} {whens} END AS SMALLINT) AS {col}"

    # Origination month: the first payment is due on the first of the second
    # month after closing (interest is paid in arrears), so the note date
    # falls about two months before the first payment date.
    con.execute(f"""
    CREATE TABLE loans AS
    WITH o AS (
      SELECT *, (first_pay // 100) * 12 + (first_pay % 100) - 1 - 2 AS om
      FROM read_parquet('{DERIVED / "orig" / "*.parquet"}')),
    e AS (SELECT * FROM read_parquet('{DERIVED / "events" / "*.parquet"}')),
    j AS (
      SELECT o.*, e.* EXCLUDE (loan_id),
        (om // 12) * 100 + om % 12 + 1 AS orig_month,
        (om // 12) * 4 + (om % 12) // 3 AS orig_qidx,
        CASE WHEN zb_code IN ({credit}) THEN zb_date END AS credit_date
      FROM o JOIN e USING (loan_id))
    SELECT j.loan_id, substr(j.loan_id, 2, 4) AS vintage_q, j.state,
      j.state IN ({judicial}) AS judicial,
      NULLIF(j.fico, 9999) AS fico, NULLIF(j.ltv, 999) AS ltv, NULLIF(j.cltv, 999) AS cltv,
      NULLIF(j.dti, 999) AS dti, j.orig_rate, j.orig_rate - pm.pmms AS spread,
      NULLIF(j.units, 99) AS units, NULLIF(j.n_borrowers, 99) AS n_borrowers,
      NULLIF(j.mi_pct, 999) AS mi_pct, j.orig_term, j.orig_upb, j.first_pay, j.harp,
      {", ".join(code(c) for c in CATEGORICAL)},
      j.orig_upb / (NULLIF(j.ltv, 999) / 100.0) AS value, h.hpi AS hpi_orig,
      j.first_period, j.last_period, j.zb_code, j.zb_date, j.zb_upb, j.actual_loss,
      j.defect_date, j.cum_mod_cost, j.ever_mod, j.ever_deferral, j.first_180,
      CASE WHEN j.first_180 IS NULL THEN j.credit_date
           WHEN j.credit_date IS NULL THEN j.first_180
           ELSE least(j.first_180, j.credit_date) END AS default_period,
      CASE WHEN j.first_180 IS NOT NULL AND (j.credit_date IS NULL OR j.first_180 <= j.credit_date)
           THEN coalesce(NULLIF(j.upb_180, 0), j.zb_upb)
           ELSE j.zb_upb END AS default_upb
    FROM j
    LEFT JOIN read_parquet('{MACRO / "hpi_quarter.parquet"}') h ON h.state = j.state AND h.qidx = j.orig_qidx
    LEFT JOIN read_parquet('{MACRO / "pmms_month.parquet"}') pm ON pm.month = j.orig_month
    """)
    flow = {"loans_ingested": con.execute("SELECT count(*) FROM loans").fetchone()[0]}
    flow["excluded_no_state_hpi"] = con.execute("SELECT count(*) FROM loans WHERE hpi_orig IS NULL").fetchone()[0]
    flow["excluded_no_ltv"] = con.execute(
        "SELECT count(*) FROM loans WHERE hpi_orig IS NOT NULL AND value IS NULL").fetchone()[0]
    flow["states_without_hpi"] = [r[0] for r in con.execute(
        "SELECT DISTINCT state FROM loans WHERE hpi_orig IS NULL ORDER BY 1").fetchall()]
    flow["no_pmms"] = con.execute("SELECT count(*) FROM loans WHERE spread IS NULL").fetchone()[0]
    flow["excluded_no_rate"] = con.execute(
        "SELECT count(*) FROM loans WHERE hpi_orig IS NOT NULL AND value IS NOT NULL AND orig_rate IS NULL").fetchone()[0]
    con.execute("DELETE FROM loans WHERE hpi_orig IS NULL OR value IS NULL OR orig_rate IS NULL")
    flow["loans_kept"] = con.execute("SELECT count(*) FROM loans").fetchone()[0]
    flow["default_upb_missing"] = con.execute(
        "SELECT count(*) FROM loans WHERE default_period IS NOT NULL AND (default_upb IS NULL OR default_upb <= 0)").fetchone()[0]
    con.execute(f"COPY loans TO '{DERIVED / 'loans.parquet'}' (FORMAT parquet)")

    # Loan-years. Loan age follows the dataset's own definition (reporting
    # month minus first payment month plus one) but is computed from dates,
    # so a modification does not reset it.
    last = LAST_OUTCOME_YEAR
    sy = f"read_parquet('{MACRO / "state_year.parquet"}')"
    snaps = f"read_parquet('{DERIVED / "snap" / "*.parquet"}')"
    years = [r[0] for r in con.execute(f"SELECT DISTINCT period // 100 FROM {snaps} ORDER BY 1").fetchall()]
    # One snapshot year at a time: a single partitioned write of ~250M rows
    # exceeds memory.
    for yr in years:
        (DERIVED / "panel" / f"y={yr}").mkdir(parents=True, exist_ok=True)
        con.execute(f"""
    COPY (
    WITH s AS (SELECT * FROM {snaps} WHERE period = {yr * 100 + 12})
    SELECT l.loan_id, l.vintage_q, l.state, l.judicial,
      l.fico, l.ltv, l.cltv, l.dti, l.orig_rate, l.spread, l.units, l.n_borrowers, l.mi_pct,
      l.orig_term, {", ".join(f"l.{c}" for c in CATEGORICAL)},
      CAST({yr} * 12 + 12 - ((l.first_pay // 100) * 12 + l.first_pay % 100) + 1 AS SMALLINT) AS loan_age,
      100.0 * s.upb / (l.value * sy.hpi_q4 / l.hpi_orig) AS cur_ltv,
      sy.hpi_g, sy.ur, sy.ur_chg,
      s.upb, s.nib_upb, s.cur_rate, s.rem_months, s.mod, s.dlq_m,
      l.value, l.hpi_orig, sy.hpi_q4,
      CASE
        WHEN {yr} >= {last} THEN NULL
        WHEN l.default_period BETWEEN {(yr + 1) * 100 + 1} AND {(yr + 1) * 100 + 12} THEN 1
        WHEN l.zb_code = '01' AND l.zb_date BETWEEN {(yr + 1) * 100 + 1} AND {(yr + 1) * 100 + 12} THEN 2
        WHEN l.zb_date BETWEEN {(yr + 1) * 100 + 1} AND {(yr + 1) * 100 + 12} THEN -1
        WHEN l.zb_code IS NULL AND l.last_period < {(yr + 1) * 100 + 12} THEN -2
        ELSE 0 END AS outcome
    FROM s JOIN loans l USING (loan_id)
    JOIN {sy} sy ON sy.state = l.state AND sy.year = {yr}
    WHERE s.upb > 0 AND NOT s.zb
      AND (l.default_period IS NULL OR l.default_period > s.period)
      AND (l.zb_date IS NULL OR l.zb_date > s.period)
    ) TO '{DERIVED / "panel" / f"y={yr}" / "data.parquet"}' (FORMAT parquet)
    """)
        print("panel", yr, flush=True)
    panel = f"read_parquet('{DERIVED / 'panel' / '*' / '*.parquet'}', hive_partitioning=true)"
    rows = con.execute(f"""SELECT y, outcome, count(*) FROM {panel} GROUP BY ALL ORDER BY 1, 2""").fetchall()
    flow["loan_years"] = {}
    for y, oc, n in rows:
        flow["loan_years"].setdefault(str(y), {})[str(oc)] = n
    flow["panel_dlq_ge_default"] = con.execute(
        f"SELECT count(*) FROM {panel} WHERE dlq_m >= {DEFAULT_MONTHS}").fetchone()[0]
    flow["panel_dup"] = con.execute(
        f"SELECT count(*) - count(DISTINCT (loan_id, y)) FROM {panel}").fetchone()[0]
    flow["panel_loans"] = con.execute(f"SELECT count(DISTINCT loan_id) FROM {panel}").fetchone()[0]
    assert flow["panel_dlq_ge_default"] == 0 and flow["panel_dup"] == 0, flow
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "flow.json").write_text(json.dumps(flow, indent=1))
    print(json.dumps({k: v for k, v in flow.items() if k != "loan_years"}, indent=1))


if __name__ == "__main__":
    main()
