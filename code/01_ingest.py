"""Reduce one origination quarter of the Freddie Mac Standard Dataset to three
loan-level parquet files.

Usage: python code/01_ingest.py 2003Q3

Outputs (DERIVED/, outside the repository):
  orig/{q}.parquet    one row per loan: the origination record, typed
  events/{q}.parquet  one row per loan: first and last period, first month
                      180+ days delinquent and the balance then, zero-balance
                      code, date, removal balance, actual loss, defect date
  snap/{q}.parquet    one row per loan and December: balance, delinquency,
                      remaining term, current rate, modification flag
  log/{q}.json        row counts and the checks this script asserts
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

import duckdb
import polars as pl

from ecl import DEFAULT_MONTHS, DERIVED, ORIG_COLS, PERF_COLS, ZIPS

ORIG_TYPES = {c: pl.Utf8 for c in ORIG_COLS}
THREADS = int(os.environ.get("ECL_THREADS", "18"))
MEM = os.environ.get("ECL_MEM", "48GB")
ID_RE = r"^F\d{2}Q[1-4]\d{7}$"


def extract(q: str, work) -> tuple:
    year = q[:4]
    inner = f"historical_data_{q}.zip"
    with zipfile.ZipFile(ZIPS / f"historical_data_{year}.zip") as z:
        z.extract(inner, work)
    subprocess.run(["unzip", "-o", "-q", str(work / inner), "-d", str(work)], check=True)
    (work / inner).unlink()
    return work / f"orig_{q}.txt", work / f"perf_{q}.txt"


def first_line_fields(path) -> int:
    with open(path, encoding="latin-1") as f:
        return len(f.readline().rstrip("\r\n").split("|"))


def main() -> None:
    q = sys.argv[1]
    t0 = time.time()
    for d in ("orig", "events", "snap", "log"):
        (DERIVED / d).mkdir(parents=True, exist_ok=True)
    work = DERIVED / "work" / q
    work.mkdir(parents=True, exist_ok=True)
    orig_txt, perf_txt = extract(q, work)
    log = {"quarter": q, "orig_fields": first_line_fields(orig_txt),
           "perf_fields": first_line_fields(perf_txt)}
    assert log["orig_fields"] == len(ORIG_COLS), log
    assert log["perf_fields"] == len(PERF_COLS), log

    # Origination file. Every field is read as text first so that a malformed
    # value fails the cast loudly instead of silently becoming null.
    o = pl.read_csv(orig_txt, separator="|", has_header=False, schema=ORIG_TYPES, encoding="utf8-lossy", quote_char=None)
    # A lone "." marks a value Freddie Mac did not disclose (seen once, an
    # interest rate in 2006Q3); it becomes missing and is counted.
    num_cols = ("fico", "first_pay", "maturity", "mi_pct", "units", "cltv", "dti", "ltv",
                "orig_term", "n_borrowers", "orig_upb", "orig_rate")
    log["orig_dot_values"] = {c: int((o[c].str.strip_chars() == ".").sum()) for c in num_cols
                              if (o[c].str.strip_chars() == ".").any()}
    o = o.with_columns([pl.when(pl.col(c).str.strip_chars() == ".").then(None).otherwise(pl.col(c)).alias(c)
                        for c in num_cols])
    o = o.with_columns(
        [pl.col(c).str.strip_chars().cast(pl.Int32, strict=True)
         for c in ("fico", "first_pay", "maturity", "mi_pct", "units", "cltv", "dti",
                   "ltv", "orig_term", "n_borrowers")]
        + [pl.col(c).str.strip_chars().cast(pl.Float64, strict=True) for c in ("orig_upb", "orig_rate")]
        + [pl.col(c).str.strip_chars() for c in ("fthb", "occ", "channel", "ppm", "amort", "state",
                                                 "prop_type", "purpose", "harp", "program", "zip3")]
    )
    log["orig_rows"] = len(o)
    log["orig_bad_id"] = int((~o["loan_id"].str.contains(ID_RE)).sum())
    log["orig_dup_id"] = len(o) - o["loan_id"].n_unique()
    log["orig_not_frm"] = int((o["amort"] != "FRM").sum())
    log["orig_id_quarter_mismatch"] = int((o["loan_id"].str.slice(1, 4) != f"{q[2:4]}{q[4:]}").sum())
    assert log["orig_bad_id"] == 0 and log["orig_dup_id"] == 0, log
    o.select([c for c in ORIG_COLS if c not in ("seller", "pre_harp_id", "msa")]).write_parquet(
        DERIVED / "orig" / f"{q}.parquet")

    # Performance file (up to ~13 GB of text per quarter): DuckDB reads it in
    # parallel and reduces it in SQL. Every field is read as text and cast
    # with CAST, which raises on a malformed value rather than nulling it.
    con = duckdb.connect()
    con.execute(f"SET threads={THREADS}; SET memory_limit='{MEM}'; SET preserve_insertion_order=false")
    cols = ", ".join(f"'{c}': 'VARCHAR'" for c in PERF_COLS)
    con.execute(f"""
        CREATE TABLE p AS SELECT
          loan_id, CAST(period AS INTEGER) period, CAST(upb AS DOUBLE) upb, dlq,
          CAST(CASE dlq WHEN 'RA' THEN '99' WHEN 'XX' THEN NULL ELSE dlq END AS SMALLINT) dlq_m,
          CAST(rem_months AS INTEGER) rem_months, defect_date, mod_flag, zb_code,
          CAST(zb_date AS INTEGER) zb_date, CAST(cur_rate AS DOUBLE) cur_rate,
          CAST(nib_upb AS DOUBLE) nib_upb, CAST(actual_loss AS DOUBLE) actual_loss,
          CAST(zb_upb AS DOUBLE) zb_upb, CAST(cum_mod_cost AS DOUBLE) cum_mod_cost, deferral
        FROM read_csv('{perf_txt}', delim='|', header=false, quote='', escape='',
                      auto_detect=false, columns={{{cols}}}, null_padding=false)""")
    con.register("o_ids", o.select("loan_id").to_arrow())

    def one(sql: str):
        return con.execute(sql).fetchone()[0]

    log["perf_rows"] = one("SELECT count(*) FROM p")
    log["perf_null_period"] = one("SELECT count(*) FROM p WHERE period IS NULL")
    log["perf_bad_dlq"] = one("SELECT count(*) FROM p WHERE dlq IS NULL")
    log["perf_unknown_loans"] = one("SELECT count(*) FROM p ANTI JOIN o_ids USING (loan_id)")
    log["perf_dup_loan_period"] = log["perf_rows"] - one("SELECT count(*) FROM (SELECT DISTINCT loan_id, period FROM p)")
    assert log["perf_null_period"] == 0 and log["perf_unknown_loans"] == 0, log
    log["zb_codes"] = dict(con.execute("SELECT zb_code, count(*) FROM p WHERE zb_code IS NOT NULL GROUP BY 1 ORDER BY 1").fetchall())
    log["dlq_top"] = dict(con.execute("SELECT dlq, count(*) c FROM p GROUP BY 1 ORDER BY c DESC LIMIT 12").fetchall())

    con.execute(f"""COPY (SELECT loan_id,
          count(*)::INTEGER n_rec, min(period) first_period, max(period) last_period,
          arg_max(upb, period) last_upb,
          max(zb_code) zb_code, count(zb_code)::INTEGER n_zb,
          max(zb_date) zb_date,
          arg_max(zb_upb, period) FILTER (WHERE zb_upb IS NOT NULL) zb_upb,
          arg_max(actual_loss, period) FILTER (WHERE actual_loss IS NOT NULL) actual_loss,
          max(defect_date) defect_date,
          arg_max(cum_mod_cost, period) FILTER (WHERE cum_mod_cost IS NOT NULL) cum_mod_cost,
          bool_or(mod_flag IS NOT NULL) ever_mod, bool_or(deferral IS NOT NULL) ever_deferral,
          max(dlq_m) max_dlq,
          min(period) FILTER (WHERE dlq_m >= {DEFAULT_MONTHS}) first_180,
          arg_min(upb, period) FILTER (WHERE dlq_m >= {DEFAULT_MONTHS}) upb_180,
          arg_min(nib_upb, period) FILTER (WHERE dlq_m >= {DEFAULT_MONTHS}) nib_180
        FROM p GROUP BY loan_id) TO '{DERIVED / "events" / f"{q}.parquet"}' (FORMAT parquet)""")
    ev = pl.read_parquet(DERIVED / "events" / f"{q}.parquet")
    log["loans_perf"] = len(ev)
    log["loans_multi_zb"] = int((ev["n_zb"] > 1).sum())
    log["loans_without_perf"] = len(o) - len(ev)

    con.execute(f"""COPY (SELECT loan_id, period, upb, nib_upb, dlq_m, rem_months, cur_rate,
          mod_flag IS NOT NULL AS mod, zb_code IS NOT NULL AS zb
        FROM p WHERE period % 100 = 12) TO '{DERIVED / "snap" / f"{q}.parquet"}' (FORMAT parquet)""")
    log["snap_rows"] = one(f"SELECT count(*) FROM read_parquet('{DERIVED / 'snap' / f'{q}.parquet'}')")
    con.close()

    orig_txt.unlink()
    perf_txt.unlink()
    shutil.rmtree(work)
    log["seconds"] = round(time.time() - t0, 1)
    (DERIVED / "log" / f"{q}.json").write_text(json.dumps(log, indent=1))
    print(q, json.dumps({k: log[k] for k in ("orig_rows", "perf_rows", "snap_rows", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
