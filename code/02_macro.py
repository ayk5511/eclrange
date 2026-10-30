"""Download the public FRED series the design names and build the macro tables.

Usage: python code/02_macro.py [--offline]

Series (FRED, fredgraph.csv): {ST}STHPI, the FHFA all-transactions house price
index by state (quarterly, not seasonally adjusted); {ST}UR, the BLS
unemployment rate by state (monthly, seasonally adjusted); MORTGAGE30US, the
Freddie Mac PMMS 30-year fixed rate (weekly).

Outputs (MACRO/):
  raw/{id}.csv            the downloaded files, untouched
  raw/SHA256SUMS          their checksums
  hpi_quarter.parquet     state, quarter index (year*4 + q - 1), HPI level
  state_year.parquet      state, year: Q4 HPI, HPI growth over the year,
                          December unemployment rate and its 12-month change
  pmms_month.parquet      month (YYYYMM), mean of the weekly PMMS rate
"""
from __future__ import annotations

import hashlib
import io
import sys
import subprocess
import time

import polars as pl

from ecl import MACRO

STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL",
          "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE",
          "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
          "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def fetch(sid: str) -> None:
    out = MACRO / "raw" / f"{sid}.csv"
    if out.exists():
        return
    # curl over HTTP/1.1: FRED's HTTP/2 endpoint resets some streams.
    for attempt in range(5):
        r = subprocess.run(["curl", "-sS", "--http1.1", "-m", "120", URL.format(sid)],
                           capture_output=True)
        if r.returncode == 0 and r.stdout:
            data = r.stdout
            break
        time.sleep(2 + 3 * attempt)
    else:
        raise RuntimeError(f"could not download {sid}")
    assert data.startswith(b"observation_date,") or data.startswith(b"DATE,"), (sid, data[:80])
    out.write_bytes(data)


def read(sid: str) -> pl.DataFrame:
    df = pl.read_csv(io.BytesIO((MACRO / "raw" / f"{sid}.csv").read_bytes()), infer_schema_length=0)
    df.columns = ["date", "value"]
    return df.filter(pl.col("value") != ".").with_columns(
        pl.col("date").str.to_date(), pl.col("value").cast(pl.Float64))


def main() -> None:
    (MACRO / "raw").mkdir(parents=True, exist_ok=True)
    ids = [f"{s}STHPI" for s in STATES] + [f"{s}UR" for s in STATES] + ["MORTGAGE30US"]
    if "--offline" not in sys.argv:
        for sid in ids:
            fetch(sid)
    sums = [f"{hashlib.sha256((MACRO / 'raw' / f'{i}.csv').read_bytes()).hexdigest()}  {i}.csv" for i in ids]
    (MACRO / "raw" / "SHA256SUMS").write_text("\n".join(sums) + "\n")

    hpi, ur = [], []
    for s in STATES:
        h = read(f"{s}STHPI").with_columns(state=pl.lit(s))
        assert (h["date"].dt.day() == 1).all() and h["date"].dt.month().is_in([1, 4, 7, 10]).all(), s
        hpi.append(h)
        ur.append(read(f"{s}UR").with_columns(state=pl.lit(s)))
    hpi = pl.concat(hpi).with_columns(
        qidx=(pl.col("date").dt.year() * 4 + (pl.col("date").dt.month() - 1) // 3).cast(pl.Int32))
    hpi.select("state", "qidx", hpi="value").write_parquet(MACRO / "hpi_quarter.parquet")

    q4 = hpi.filter(pl.col("date").dt.month() == 10).select(
        "state", year=pl.col("date").dt.year().cast(pl.Int32), hpi_q4="value")
    dec = pl.concat(ur).filter(pl.col("date").dt.month() == 12).select(
        "state", year=pl.col("date").dt.year().cast(pl.Int32), ur="value")
    sy = (q4.join(dec, on=["state", "year"], how="full", coalesce=True).sort(["state", "year"])
          .with_columns(
              hpi_g=(pl.col("hpi_q4") / pl.col("hpi_q4").shift(1) - 1).over("state"),
              ur_chg=(pl.col("ur") - pl.col("ur").shift(1)).over("state")))
    # A one-year growth or change needs the previous year in the same state
    # with no gap; the series are complete, which this asserts.
    gaps = sy.with_columns(d=pl.col("year").diff().over("state")).filter(pl.col("d") > 1)
    assert gaps.is_empty(), gaps
    sy.write_parquet(MACRO / "state_year.parquet")

    pm = read("MORTGAGE30US").with_columns(
        month=(pl.col("date").dt.year() * 100 + pl.col("date").dt.month()).cast(pl.Int32))
    pm.group_by("month").agg(pmms=pl.col("value").mean()).sort("month").write_parquet(
        MACRO / "pmms_month.parquet")
    print(f"states {len(STATES)}  state-years {len(sy)}  "
          f"HPI {hpi['date'].min()}..{hpi['date'].max()}  UR Dec years {dec['year'].min()}..{dec['year'].max()}")


if __name__ == "__main__":
    main()
