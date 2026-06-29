"""Tiny cache I/O helper: parquet when an engine is available, else CSV.

The real WRDS pull is 15-30M rows -> parquet (install pyarrow; see environment.yml).
But so the providers stay testable in environments without a parquet engine, we
transparently fall back to CSV. Readers detect whichever file exists.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _parquet_available() -> bool:
    for mod in ("pyarrow", "fastparquet"):
        try:
            __import__(mod)
            return True
        except Exception:
            continue
    return False


PARQUET = _parquet_available()


def write_table(df: pd.DataFrame, cache_dir, name: str) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if PARQUET:
        path = cache_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
    else:
        path = cache_dir / f"{name}.csv"
        df.to_csv(path, index=False)
    return path


def read_table(cache_dir, name: str, parse_dates=None) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    pq = cache_dir / f"{name}.parquet"
    csv = cache_dir / f"{name}.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
        for c in (parse_dates or []):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
        return df
    if csv.exists():
        return pd.read_csv(csv, parse_dates=parse_dates)
    raise FileNotFoundError(
        f"no cache table '{name}' (.parquet or .csv) in {cache_dir} - run ingest_wrds.py first")
