#!/usr/bin/env python3
"""
Fetch 5-minute bars from Yahoo Finance and save them as CSVs that
backtest.py can read directly.

    pip install yfinance pandas
    python fetch_data.py                 # default tickers -> ./data/
    python fetch_data.py --days 59 --out ./data

Note: Yahoo only provides 5m intraday data for ~the last 60 days.
"""

import argparse
import os
import sys

import pandas as pd
import yfinance as yf

# Tickers reconstructed from the order history, plus benchmark + a small
# control basket of liquid names Dad did NOT trade (edit freely).
TRADED = ["AAPL", "TSLA", "MU", "WDC", "HOOD", "NTES", "ALAB", "NVDA",
          "PLTR", "MSFT", "META", "SNDK", "NFLX", "GLW", "COHR", "NBIS",
          "APTV", "SPCX", "IVDA", "FERG", "TROW"]
CONTROL = ["AMD", "INTC", "QCOM", "AVGO", "CRM", "ORCL", "ADBE", "COIN",
           "SHOP", "UBER", "ABNB", "SQ", "PYPL", "SNOW", "DDOG"]
BENCHMARKS = ["QQQ", "SPY"]


def fetch_one(ticker: str, days: int, out_dir: str) -> bool:
    try:
        df = yf.download(ticker, period=f"{days}d", interval="5m",
                         prepost=False, auto_adjust=False, progress=False)
    except Exception as e:
        print(f"  {ticker}: FAILED ({e})", file=sys.stderr)
        return False
    if df.empty:
        print(f"  {ticker}: no data returned", file=sys.stderr)
        return False
    # yfinance sometimes returns MultiIndex columns even for one ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower).reset_index()
    tcol = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df["time"] = pd.to_datetime(df[tcol]).dt.tz_localize(None)
    cols = ["time", "open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
    out = df[cols].dropna()
    path = os.path.join(out_dir, f"{ticker}.csv")
    out.to_csv(path, index=False)
    print(f"  {ticker}: {len(out)} bars  ({out.time.min()} -> {out.time.max()})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=59,
                    help="lookback in days (Yahoo caps 5m data at ~60)")
    ap.add_argument("--out", default="data", help="output folder")
    ap.add_argument("--skip-control", action="store_true",
                    help="only fetch traded tickers + benchmarks")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tickers = BENCHMARKS + TRADED + ([] if args.skip_control else CONTROL)
    print(f"Fetching {len(tickers)} tickers, {args.days} days of 5m bars...")
    ok = sum(fetch_one(t, args.days, args.out) for t in tickers)
    print(f"\nDone: {ok}/{len(tickers)} tickers saved to {args.out}/")
    print("\nNext steps:")
    print(f"  python backtest.py --data-dir {args.out} --benchmark {args.out}/QQQ.csv")
    print("  # and for the honest test, split by date: run on pre-July-22 data")


if __name__ == "__main__":
    main()