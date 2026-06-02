#!/usr/bin/env python3
"""
Standalone ETF Markowitz rebalancer.

Computes today's target weights for a fixed 7-ETF universe from the most recent
trailing window (the live analogue of one iteration of the weekly Markowitz
backtest in ETF_Markowitz.ipynb), diffs them against the portfolio's current
weights, and reports the trades needed to move current -> target.

This script is intentionally SELF-CONTAINED: it inlines the price loader and the
weighting math rather than importing from src/backtest_lib.py (or any other
helper module), so it has no code dependency on the rest of the repo.

The "current portfolio" is represented as a plain-text weights file
(default: data/etf_current_weights.txt), one `TICKER WEIGHT` per line. If the
file is missing, an example equal-weight (1/7) baseline is generated.

Usage (run from the project root so data/... paths resolve):
    python src/rebalance_etf.py
    python src/rebalance_etf.py --scheme equal_weight
    python src/rebalance_etf.py --scheme min_variance --capital 100000
    python src/rebalance_etf.py --scheme max_sharpe --save
    python src/rebalance_etf.py --no-refresh          # cached prices only
    python src/rebalance_etf.py --force-refresh       # rebuild prices
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

sys.stdout.reconfigure(encoding="utf-8")  # Fix Windows console encoding


# =============================================================================
#  Configuration (mirrors ETF_Markowitz.ipynb cell 1)
# =============================================================================

UNIVERSE         = ["QQQ", "VOO", "VGT", "SOXX", "QUAL", "VBR", "VTV"]
BENCH            = "^GSPC"          # SPY proxy (loaded for cache parity; not weighted)
COV_LOOKBACK     = 52              # trailing weeks for mu / covariance
SPREAD_PER_SHARE = 0.02           # $/share bid-ask; conservative for liquid ETFs
RISK_FREE        = 0.0            # weekly-annualized rf for max-Sharpe
DATA_START       = "2015-01-01"
PRICE_CSV        = "data/etf_prices.csv"
WEIGHTS_FILE     = "data/etf_current_weights.txt"
WEEKS_PER_YEAR   = 52


# =============================================================================
#  Price loader (inlined from ETF_Markowitz.ipynb cell 3)
# =============================================================================

def _last_friday() -> pd.Timestamp:
    today = pd.Timestamp.today().normalize()
    return today - pd.Timedelta(days=(today.weekday() - 4) % 7)


def download_weekly_closes(tickers, csv_path=PRICE_CSV, start=DATA_START,
                           force_refresh=False):
    """Wide weekly (W-FRI) auto-adjusted close matrix: Date x ticker.

    Reuses cached columns whose last value reaches the most recent Friday;
    only missing/stale tickers are downloaded. Writes the merged matrix back
    to csv_path (gitignored)."""
    threshold = _last_friday() - pd.Timedelta(days=1)

    cached, fresh = None, set()
    if not force_refresh and os.path.exists(csv_path):
        cached = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")
        for col in cached.columns:
            s = cached[col].dropna()
            if not s.empty and s.index.max() >= threshold:
                fresh.add(col)

    to_dl = [t for t in tickers if t not in fresh]
    print(f"{len(tickers) - len(to_dl)} cached + fresh; {len(to_dl)} to fetch from yfinance")

    frames = []
    for t in to_dl:
        df = yf.download(t, interval="1d", start=start,
                         auto_adjust=True, progress=False)
        if df.empty:
            print(f"  WARNING: no data for {t}")
            continue
        close = df["Close"]                      # handle MultiIndex or flat columns
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        wk = close.resample("W-FRI").last()
        wk.name = t
        frames.append(wk)

    new = pd.concat(frames, axis=1) if frames else None
    if cached is not None:
        keep = cached[[c for c in cached.columns if c in fresh]]
        combined = keep if new is None else pd.concat([keep, new], axis=1)
    else:
        combined = new
    combined = combined.sort_index()
    combined = combined[[t for t in tickers if t in combined.columns]]  # order + prune
    combined.index.name = "Date"
    combined.to_csv(csv_path)
    return combined


def load_prices(no_refresh: bool, force_refresh: bool) -> pd.DataFrame:
    """Load the wide weekly-close matrix. Refresh from yfinance, falling back to
    the cached CSV on any network/yfinance error (per 'refresh, fall back to
    cache'). --no-refresh skips the network; --force-refresh rebuilds."""
    tickers = UNIVERSE + [BENCH]

    if no_refresh:
        if not os.path.exists(PRICE_CSV):
            sys.exit(f"ERROR: --no-refresh set but {PRICE_CSV} does not exist.")
        print(f"Using cached prices only: {PRICE_CSV}")
        prices = pd.read_csv(PRICE_CSV, parse_dates=["Date"]).set_index("Date")
        return prices[[t for t in tickers if t in prices.columns]].ffill()

    try:
        return download_weekly_closes(tickers, force_refresh=force_refresh).ffill()
    except Exception as e:
        print(f"  ! price refresh failed ({e}); falling back to cached {PRICE_CSV}")
        if not os.path.exists(PRICE_CSV):
            sys.exit(f"ERROR: refresh failed and no cache at {PRICE_CSV}.")
        prices = pd.read_csv(PRICE_CSV, parse_dates=["Date"]).set_index("Date")
        return prices[[t for t in tickers if t in prices.columns]].ffill()


# =============================================================================
#  Weighting schemes (inlined from ETF_Markowitz.ipynb cell 7)
# =============================================================================

def _equal(cols) -> dict:
    cols = list(cols)
    return {c: 1.0 / len(cols) for c in cols} if cols else {}


def _ef(close_window):
    """Shared EfficientFrontier on trailing prices, or None if not enough data."""
    cw = close_window.dropna(axis=1, how="any")
    if cw.shape[1] < 2 or cw.shape[0] < 3:
        return None, cw
    mu = expected_returns.mean_historical_return(cw, frequency=WEEKS_PER_YEAR)
    S = risk_models.sample_cov(cw, frequency=WEEKS_PER_YEAR)
    return EfficientFrontier(mu, S), cw


def max_sharpe_weights_etf(close_window, risk_free_rate=RISK_FREE) -> dict:
    ef, cw = _ef(close_window)
    if ef is None:
        return _equal(cw.columns)
    try:
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        w = {k: v for k, v in ef.clean_weights().items() if v != 0.0}
        return w or _equal(cw.columns)
    except Exception:
        return _equal(cw.columns)


def min_variance_weights_etf(close_window) -> dict:
    ef, cw = _ef(close_window)
    if ef is None:
        return _equal(cw.columns)
    try:
        ef.min_volatility()
        w = {k: v for k, v in ef.clean_weights().items() if v != 0.0}
        return w or _equal(cw.columns)
    except Exception:
        return _equal(cw.columns)


SCHEMES = {
    "max_sharpe":   max_sharpe_weights_etf,
    "min_variance": min_variance_weights_etf,
    "equal_weight": lambda cw: _equal(cw.columns),
}


# =============================================================================
#  Current-weights file (read / write / example baseline)
# =============================================================================

def read_weights_file(path: str) -> dict | None:
    """Parse a `TICKER WEIGHT` weights file (# comments allowed). None if absent."""
    if not os.path.exists(path):
        return None
    weights = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                print(f"  ! skipping malformed line in {path}: {raw.rstrip()}")
                continue
            ticker, val = parts
            try:
                weights[ticker.upper()] = float(val)
            except ValueError:
                print(f"  ! skipping non-numeric weight in {path}: {raw.rstrip()}")
    return weights


def write_weights_file(path: str, weights: dict, header: str) -> None:
    """Write weights as `TICKER WEIGHT` lines, sorted by descending weight."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ordered = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {header}\n")
        f.write("# ticker   weight   (one position per line; weights sum to ~1.0)\n")
        for ticker, w in ordered:
            f.write(f"{ticker:<6} {w:.6f}\n")


def example_weights() -> dict:
    """Equal-weight 1/N over the universe — the example baseline."""
    return _equal(UNIVERSE)


# =============================================================================
#  Rebalance report
# =============================================================================

def estimate_turnover_cost(target: dict, current: dict, price_row: pd.Series):
    """Total turnover Σ|Δw| and per-ticker spread cost (spread/2)/price·|Δw|."""
    keys = set(target) | set(current)
    turnover, cost = 0.0, 0.0
    for k in keys:
        dw = abs(target.get(k, 0.0) - current.get(k, 0.0))
        if dw == 0.0:
            continue
        turnover += dw
        price = price_row.get(k, np.nan)
        if pd.notna(price) and price > 0:
            cost += (SPREAD_PER_SHARE / 2.0) / float(price) * dw
    return turnover, cost


def print_report(scheme: str, target: dict, current: dict, price_row: pd.Series,
                 as_of: pd.Timestamp, capital: float | None) -> None:
    """Print the target-vs-current diff and (optionally) dollar/share trades."""
    keys = sorted(set(target) | set(current))
    w = 78

    print("\n" + "=" * w)
    print(f"  ETF REBALANCE  |  scheme: {scheme}  |  prices as of {as_of.date()}")
    print("=" * w)
    print(f"  {'Ticker':<8} {'Current':>10} {'Target':>10} {'Delta':>10} "
          f"{'Price':>10} {'Action':>8}")
    print("-" * w)

    for k in keys:
        cur_w = current.get(k, 0.0)
        tgt_w = target.get(k, 0.0)
        dw = tgt_w - cur_w
        price = price_row.get(k, np.nan)
        price_s = f"${price:,.2f}" if pd.notna(price) else "    n/a"
        if abs(dw) < 1e-9:
            action = "hold"
        elif dw > 0:
            action = "BUY"
        else:
            action = "SELL"
        print(f"  {k:<8} {cur_w:>9.2%} {tgt_w:>9.2%} {dw:>+9.2%} "
              f"{price_s:>10} {action:>8}")

    turnover, cost = estimate_turnover_cost(target, current, price_row)
    print("-" * w)
    print(f"  Total turnover (Σ|Δw|): {turnover:.4f}"
          f"   ({turnover/2:.2%} of book traded one-way)")
    print(f"  Est. spread cost:       {cost:.4%} of capital"
          f"   (${cost*capital:,.2f})" if capital else
          f"  Est. spread cost:       {cost:.4%} of capital")

    if capital:
        print("\n" + "-" * w)
        print(f"  DOLLAR / SHARE TRADES  |  capital: ${capital:,.2f}")
        print("-" * w)
        print(f"  {'Ticker':<8} {'Curr $':>14} {'Target $':>14} "
              f"{'Trade $':>14} {'Shares':>10} {'Action':>8}")
        print("-" * w)
        for k in keys:
            cur_w = current.get(k, 0.0)
            tgt_w = target.get(k, 0.0)
            cur_d = cur_w * capital
            tgt_d = tgt_w * capital
            trade_d = tgt_d - cur_d
            price = price_row.get(k, np.nan)
            if pd.notna(price) and price > 0:
                shares = int(round(trade_d / float(price)))
            else:
                shares = 0
            if abs(trade_d) < 1e-9:
                action = "hold"
            elif trade_d > 0:
                action = "BUY"
            else:
                action = "SELL"
            print(f"  {k:<8} {cur_d:>14,.2f} {tgt_d:>14,.2f} "
                  f"{trade_d:>+14,.2f} {shares:>+10,d} {action:>8}")

    print("=" * w)


# =============================================================================
#  Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Standalone ETF Markowitz rebalancer (self-contained)."
    )
    parser.add_argument("--scheme", choices=list(SCHEMES), default="max_sharpe",
                        help="Weighting scheme for the target (default: max_sharpe)")
    parser.add_argument("--capital", type=float, default=None,
                        help="Portfolio value in $ — print dollar/share trades")
    parser.add_argument("--weights-file", default=WEIGHTS_FILE,
                        help=f"Current-weights file (default: {WEIGHTS_FILE})")
    parser.add_argument("--no-refresh", action="store_true",
                        help="Use cached etf_prices.csv only (no network)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Rebuild prices from yfinance")
    parser.add_argument("--save", action="store_true",
                        help="Persist the new target as the current-weights file")
    args = parser.parse_args()

    # --- prices ---
    prices = load_prices(args.no_refresh, args.force_refresh)
    closes = prices[UNIVERSE].dropna()           # weeks where all 7 ETFs exist
    if closes.empty:
        sys.exit("ERROR: no weeks where all universe ETFs have prices.")
    close_window = closes.tail(COV_LOOKBACK)
    as_of = close_window.index[-1]
    price_row = close_window.iloc[-1]
    print(f"{closes.shape[0]} weekly bars "
          f"{closes.index.min().date()} -> {closes.index.max().date()}; "
          f"using trailing {len(close_window)} weeks.")

    # --- target weights ---
    target = SCHEMES[args.scheme](close_window)

    # --- current weights (generate example baseline if missing) ---
    current = read_weights_file(args.weights_file)
    if current is None:
        current = example_weights()
        write_weights_file(
            args.weights_file, current,
            "EXAMPLE baseline (equal-weight 1/N) — auto-generated; edit to your "
            "actual current weights.",
        )
        print(f"\nNo weights file found — wrote example equal-weight baseline to "
              f"{args.weights_file}")
        print("  (Edit it to reflect your real current holdings, then re-run.)")

    # --- report ---
    print_report(args.scheme, target, current, price_row, as_of, args.capital)

    # --- optionally advance the baseline ---
    if args.save:
        write_weights_file(
            args.weights_file, target,
            f"Saved target weights - scheme={args.scheme}, as of {as_of.date()}.",
        )
        print(f"\nSaved new target weights to {args.weights_file} "
              f"(next run will diff against these).")


if __name__ == "__main__":
    main()
