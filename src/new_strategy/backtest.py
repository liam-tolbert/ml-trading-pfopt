#!/usr/bin/env python3
"""
Relative mean-reversion dip-buyer backtest
==========================================

Codifies the strategy hypothesis recovered from Jason's 24 paper trades:

  1. Stock declined over the prior 60 minutes (12 x 5-min bars).
  2. Stock UNDERPERFORMED the benchmark (QQQ) over that same hour.
  3. Price sits in the lower portion of its trailing 100-minute range.
  4. The decline is decelerating: slope of the most recent 6 bars is
     higher (less negative) than the slope of the 6 bars before that
     ("the parabola is flattening").
  5. Guard against structural repricing: skip if the stock is down more
     than `max_day_drawdown` from the session high (the APTV/SPCX
     failure mode — buying a bounce inside a fundamental repricing).
  6. Enter with a limit order slightly below the signal bar's close.
  7. No tight stop. Exit on: (a) price recovering to the upper region of
     the entry-time range (profit target), (b) a wide "structure broken"
     stop, or (c) a max-hold timeout.

Data format: TradingView 5-minute CSV exports (the same files the
strategy's author already produces). Only time/open/high/low/close are
required; extra indicator columns are ignored. A benchmark file (QQQ)
is required for the relative-weakness condition.

Usage:
    python backtest.py --data-dir ./data --benchmark ./data/QQQ.csv
    python backtest.py --data-dir ./data --benchmark ./data/QQQ.csv --sweep

This is a research tool for paper-trading analysis, not investment advice.
"""

import argparse
import glob
import itertools
import math
import os
import sys
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Parameters (defaults calibrated to the 24 reconstructed paper trades)
# ----------------------------------------------------------------------
@dataclass
class Params:
    # --- entry conditions -------------------------------------------------
    lookback_ret_bars: int = 12      # 60 min of 5-min bars
    require_negative_ret: bool = True
    rel_underperf_max: float = -0.5  # stock 60-min return minus QQQ 60-min
                                     # return must be <= this (%). Winners
                                     # in the sample ranged -1.2% to -5.5%.
    range_bars: int = 20             # 100-minute trailing range
    range_pos_max: float = 0.35      # enter only in lower ~1/3 of range
    require_deceleration: bool = True
    slope_bars: int = 6              # bars per slope window for curvature
    max_day_drawdown: float = -12.0  # skip if > this % below session high
                                     # (structural-repricing guard)
    limit_discount: float = 0.15     # limit order this % below signal close
    limit_ttl_bars: int = 6          # cancel unfilled limit after 30 min

    # --- screener gate (candidate-universe replication) -------------------
    # Replicates the "Bull Flag Momentum" TradingView screener: only allow
    # entries while the stock currently passes the same filters Dad's
    # screener applied. Toggle with --no-screener to compare.
    use_screener: bool = True
    scr_day_chg_min: float = 5.0     # up >5% on the day vs prior close
    scr_from_open_min: float = -5.0  # change from today's open between
    scr_from_open_max: float = 0.0   #   -5% and 0% (the pullback)
    scr_rel_vol_min: float = 1.5     # relative volume (needs volume data)
    scr_atr_pct_min: float = 2.0     # 14-day ATR as % of price
    scr_or_gap: bool = True          # ALSO admit pre-market-gap candidates:
    scr_gap_min: float = 2.0         #   open >2% above prior close

    # --- exit conditions --------------------------------------------------
    target_range_pos: float = 0.75   # take profit when price recovers to
                                     # this position in the ENTRY-time range
    min_target_pct: float = 0.4      # but never accept less than this %
    structural_stop_pct: float = -10.0  # wide stop; winners survived MAE
                                        # as bad as -8.8% (COHR)
    max_hold_bars: int = 160         # ~2 trading days of 5-min bars
    eod_flat: bool = False           # True = no overnight holds (author
                                     # holds overnight, so default False)

    # --- accounting -------------------------------------------------------
    position_notional: float = 40_000.0  # $ per trade (median observed)
    one_position_per_symbol: bool = True
    cooldown_bars: int = 6           # bars to wait after an exit


# ----------------------------------------------------------------------
# Data loading — tolerant of TradingView export quirks
# ----------------------------------------------------------------------
def load_tv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in df.columns if c in ("time", "date", "datetime", "timestamp")), None)
    if tcol is None:
        raise ValueError(f"{path}: no time column found")
    # TradingView exports either ISO strings or unix seconds
    if pd.api.types.is_numeric_dtype(df[tcol]):
        df["time"] = pd.to_datetime(df[tcol], unit="s", utc=True).dt.tz_localize(None)
    else:
        df["time"] = pd.to_datetime(df[tcol], utc=True, errors="coerce").dt.tz_localize(None)
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"{path}: missing '{col}' column")
    keep = ["time", "open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
    out = (df[keep]
           .dropna()
           .sort_values("time")
           .drop_duplicates("time")
           .reset_index(drop=True))
    return out


def symbol_from_path(path: str) -> str:
    """Best-effort ticker from a TradingView filename like
    'NASDAQ_AAPL, 5_abc123.csv' or 'AAPL.csv'."""
    name = os.path.basename(path)
    name = name.split(",")[0].split(".csv")[0]
    if "_" in name:
        parts = name.split("_")
        # NASDAQ_AAPL -> AAPL ; keep last alpha chunk
        name = parts[1] if len(parts) > 1 and parts[1].isalpha() else parts[0]
    return name.strip().upper()


# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------
def slope(y: np.ndarray) -> float:
    """OLS slope of y vs bar index, in % of the window's mean price per bar."""
    n = len(y)
    if n < 2 or np.any(~np.isfinite(y)):
        return np.nan
    x = np.arange(n, dtype=float)
    b = np.polyfit(x, y, 1)[0]
    m = np.mean(y)
    return 100.0 * b / m if m else np.nan


def add_features(df: pd.DataFrame, bench: pd.DataFrame, p: Params) -> pd.DataFrame:
    df = df.copy()
    # align benchmark closes onto the stock's timestamps (forward-fill)
    b = bench.set_index("time")["close"]
    df["bench_close"] = df["time"].map(b).astype(float)
    df["bench_close"] = df["bench_close"].ffill()

    k = p.lookback_ret_bars
    df["ret_60"] = 100.0 * (df["close"] / df["close"].shift(k) - 1)
    df["bench_ret_60"] = 100.0 * (df["bench_close"] / df["bench_close"].shift(k) - 1)
    df["rel_60"] = df["ret_60"] - df["bench_ret_60"]

    r = p.range_bars
    lo = df["low"].rolling(r).min()
    hi = df["high"].rolling(r).max()
    df["range_lo"], df["range_hi"] = lo, hi
    rng = (hi - lo).replace(0, np.nan)
    df["range_pos"] = (df["close"] - lo) / rng

    s = p.slope_bars
    closes = df["close"].to_numpy()
    slopes_recent = np.full(len(df), np.nan)
    slopes_prior = np.full(len(df), np.nan)
    for i in range(2 * s - 1, len(df)):
        slopes_recent[i] = slope(closes[i - s + 1: i + 1])
        slopes_prior[i] = slope(closes[i - 2 * s + 1: i - s + 1])
    df["slope_recent"], df["slope_prior"] = slopes_recent, slopes_prior
    df["decelerating"] = df["slope_recent"] > df["slope_prior"]

    # intraday drawdown from session high (structural-repricing guard)
    sess = df["time"].dt.date
    day_high = df.groupby(sess)["high"].cummax()
    df["day_dd"] = 100.0 * (df["close"] / day_high - 1)

    # ---- screener-gate features -----------------------------------------
    df["session"] = sess
    day_open = df.groupby("session")["open"].transform("first")
    daily = df.groupby("session").agg(d_high=("high", "max"), d_low=("low", "min"),
                                      d_close=("close", "last"))
    prior_close = df["session"].map(daily["d_close"].shift(1))
    df["day_chg"] = 100.0 * (df["close"] / prior_close - 1)          # vs prior close
    df["from_open"] = 100.0 * (df["close"] / day_open - 1)           # vs today's open
    df["gap"] = 100.0 * (day_open / prior_close - 1)                 # open vs prior close

    # 14-day ATR% (classic true range on daily aggregates)
    d = daily.copy()
    d["pc"] = d["d_close"].shift(1)
    tr = pd.concat([d["d_high"] - d["d_low"],
                    (d["d_high"] - d["pc"]).abs(),
                    (d["d_low"] - d["pc"]).abs()], axis=1).max(axis=1)
    atr_pct = 100.0 * (tr.rolling(14, min_periods=5).mean() / d["d_close"])
    df["atr_pct"] = df["session"].map(atr_pct.shift(1))  # known at day start

    # relative volume proxy: today's cumulative volume, scaled up by the
    # fraction of the session elapsed, vs the 10-day average daily volume
    if "volume" in df.columns:
        cumv = df.groupby("session")["volume"].cumsum()
        bar_n = df.groupby("session").cumcount() + 1
        bars_per_day = df.groupby("session")["time"].transform("count")
        proj_day_vol = cumv * bars_per_day / bar_n
        avg_day_vol = df["session"].map(
            df.groupby("session")["volume"].sum().rolling(10, min_periods=3).mean().shift(1))
        df["rel_vol"] = proj_day_vol / avg_day_vol
    else:
        df["rel_vol"] = np.nan
    return df


def passes_screener(row: pd.Series, p: Params) -> bool:
    """Would this bar's stock currently appear on the screeners?"""
    momentum_ok = (
        np.isfinite(row["day_chg"]) and row["day_chg"] >= p.scr_day_chg_min
        and np.isfinite(row["from_open"])
        and p.scr_from_open_min <= row["from_open"] <= p.scr_from_open_max
    )
    gap_ok = p.scr_or_gap and np.isfinite(row["gap"]) and row["gap"] >= p.scr_gap_min
    if not (momentum_ok or gap_ok):
        return False
    if np.isfinite(row["atr_pct"]) and row["atr_pct"] < p.scr_atr_pct_min:
        return False
    # rel-vol filter only enforceable when volume data exists
    if np.isfinite(row["rel_vol"]) and row["rel_vol"] < p.scr_rel_vol_min:
        return False
    return True


def entry_signal(row: pd.Series, p: Params) -> bool:
    if not np.isfinite(row["rel_60"]) or not np.isfinite(row["range_pos"]):
        return False
    if p.use_screener and not passes_screener(row, p):
        return False
    if p.require_negative_ret and not (row["ret_60"] < 0):
        return False
    if row["rel_60"] > p.rel_underperf_max:
        return False
    if row["range_pos"] > p.range_pos_max:
        return False
    if p.require_deceleration and not bool(row["decelerating"]):
        return False
    if np.isfinite(row["day_dd"]) and row["day_dd"] < p.max_day_drawdown:
        return False
    return True


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------
@dataclass
class Trade:
    symbol: str
    entry_time: object
    exit_time: object
    entry: float
    exit: float
    shares: float
    reason: str
    ret_pct: float
    pnl: float
    mae_pct: float
    mfe_pct: float
    hold_bars: int


def run_symbol(symbol: str, df: pd.DataFrame, p: Params) -> list:
    trades = []
    i, n = 0, len(df)
    pending = None       # (limit_price, placed_index)
    pos = None           # dict with entry info
    cooldown_until = -1

    while i < n:
        row = df.iloc[i]

        if pos is not None:
            entry_px = pos["entry"]
            pos["mae"] = min(pos["mae"], 100.0 * (row["low"] / entry_px - 1))
            pos["mfe"] = max(pos["mfe"], 100.0 * (row["high"] / entry_px - 1))
            reason = None
            exit_px = None
            # profit target: recover into upper region of entry-time range
            if row["high"] >= pos["target"]:
                exit_px, reason = max(pos["target"], row["open"]), "target"
            elif 100.0 * (row["low"] / entry_px - 1) <= p.structural_stop_pct:
                exit_px = entry_px * (1 + p.structural_stop_pct / 100.0)
                exit_px, reason = min(exit_px, row["open"]), "stop"
            elif i - pos["i"] >= p.max_hold_bars:
                exit_px, reason = row["close"], "timeout"
            elif p.eod_flat and (i + 1 >= n or df.iloc[i + 1]["time"].date() != row["time"].date()):
                exit_px, reason = row["close"], "eod"
            elif i == n - 1:
                exit_px, reason = row["close"], "end_of_data"
            if reason:
                ret = 100.0 * (exit_px / entry_px - 1)
                trades.append(Trade(symbol, pos["time"], row["time"], entry_px,
                                    exit_px, pos["shares"], reason, ret,
                                    pos["shares"] * (exit_px - entry_px),
                                    pos["mae"], pos["mfe"], i - pos["i"]))
                pos = None
                cooldown_until = i + p.cooldown_bars
            i += 1
            continue

        # fill pending limit order?
        if pending is not None:
            limit_px, placed = pending
            if i - placed > p.limit_ttl_bars:
                pending = None
            elif row["low"] <= limit_px:
                fill = min(limit_px, row["open"])
                shares = p.position_notional / fill
                target = max(
                    row["range_lo"] + p.target_range_pos * (df.iloc[placed]["range_hi"] - df.iloc[placed]["range_lo"]),
                    fill * (1 + p.min_target_pct / 100.0),
                )
                pos = {"entry": fill, "time": row["time"], "i": i, "shares": shares,
                       "target": target, "mae": 0.0, "mfe": 0.0}
                pending = None
                i += 1
                continue

        # new signal?
        if pending is None and i > cooldown_until and entry_signal(row, p):
            limit_px = row["close"] * (1 - p.limit_discount / 100.0)
            pending = (limit_px, i)
        i += 1

    return trades


def summarize(trades: list, p: Params) -> dict:
    if not trades:
        return {"trades": 0}
    t = pd.DataFrame([asdict(x) for x in trades])
    wins = t[t.pnl > 0]
    return {
        "trades": len(t),
        "win_rate": round((t.pnl > 0).mean(), 3),
        "total_pnl": round(t.pnl.sum(), 0),
        "avg_ret_pct": round(t.ret_pct.mean(), 3),
        "median_ret_pct": round(t.ret_pct.median(), 3),
        "avg_win_pct": round(wins.ret_pct.mean(), 3) if len(wins) else 0,
        "avg_loss_pct": round(t[t.pnl <= 0].ret_pct.mean(), 3) if (t.pnl <= 0).any() else 0,
        "worst_mae_pct": round(t.mae_pct.min(), 2),
        "median_hold_bars": int(t.hold_bars.median()),
        "profit_factor": round(wins.pnl.sum() / abs(t[t.pnl <= 0].pnl.sum()), 2)
        if (t.pnl <= 0).any() and t[t.pnl <= 0].pnl.sum() != 0 else float("inf"),
    }


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def run_all(data_dir: str, benchmark_path: str, p: Params, out_csv=None, quiet=False):
    bench = load_tv_csv(benchmark_path)
    all_trades = []
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    files = [f for f in files if os.path.abspath(f) != os.path.abspath(benchmark_path)]
    if not files:
        sys.exit(f"No CSV files found in {data_dir}")
    for f in files:
        sym = symbol_from_path(f)
        if sym in ("QQQ", "SPY"):
            continue
        try:
            df = load_tv_csv(f)
        except ValueError as e:
            print(f"skip {f}: {e}", file=sys.stderr)
            continue
        df = add_features(df, bench, p)
        all_trades += run_symbol(sym, df, p)

    all_trades.sort(key=lambda t: t.entry_time)
    stats = summarize(all_trades, p)
    if not quiet:
        print(f"\n{'='*60}\nBacktest results  ({len(files)} symbols)\n{'='*60}")
        for k, v in stats.items():
            print(f"  {k:>18}: {v}")
    if out_csv and all_trades:
        pd.DataFrame([asdict(t) for t in all_trades]).to_csv(out_csv, index=False)
        if not quiet:
            print(f"\nTrade log written to {out_csv}")
    return stats, all_trades


def sweep(data_dir: str, benchmark_path: str, base: Params):
    """Small grid search over the parameters most likely to matter."""
    grid = {
        "rel_underperf_max": [-0.25, -0.5, -1.0, -1.5],
        "range_pos_max": [0.25, 0.35, 0.5],
        "target_range_pos": [0.6, 0.75, 0.9],
        "structural_stop_pct": [-6.0, -10.0, -15.0],
    }
    rows = []
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        p = Params(**{**asdict(base), **dict(zip(keys, combo))})
        stats, _ = run_all(data_dir, benchmark_path, p, quiet=True)
        rows.append({**dict(zip(keys, combo)), **stats})
    res = pd.DataFrame(rows).sort_values("total_pnl", ascending=False)
    res.to_csv("sweep_results.csv", index=False)
    print(res.head(15).to_string(index=False))
    print("\nFull grid written to sweep_results.csv")
    print("NOTE: picking the best row here is overfitting to one sample. "
          "Use the sweep to check ROBUSTNESS (do most rows make money?), "
          "not to choose 'the' parameters.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder of TradingView 5-min CSV exports")
    ap.add_argument("--benchmark", required=True, help="path to QQQ 5-min CSV")
    ap.add_argument("--out", default="trades.csv", help="trade log output path")
    ap.add_argument("--sweep", action="store_true", help="run parameter robustness sweep")
    ap.add_argument("--eod-flat", action="store_true", help="force flat at end of each day")
    ap.add_argument("--no-screener", action="store_true",
                    help="disable the candidate-universe screener gate")
    args = ap.parse_args()

    p = Params(eod_flat=args.eod_flat, use_screener=not args.no_screener)
    if args.sweep:
        sweep(args.data_dir, args.benchmark, p)
    else:
        run_all(args.data_dir, args.benchmark, p, out_csv=args.out)


if __name__ == "__main__":
    main()