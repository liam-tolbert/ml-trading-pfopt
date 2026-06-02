"""Survivorship-bias audit for the trading universe (cheap, read-only).

THE STRUCTURAL FINDING (established by reading the code, not by this script):
The universe is a HARDCODED CURRENT SNAPSHOT -- 253 tickers in Main.ipynb cell
`a30b8c33`, sampled (seed=42) from a 2025-era spreadsheet, with NO point-in-time
index membership and NO handling of delisted/merged names. So every name in the
2018-2025 backtest is a company that still existed when the list was built. Any
stock that died in that window is ABSENT BY CONSTRUCTION, which biases backtest
returns upward (survivorship bias). That bias is UNMEASURABLE from this data --
the dead names simply aren't here. Removing it would require an external
point-in-time constituent list + delisted price history (out of scope this round).

What this script CAN do is bound the *milder, measurable* slices of the problem:
  1. Requested-but-missing: tickers in the intended list that yfinance never
     returned (silently dropped from the universe).
  2. Late entrants: tickers whose price history starts AFTER the backtest start
     (IPO'd / first-listed mid-window) -- they only contribute once they exist,
     a mild look-ahead-flavored "incubation" effect, not classic survivorship.
  3. Early exits: tickers whose data ends well before the backtest end -- the only
     in-sample hint of a name leaving the universe.

For scale: the academic literature puts survivorship bias in US-equity backtests
on the order of ~1-4%/yr of overstated return; treat the headline backtest numbers
as optimistic by roughly that order until a point-in-time universe is built.

Run from the project root:  python scripts/check_universe_coverage.py
"""
from __future__ import annotations

import ast
import os

import pandas as pd

NB_PATH = "Main.ipynb"
CSV_PATH = "data/training_set.csv"
BT_START = pd.Timestamp("2018-01-01")
BT_END = pd.Timestamp("2025-08-15")
START_TOL = pd.Timedelta(days=21)   # >3wk after BT_START counts as a late entrant
END_TOL = pd.Timedelta(days=21)     # ends >3wk before BT_END counts as an early exit


def intended_tickers() -> list[str]:
    """Extract the `stocks = [...]` literal from the notebook without executing it."""
    import json
    nb = json.load(open(NB_PATH, encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if src.lstrip().startswith("stocks = ["):
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if (isinstance(node, ast.Assign)
                        and any(getattr(t, "id", None) == "stocks" for t in node.targets)):
                    return [ast.literal_eval(e) for e in node.value.elts]
    raise SystemExit("could not find `stocks = [...]` in the notebook")


def main() -> None:
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"{CSV_PATH} not found -- run from the project root.")

    intended = intended_tickers()
    df = pd.read_csv(CSV_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
    present = [c[len("Close_"):] for c in df.columns if c.startswith("Close_")]
    present_set = set(present)

    missing = [t for t in intended if t not in present_set]

    # Per-ticker first/last valid Close date.
    rows = []
    for tk in present:
        s = df[f"Close_{tk}"].dropna()
        if s.empty:
            rows.append((tk, None, None))
            continue
        rows.append((tk, s.index.min(), s.index.max()))
    cov = pd.DataFrame(rows, columns=["ticker", "first", "last"]).set_index("ticker")

    late = cov[cov["first"] > BT_START + START_TOL].sort_values("first")
    early = cov[cov["last"] < BT_END - END_TOL].sort_values("last")
    full = cov[(cov["first"] <= BT_START + START_TOL) & (cov["last"] >= BT_END - END_TOL)]

    def _fmt(ts) -> str:
        return ts.date().isoformat() if ts is not None and pd.notna(ts) else "NA"

    print("=" * 72)
    print("UNIVERSE SURVIVORSHIP / COVERAGE AUDIT")
    print(f"  backtest window: {BT_START.date()} -> {BT_END.date()}")
    print("=" * 72)
    print(f"intended tickers (notebook list): {len(intended)}")
    print(f"present in {CSV_PATH}:             {len(present)}")
    print(f"full-window coverage:             {len(full)}")
    print()

    print(f"[1] requested-but-missing (yfinance returned nothing): {len(missing)}")
    print(f"    {sorted(missing) if missing else '(none)'}")
    print()

    print(f"[2] late entrants (first price > {(BT_START + START_TOL).date()}): {len(late)}")
    for tk, r in late.iterrows():
        print(f"    {tk:<8} first={_fmt(r['first'])}  last={_fmt(r['last'])}")
    print()

    print(f"[3] early exits (last price < {(BT_END - END_TOL).date()}): {len(early)}")
    for tk, r in early.iterrows():
        print(f"    {tk:<8} first={_fmt(r['first'])}  last={_fmt(r['last'])}")
    print()

    print("-" * 72)
    print("BOUND / CAVEAT:")
    print(f"  {len(late)}/{len(present)} names are late entrants -> the universe")
    print("  effectively grows over the window (breadth is smaller early on).")
    print("  This is measurable. The CLASSIC survivorship bias -- companies that")
    print("  DELISTED/merged 2018-2025 -- is NOT visible here: they were never in")
    print("  the current-snapshot list. Treat headline returns as optimistic by")
    print("  ~1-4%/yr until a point-in-time universe is built.")


if __name__ == "__main__":
    main()
