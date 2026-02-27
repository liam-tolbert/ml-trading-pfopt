#!/usr/bin/env python3
"""
Portfolio Change Tracker for The All-Weather Portfolio (Q1 2026).

Fetches the published Google Sheet, saves timestamped snapshots,
and diffs against the previous snapshot to report changes:
  - New / closed positions
  - New lot additions and sales within the quarter
  - Share count changes
  - Price moves (top movers)

Usage:
    python track_portfolio.py              # fetch, diff, save
    python track_portfolio.py --holdings   # just print current holdings
    python track_portfolio.py --history    # list all saved snapshots
    python track_portfolio.py --no-save    # run without saving a snapshot
"""

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")  # Fix Windows console encoding


# =============================================================================
#  Configuration
# =============================================================================

SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQT7uecuE4ONP7z6L71E1y9F0mWp-Wbs6MrXpBtJ20toZwZhUuo0MVI36ahr1jpEqJJi1hXMKTnseRI"
    "/pub?gid=1391383294&single=true&output=csv"
)

SNAPSHOT_DIR = Path(__file__).parent / "data" / "portfolio_snapshots"

# Column names mapped to the published CSV (order matters)
CSV_COLUMNS = [
    "portfolio", "pct_total_assets", "pct_portfolio_assets",
    "company", "ticker", "tt_score",
    "shares", "starting_position", "current_position",
    "price_at_start", "day_chng_pct", "current_price",
    "sale_price_strike", "price_change", "pct_gain_loss", "gain_loss",
    "div_yield", "div_income",
    "active_ccd_info", "ccd_income", "active_collar_put_info",
    "sale_ticker", "sell_date", "sale_price", "sale_position_size",
    "add_ticker", "date_added", "buy_in_price", "add_position_size",
]

# Rows whose ticker or company matches these are summary/header rows to skip
SKIP_TICKERS = {"", "N/A", "Total", "Position Total", "Ticker"}
SKIP_COMPANY_PREFIXES = ("Total", "Position", "Cash", "Major", "The Prophet", "% of")

# Report widths
HOLDINGS_WIDTH = 103
REPORT_WIDTH = 70


# =============================================================================
#  Parsing Helpers
# =============================================================================

def parse_dollar(val) -> float:
    """Parse '$1,234,567' or '-$123' into a float. Returns 0.0 on failure."""
    if pd.isna(val) or val in ("", "$0"):
        return 0.0
    cleaned = str(val).replace("$", "").replace(",", "").replace('"', "").strip()
    try:
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0


def parse_percent(val) -> float | None:
    """Parse '▲ 9.48%' or '▼ -9.13%' into a float. Returns None on failure."""
    if pd.isna(val) or val == "":
        return None
    cleaned = str(val).replace("▲", "").replace("▼", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_number(val) -> float:
    """Parse '15,000' into 15000.0. Returns 0.0 on failure."""
    if pd.isna(val) or val == "":
        return 0.0
    try:
        return float(str(val).replace(",", "").strip())
    except ValueError:
        return 0.0


def clean_str(val) -> str:
    """Strip whitespace; convert pandas NaN to empty string."""
    s = str(val).strip()
    return "" if s == "nan" else s


def fmt_dollar(val) -> str:
    """Format a number as '$1,234' or '-$1,234'. Returns 'N/A' for None."""
    if val is None:
        return "N/A"
    return f"-${abs(val):,.0f}" if val < 0 else f"${val:,.0f}"


# =============================================================================
#  Fetching & Cleaning
# =============================================================================

def fetch_sheet() -> pd.DataFrame:
    """Download the Q1 2026 sheet as a DataFrame from the published CSV URL."""
    resp = requests.get(SHEET_CSV_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), header=0)
    df.columns = CSV_COLUMNS[:len(df.columns)]
    return df


def is_summary_row(ticker: str, company: str) -> bool:
    """Return True if this row is a total/header/summary row, not a real holding."""
    if ticker in SKIP_TICKERS:
        return True
    if any(company.startswith(prefix) for prefix in SKIP_COMPANY_PREFIXES):
        return True
    if ticker.startswith("$") or ticker.startswith("-$"):
        return True
    # Ticker is a bare number (e.g. a totals row)
    try:
        float(ticker.replace(",", ""))
        return True
    except ValueError:
        return False


def parse_row(row: pd.Series) -> dict | None:
    """Parse a single CSV row into a lot record dict, or None if it should be skipped."""
    company = clean_str(row.get("company"))
    ticker = clean_str(row.get("ticker"))
    if not ticker or not company:
        return None

    ticker = ticker.replace("*", "").strip()
    company = company.replace("*", "").strip()
    if is_summary_row(ticker, company):
        return None

    shares = parse_number(row.get("shares"))
    starting_pos = parse_dollar(row.get("starting_position"))
    current_pos = parse_dollar(row.get("current_position"))
    current_price = parse_dollar(row.get("current_price"))
    gain_loss = parse_dollar(row.get("gain_loss"))
    sale_price = parse_dollar(row.get("sale_price_strike"))
    is_sold = (current_pos == 0) and (sale_price > 0)

    return {
        "company":           company,
        "ticker":            ticker,
        "tt_score":          parse_number(row.get("tt_score")),
        "shares":            shares,
        "starting_position": starting_pos,
        "current_position":  current_pos,
        "price_at_start":    parse_dollar(row.get("price_at_start")),
        "current_price":     current_price,
        "gain_loss":         gain_loss,
        "pct_gain_loss":     parse_percent(row.get("pct_gain_loss")),
        "div_yield":         clean_str(row.get("div_yield")),
        "is_sold":           is_sold,
        "sale_price":        sale_price if is_sold else None,
        "sell_date":         clean_str(row.get("sell_date")) or None,
        "is_add":            bool(clean_str(row.get("add_ticker"))),
        "date_added":        clean_str(row.get("date_added")) or None,
        "buy_in_price":      parse_dollar(row.get("buy_in_price")),
    }


def clean_data(df: pd.DataFrame) -> list[dict]:
    """Parse the full DataFrame into a list of lot records, filtering out junk rows."""
    records = []
    for _, row in df.iterrows():
        record = parse_row(row)
        if record:
            records.append(record)
    return records


def aggregate_positions(records: list[dict]) -> dict:
    """
    Roll up individual lot records into per-ticker position summaries.

    Returns { ticker: { company, ticker, total_shares, total_current_value, ... , lots: [...] } }
    """
    positions: dict[str, dict] = {}

    for lot in records:
        ticker = lot["ticker"]

        if ticker not in positions:
            positions[ticker] = {
                "company":              lot["company"],
                "ticker":               ticker,
                "tt_score":             lot["tt_score"],
                "total_shares":         0,
                "total_current_value":  0,
                "total_starting_value": 0,
                "current_price":        lot["current_price"],
                "total_gain_loss":      0,
                "lots":                 [],
                "num_active_lots":      0,
                "num_sold_lots":        0,
            }

        position = positions[ticker]

        if lot["is_sold"]:
            position["num_sold_lots"] += 1
        else:
            position["total_shares"] += lot["shares"]
            position["total_current_value"] += lot["current_position"]
            position["num_active_lots"] += 1

        position["total_starting_value"] += lot["starting_position"]
        position["total_gain_loss"] += lot["gain_loss"]
        position["current_price"] = lot["current_price"]
        position["lots"].append(lot)

    return positions


# =============================================================================
#  Snapshots (save / load / list)
# =============================================================================

def save_snapshot(records: list[dict], positions: dict) -> Path:
    """Save the current portfolio state as a timestamped JSON file."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = SNAPSHOT_DIR / f"snapshot_{timestamp}.json"

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "records": records,
        "positions": positions,
    }
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    print(f"Snapshot saved: {path}")
    return path


def load_latest_snapshot() -> dict | None:
    """Load the most recent snapshot from disk, or None if none exist."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SNAPSHOT_DIR.glob("snapshot_*.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def list_snapshots():
    """Print a summary of all saved snapshots."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SNAPSHOT_DIR.glob("snapshot_*.json"))

    if not files:
        print("No snapshots saved yet.")
        return

    print(f"\n{'─' * 50}")
    print(f"  Saved snapshots ({len(files)} total)")
    print(f"{'─' * 50}")
    for filepath in files:
        with open(filepath) as f:
            data = json.load(f)
        active_count = sum(1 for p in data["positions"].values() if p["total_shares"] > 0)
        print(f"  {filepath.name}  ({active_count} active positions)")


# =============================================================================
#  Diffing — compare two snapshots
# =============================================================================

def compare_snapshots(old_positions: dict, new_positions: dict) -> dict:
    """
    Compare two position dicts and return a changes dict with keys:
      new_positions, removed_positions, share_changes,
      price_changes, value_changes, new_lots, new_sales
    """
    changes = {
        "new_positions":     [],
        "removed_positions": [],
        "share_changes":     [],
        "price_changes":     [],
        "value_changes":     [],
        "new_lots":          [],
        "new_sales":         [],
    }

    old_tickers = set(old_positions)
    new_tickers = set(new_positions)

    # --- Brand-new tickers ---
    for ticker in new_tickers - old_tickers:
        pos = new_positions[ticker]
        changes["new_positions"].append({
            "ticker":  ticker,
            "company": pos["company"],
            "shares":  pos["total_shares"],
            "value":   pos["total_current_value"],
        })

    # --- Tickers that disappeared ---
    for ticker in old_tickers - new_tickers:
        pos = old_positions[ticker]
        changes["removed_positions"].append({
            "ticker":      ticker,
            "company":     pos["company"],
            "last_shares": pos["total_shares"],
            "last_value":  pos["total_current_value"],
        })

    # --- Tickers present in both snapshots ---
    for ticker in old_tickers & new_tickers:
        old_pos = old_positions[ticker]
        new_pos = new_positions[ticker]

        # Share count change
        if old_pos["total_shares"] != new_pos["total_shares"]:
            changes["share_changes"].append({
                "ticker":     ticker,
                "company":    new_pos["company"],
                "old_shares": old_pos["total_shares"],
                "new_shares": new_pos["total_shares"],
                "change":     new_pos["total_shares"] - old_pos["total_shares"],
            })

        # Price change
        old_price = old_pos["current_price"]
        new_price = new_pos["current_price"]
        if old_price and new_price and old_price != new_price:
            changes["price_changes"].append({
                "ticker":     ticker,
                "company":    new_pos["company"],
                "old_price":  old_price,
                "new_price":  new_price,
                "change":     new_price - old_price,
                "pct_change": (new_price - old_price) / old_price * 100,
            })

        # Value change
        if old_pos["total_current_value"] != new_pos["total_current_value"]:
            changes["value_changes"].append({
                "ticker":    ticker,
                "company":   new_pos["company"],
                "old_value": old_pos["total_current_value"],
                "new_value": new_pos["total_current_value"],
                "change":    new_pos["total_current_value"] - old_pos["total_current_value"],
            })

        # New lot additions (buy lots that weren't in the old snapshot)
        old_add_keys = {
            (lot.get("date_added"), lot.get("shares"))
            for lot in old_pos["lots"] if lot.get("is_add")
        }
        for lot in new_pos["lots"]:
            if lot.get("is_add") and (lot["date_added"], lot["shares"]) not in old_add_keys:
                changes["new_lots"].append({
                    "ticker":       ticker,
                    "company":      new_pos["company"],
                    "shares":       lot["shares"],
                    "date_added":   lot["date_added"],
                    "buy_in_price": lot["buy_in_price"],
                })

        # New sales (sold lots that weren't in the old snapshot)
        old_sell_keys = {
            (lot.get("sell_date"), lot.get("shares"))
            for lot in old_pos["lots"] if lot.get("is_sold")
        }
        for lot in new_pos["lots"]:
            if lot.get("is_sold") and (lot["sell_date"], lot["shares"]) not in old_sell_keys:
                changes["new_sales"].append({
                    "ticker":     ticker,
                    "company":    new_pos["company"],
                    "shares":     lot["shares"],
                    "sell_date":  lot["sell_date"],
                    "sale_price": lot["sale_price"],
                })

    return changes


# =============================================================================
#  Reporting — print holdings table & change report
# =============================================================================

def print_holdings(positions: dict):
    """Print the full holdings table sorted by current value."""
    total_value = sum(
        p["total_current_value"]
        for p in positions.values()
        if p["total_shares"] > 0
    )

    w = HOLDINGS_WIDTH
    print("\n" + "=" * w)
    print(f"  ALL-WEATHER PORTFOLIO - Q1 2026  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * w)
    print(
        f"  {'Ticker':<8} {'Company':<22} {'Shares':>10} {'Price':>10} "
        f"{'Value':>14} {'Gain/Loss':>14} {'%':>8} {'% Port':>7}"
    )
    print("─" * w)

    total_gl = 0
    sorted_positions = sorted(
        positions.values(),
        key=lambda p: p["total_current_value"],
        reverse=True,
    )

    for pos in sorted_positions:
        if pos["total_shares"] <= 0:
            continue

        total_gl += pos["total_gain_loss"]
        gain_pct = (
            pos["total_gain_loss"] / pos["total_starting_value"] * 100
            if pos["total_starting_value"] else 0
        )
        port_pct = pos["total_current_value"] / total_value * 100 if total_value else 0

        print(
            f"  {pos['ticker']:<8} {pos['company']:<22} {pos['total_shares']:>10,.0f} "
            f"${pos['current_price']:>8,.2f} {fmt_dollar(pos['total_current_value']):>14} "
            f"{fmt_dollar(pos['total_gain_loss']):>14} {gain_pct:>+7.1f}% {port_pct:>6.1f}%"
        )

    print("─" * w)
    total_gain_pct = total_gl / (total_value - total_gl) * 100 if (total_value - total_gl) else 0
    print(
        f"  {'TOTAL':<8} {'':<22} {'':>10} {'':>10} "
        f"{fmt_dollar(total_value):>14} {fmt_dollar(total_gl):>14} "
        f"{total_gain_pct:>+7.1f}% {'100.0%':>7}"
    )
    print("=" * w)


def print_change_report(changes: dict, positions: dict):
    """Print the diff report between the previous snapshot and current data."""
    total_value = sum(p["total_current_value"] for p in positions.values())
    total_gl = sum(p["total_gain_loss"] for p in positions.values())
    active_count = sum(1 for p in positions.values() if p["total_shares"] > 0)

    w = REPORT_WIDTH
    lines = [
        "=" * w,
        f"  CHANGE REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * w,
        f"\n  Portfolio Value:  {fmt_dollar(total_value)}",
        f"  Total Gain/Loss: {fmt_dollar(total_gl)}",
        f"  Active Tickers:  {active_count}",
    ]

    has_changes = any(changes[key] for key in changes)
    if not has_changes:
        lines.append("\n  No changes detected since last snapshot.")
        lines.append("=" * w)
        print("\n".join(lines))
        return

    def pct_of_portfolio(value):
        return value / total_value * 100 if total_value else 0

    # — New positions —
    if changes["new_positions"]:
        lines += [f"\n{'─' * w}", "  NEW POSITIONS", f"{'─' * w}"]
        for chg in changes["new_positions"]:
            pct = pct_of_portfolio(chg["value"])
            lines.append(
                f"  + {chg['ticker']:8s} {chg['company']:20s} "
                f"{chg['shares']:>10,.0f} shares  {fmt_dollar(chg['value']):>12s}"
                f"  ({pct:.1f}% of portfolio)"
            )

    # — Closed positions —
    if changes["removed_positions"]:
        lines += [f"\n{'─' * w}", "  CLOSED POSITIONS", f"{'─' * w}"]
        for chg in changes["removed_positions"]:
            pct = pct_of_portfolio(chg["last_value"])
            lines.append(
                f"  - {chg['ticker']:8s} {chg['company']:20s} "
                f"{chg['last_shares']:>10,.0f} shares  {fmt_dollar(chg['last_value']):>12s}"
                f"  ({pct:.1f}% of portfolio)"
            )

    # — New lot additions —
    if changes["new_lots"]:
        lines += [f"\n{'─' * w}", "  NEW ADDITIONS (lots)", f"{'─' * w}"]
        for chg in changes["new_lots"]:
            lot_value = chg["shares"] * chg["buy_in_price"]
            pct = pct_of_portfolio(lot_value)
            lines.append(
                f"  + {chg['ticker']:8s} {chg['company']:20s} "
                f"{chg['shares']:>10,.0f} shares @ ${chg['buy_in_price']:,.2f}  "
                f"{fmt_dollar(lot_value):>12s}  ({pct:.1f}%)  added {chg['date_added']}"
            )

    # — New sales —
    if changes["new_sales"]:
        lines += [f"\n{'─' * w}", "  NEW SALES", f"{'─' * w}"]
        for chg in changes["new_sales"]:
            sale_value = chg["shares"] * chg["sale_price"]
            pct = pct_of_portfolio(sale_value)
            lines.append(
                f"  - {chg['ticker']:8s} {chg['company']:20s} "
                f"{chg['shares']:>10,.0f} shares @ ${chg['sale_price']:,.2f}  "
                f"{fmt_dollar(sale_value):>12s}  ({pct:.1f}%)  sold {chg['sell_date']}"
            )

    # — Share count changes —
    if changes["share_changes"]:
        lines += [f"\n{'─' * w}", "  SHARE COUNT CHANGES", f"{'─' * w}"]
        sorted_by_magnitude = sorted(
            changes["share_changes"],
            key=lambda c: abs(c["change"]),
            reverse=True,
        )
        for chg in sorted_by_magnitude:
            sign = "+" if chg["change"] > 0 else ""
            lines.append(
                f"  {chg['ticker']:8s} {chg['company']:20s} "
                f"{chg['old_shares']:>10,.0f} -> {chg['new_shares']:>10,.0f}  "
                f"({sign}{chg['change']:,.0f})"
            )

    # — Price changes (top 15 movers) —
    if changes["price_changes"]:
        lines += [f"\n{'─' * w}", "  PRICE CHANGES (top movers)", f"{'─' * w}"]
        top_movers = sorted(
            changes["price_changes"],
            key=lambda c: abs(c["pct_change"]),
            reverse=True,
        )[:15]
        for chg in top_movers:
            arrow = "▲" if chg["pct_change"] > 0 else "▼"
            lines.append(
                f"  {chg['ticker']:8s} {chg['company']:20s} "
                f"${chg['old_price']:>8,.2f} -> ${chg['new_price']:>8,.2f}  "
                f"{arrow} {chg['pct_change']:+.2f}%"
            )

    lines.append("\n" + "=" * w)
    print("\n".join(lines))


# =============================================================================
#  Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Track All-Weather Portfolio changes (Q1 2026)"
    )
    parser.add_argument("--holdings", action="store_true", help="Print current holdings only")
    parser.add_argument("--history",  action="store_true", help="List saved snapshots")
    parser.add_argument("--no-save",  action="store_true", help="Don't save snapshot")
    args = parser.parse_args()

    if args.history:
        list_snapshots()
        return

    # Fetch and parse
    print("Fetching Q1 2026 portfolio data...")
    df = fetch_sheet()
    records = clean_data(df)
    positions = aggregate_positions(records)
    print(f"Parsed {len(records)} lot entries across {len(positions)} tickers.")

    print_holdings(positions)

    if args.holdings:
        return

    # Diff against previous snapshot
    previous = load_latest_snapshot()
    if previous:
        changes = compare_snapshots(previous["positions"], positions)
        print_change_report(changes, positions)
    else:
        print("\nFirst run — saving baseline snapshot. Run again after the sheet updates to see changes.")

    if not args.no_save:
        save_snapshot(records, positions)


if __name__ == "__main__":
    main()