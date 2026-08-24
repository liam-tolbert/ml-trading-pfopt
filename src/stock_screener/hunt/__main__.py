"""Weekend-hunt CLI — every deterministic step, invocable without any AI.

    python -m src.stock_screener.hunt status
    python -m src.stock_screener.hunt candidates [--min-rs 70] [--date YYYY-MM-DD]
    python -m src.stock_screener.hunt charts [--limit N] [--per-fig 4] [--date ...]
    python -m src.stock_screener.hunt validate-verdicts [--date ...]
    python -m src.stock_screener.hunt gates [--min-fund 0] [--date ...]
    python -m src.stock_screener.hunt report [--min-fund 0] [--date ...]

Run from the repo root inside the ml-trading env (mamba run -n ml-trading ...).
State lands in data/cockpit/hunt/<date>/ — diagnostics.csv, meta.json,
charts/sheet_NNN.png, verdicts.csv (written by the reviewer), report.html.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# `python -m src.stock_screener.hunt` from the repo root already has ROOT on
# sys.path; this insert covers being launched from elsewhere (test style).
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.stock_screener.hunt import pipeline as pl


def _load_state(args):
    d = pl.hunt_dir(args.date)
    diag_path = d / "diagnostics.csv"
    if not diag_path.exists():
        raise pl.HuntError(f"{diag_path} not found — run `candidates` first.")
    return d, pd.read_csv(diag_path)


def cmd_status(args) -> int:
    print(json.dumps(pl.status(pl.load_scan()), indent=2, default=str))
    return 0


def cmd_candidates(args) -> int:
    bundle = pl.load_scan()
    cand = pl.candidates(bundle, min_rs=args.min_rs)
    diag = pl.diagnostics(bundle, cand)
    d = pl.hunt_dir(args.date)
    diag.to_csv(d / "diagnostics.csv", index=False)
    meta = pl.status(bundle)
    meta["min_rs"] = args.min_rs
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"hunt_dir": str(d), "candidates": len(diag),
                      "diagnostics": str(d / "diagnostics.csv")}, indent=2))
    return 0


def cmd_charts(args) -> int:
    from src.stock_screener.hunt.charts import render_sheets   # matplotlib import stays lazy
    bundle = pl.load_scan()
    d, diag = _load_state(args)
    if args.limit:
        diag = diag.head(args.limit)
    paths = render_sheets(bundle, diag, d / "charts", per_fig=args.per_fig)
    print(json.dumps({"sheets": [str(p) for p in paths],
                      "tickers": len(diag), "per_fig": args.per_fig}, indent=2))
    return 0


def cmd_validate(args) -> int:
    d, diag = _load_state(args)
    problems = pl.validate_verdicts(d / "verdicts.csv", diag)
    if problems:
        print(json.dumps({"ok": False, "problems": problems}, indent=2))
        return 1
    print(json.dumps({"ok": True, "verdicts": len(diag)}, indent=2))
    return 0


def cmd_gates(args) -> int:
    d, diag = _load_state(args)
    verdicts = pl.read_verdicts(d / "verdicts.csv")
    out = pl.gates(diag, verdicts, min_fund=args.min_fund)
    out["watchlist"] = pl.watchlist_audit(diag, verdicts)
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_report(args) -> int:
    from src.stock_screener.hunt.report import build_report
    d, _ = _load_state(args)
    out = build_report(d, min_fund=args.min_fund)
    print(json.dumps({"report": str(out)}, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hunt", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--date", default=None, help="hunt directory date (default: today)")

    sub.add_parser("status")
    p = sub.add_parser("candidates"); common(p)
    p.add_argument("--min-rs", type=int, default=pl.MIN_RS)
    p = sub.add_parser("charts"); common(p)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--per-fig", type=int, default=4)
    p = sub.add_parser("validate-verdicts"); common(p)
    p = sub.add_parser("gates"); common(p)
    p.add_argument("--min-fund", type=int, default=0)
    p = sub.add_parser("report"); common(p)
    p.add_argument("--min-fund", type=int, default=0)

    args = ap.parse_args(argv)
    try:
        return {"status": cmd_status, "candidates": cmd_candidates, "charts": cmd_charts,
                "validate-verdicts": cmd_validate, "gates": cmd_gates,
                "report": cmd_report}[args.cmd](args)
    except pl.HuntError as e:
        print(f"hunt: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
