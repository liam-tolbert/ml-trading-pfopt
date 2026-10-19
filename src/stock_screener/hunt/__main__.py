"""Weekend-hunt CLI — every deterministic step, invocable without any AI.

    python -m src.stock_screener.hunt status
    python -m src.stock_screener.hunt candidates [--min-rs 70] [--date YYYY-MM-DD]
    python -m src.stock_screener.hunt charts [--limit N] [--per-fig 4] [--date ...]
    python -m src.stock_screener.hunt review-charts [--limit N] [--date ...]
    python -m src.stock_screener.hunt review [--reviewer rules|claude] [--model M] [--effort E]
                                             [--votes 3] [--jobs 4]
    python -m src.stock_screener.hunt label-template [--sample N] [--sample-b N] [--seed S]
                                                     [--date ...]
    python -m src.stock_screener.hunt score --labels MINE.csv [--against THEIRS.csv] [--date ...]
    python -m src.stock_screener.hunt compare --date A --against B
    python -m src.stock_screener.hunt append-verdicts --file BATCH.csv [--date ...]
    python -m src.stock_screener.hunt validate-verdicts [--date ...]
    python -m src.stock_screener.hunt gates [--min-fund 0] [--date ...]
    python -m src.stock_screener.hunt news [--limit 12] [--date ...]
    python -m src.stock_screener.hunt report [--min-fund 0] [--date ...]

Run from the repo root inside the ml-trading env; scripts/hunt/hunt.sh does both.
State lands in data/cockpit/hunt/<date>/ — diagnostics.csv, meta.json,
charts/sheet_NNN.png, review/ (each ticker's two review charts and facts, every answer
in votes.jsonl, the combined answers in form.csv), news/<ticker>.json, verdicts.csv,
narrative.md and catalyst.json (the last two written by the skill), report.html,
labels.csv and label_charts/ (the hand-label sample and its tier-B sheets).
`report` also mirrors report.html and the sheets to docs/hunt/<date>/, the deliverable.

`review` answers each chart's form from the detector's numbers. With `--reviewer claude`
it calls the Claude Code CLI once per answer instead, from the images `review-charts`
drew: HUNT_CLAUDE names the executable when `claude` is not on PATH.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# `python -m src.stock_screener.hunt` from the repo root already has ROOT on sys.path.
# The insert covers a launch from anywhere else.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.stock_screener.hunt import pipeline as pl
from src.stock_screener.hunt import reviewer as rv

LABEL_CHARTS = "label_charts"     # the tier-B label sample's sheets


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
    equity = pl.load_equity()
    diag = pl.diagnostics(bundle, cand, equity=equity)
    d = pl.hunt_dir(args.date)
    diag.to_csv(d / "diagnostics.csv", index=False)
    meta = pl.status(bundle)
    meta["min_rs"] = args.min_rs
    meta["equity"] = equity
    meta["full_position_usd"] = round(pl.FULL_POSITION_PCT * equity) if equity else None
    if equity:
        meta["equity_as_of"] = json.loads(pl.ACCOUNT_JSON.read_text(encoding="utf-8"))["as_of"]
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


def cmd_review_charts(args) -> int:
    from src.stock_screener.hunt.charts import RENDER_VERSION, render_review_images
    bundle = pl.load_scan()
    d, diag = _load_state(args)
    if args.limit:
        diag = diag.head(args.limit)
    done = render_review_images(bundle, diag, d)
    _update_meta(d, {"review_render_version": RENDER_VERSION})
    print(json.dumps({"tickers": len(done), "dir": str(d / rv.REVIEW_SUBDIR)}, indent=2))
    return 0


def cmd_review(args) -> int:
    d, diag = _load_state(args)
    if args.reviewer == "rules":
        payloads = pl.load_scan().result.payloads
        _update_meta(d, {"review": {"reviewer": "rules", "rules": rv.rules_hash(),
                                    "table_version": rv.TABLE_VERSION}})
        out = rv.run_rules_review(d, diag, payloads, log=lambda m: print(m, file=sys.stderr))
        _update_meta(d, {"review_models_used": [rv.RULES_MODEL], "review_cost_usd": 0.0})
        print(json.dumps(out, indent=2))
        return 0
    claude = os.environ.get("HUNT_CLAUDE") or shutil.which("claude")
    if not claude:
        raise pl.HuntError("claude CLI not found: put it on PATH or set HUNT_CLAUDE.")
    try:
        frames = {t: p["df"] for t, p in pl.load_scan().result.payloads.items()}
    except pl.HuntError:
        frames = None                     # Claude's own swing points go unmeasured
    call = rv.claude_cli(claude, model=args.model, effort=args.effort,
                         cwd=d / rv.REVIEW_SUBDIR)
    _update_meta(d, {"review": {"reviewer": "claude", "model": args.model,
                                "effort": args.effort,
                                "votes": args.votes, "prompt": rv.prompt_hash(),
                                "table_version": rv.TABLE_VERSION}})
    out = rv.run_review(d, diag, call, votes=args.votes, jobs=args.jobs, frames=frames,
                        log=lambda m: print(m, file=sys.stderr))
    used = sorted({v.get("model") for v in _vote_records(d) if v.get("model")})
    cost = sum(v.get("cost_usd") or 0.0 for v in _vote_records(d))
    _update_meta(d, {"review_models_used": used, "review_cost_usd": round(cost, 2)})
    print(json.dumps(out, indent=2))
    return 1 if out["no_answer"] else 0


def _vote_records(d: Path):
    path = d / rv.REVIEW_SUBDIR / rv.VOTES_FILE
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _update_meta(d: Path, fields: dict) -> None:
    path = d / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    meta.update(fields)
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def cmd_label_template(args) -> int:
    d, diag = _load_state(args)
    out = d / "labels.csv"
    if out.exists():
        raise pl.HuntError(f"{out} already exists; it is never overwritten.")
    bundle, cand_b = None, pd.DataFrame(columns=["ticker"])
    if args.sample_b > 0:
        bundle = pl.load_scan()
        meta_path = d / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        cand_b = pl.candidates(bundle, min_rs=meta.get("min_rs") or pl.MIN_RS, tier="B")
    rows = rv.label_rows(list(diag["ticker"]), list(cand_b["ticker"]), args.sample,
                         args.sample_b, args.seed or d.name)
    picked_b = {t for t, tier in rows if tier == "B"}
    summary = {"labels": str(out), "tickers": len(rows),
               "tier_a": len(rows) - len(picked_b), "tier_b": len(picked_b)}
    if picked_b:
        from src.stock_screener.hunt.charts import render_sheets
        diag_b = pl.diagnostics(bundle, cand_b[cand_b["ticker"].isin(picked_b)],
                                equity=pl.load_equity())
        # Its own folder: render_sheets numbers from sheet_001 and would overwrite charts/.
        render_sheets(bundle, diag_b, d / LABEL_CHARTS)
        summary["tier_b_sheets"] = str(d / LABEL_CHARTS)
    rv.write_label_template(out, rows)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_score(args) -> int:
    d = pl.hunt_dir(args.date)
    theirs = Path(args.against) if args.against else d / rv.REVIEW_SUBDIR / rv.FORM_FILE
    print(json.dumps(rv.score(rv.read_labels(Path(args.labels)), rv.read_labels(theirs)),
                     indent=2))
    return 0


def cmd_compare(args) -> int:
    a = pl.read_verdicts(pl.HUNT_DIR / args.date / "verdicts.csv")
    b = pl.read_verdicts(pl.HUNT_DIR / args.against / "verdicts.csv")
    if not a or not b:
        raise pl.HuntError("both hunt folders need a verdicts.csv.")
    print(json.dumps(pl.compare_verdicts(a, b), indent=2))
    return 0


def cmd_append(args) -> int:
    d, diag = _load_state(args)
    n = pl.append_batch(d / "verdicts.csv", Path(args.file), diag)
    have = pl.read_verdicts(d / "verdicts.csv")
    print(json.dumps({"appended": n, "verdicts": len(have),
                      "remaining": len(set(diag["ticker"]) - set(have))}, indent=2))
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


def cmd_news(args) -> int:
    from src.stock_screener.hunt.news import fetch_news, write_news
    d, diag = _load_state(args)
    verdicts = pl.read_verdicts(d / "verdicts.csv")
    tickers = [t for t in diag["ticker"] if (verdicts.get(t) or {}).get("verdict") == "PASS"]
    if not tickers:
        raise pl.HuntError("no PASS verdicts yet — review the sheets first.")
    news = fetch_news(tickers, limit=args.limit)
    paths = write_news(d, news)
    print(json.dumps({"tickers": len(paths), "dir": str(d / "news"),
                      "headlines": sum(len(n["headlines"]) for n in news.values()),
                      "without_headlines": [t for t, n in news.items() if not n["headlines"]],
                      "feed_errors": sum(len(n["errors"]) for n in news.values())}, indent=2))
    return 0


def cmd_report(args) -> int:
    from src.stock_screener.hunt.report import DOCS_DIR, build_report, mirror_report
    d, _ = _load_state(args)
    out = build_report(d, min_fund=args.min_fund)
    docs = mirror_report(d, DOCS_DIR / d.name)
    print(json.dumps({"report": str(out), "docs": str(docs)}, indent=2))
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
    p = sub.add_parser("review-charts"); common(p)
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("review"); common(p)
    p.add_argument("--reviewer", choices=("rules", "claude"), default="rules",
                   help="who answers each chart's form (default: rules, from the numbers)")
    p.add_argument("--model", default=rv.REVIEW_MODEL)
    p.add_argument("--effort", default=rv.REVIEW_EFFORT)
    p.add_argument("--votes", type=int, default=rv.REVIEW_VOTES)
    p.add_argument("--jobs", type=int, default=4, help="calls running at once")
    p = sub.add_parser("label-template"); common(p)
    p.add_argument("--sample", type=int, default=0,
                   help="label only this many candidates, picked at random (default: all)")
    p.add_argument("--sample-b", type=int, default=0,
                   help="also label this many tier-B names, with sheets in label_charts/ "
                        "(needs the hunt's scan; default: none)")
    p.add_argument("--seed", default=None,
                   help="what picks the sample (default: the hunt's date)")
    p = sub.add_parser("score"); common(p)
    p.add_argument("--labels", required=True, help="your answers (from label-template)")
    p.add_argument("--against", default=None,
                   help="answers to score (default: the review's form.csv)")
    p = sub.add_parser("compare")
    p.add_argument("--date", required=True)
    p.add_argument("--against", required=True)
    p = sub.add_parser("append-verdicts"); common(p)
    p.add_argument("--file", required=True, help="CSV with the header ticker,verdict,notes")
    p = sub.add_parser("validate-verdicts"); common(p)
    p = sub.add_parser("gates"); common(p)
    p.add_argument("--min-fund", type=int, default=0)
    p = sub.add_parser("news"); common(p)
    p.add_argument("--limit", type=int, default=12, help="headlines kept per ticker")
    p = sub.add_parser("report"); common(p)
    p.add_argument("--min-fund", type=int, default=0)

    args = ap.parse_args(argv)
    try:
        return {"status": cmd_status, "candidates": cmd_candidates, "charts": cmd_charts,
                "review-charts": cmd_review_charts, "review": cmd_review,
                "label-template": cmd_label_template, "score": cmd_score,
                "compare": cmd_compare,
                "append-verdicts": cmd_append, "validate-verdicts": cmd_validate,
                "gates": cmd_gates, "news": cmd_news, "report": cmd_report}[args.cmd](args)
    except pl.HuntError as e:
        print(f"hunt: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
