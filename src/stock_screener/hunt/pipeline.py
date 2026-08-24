"""Deterministic core of the weekend hunt.

Reads the cockpit's last completed scan (READ-ONLY — scan freshness is
scan_worker's job; Sat 10:30 ET feeds this) and turns it into the reviewable
state the /weekend-hunt skill works from: candidate diagnostics, verdict
bookkeeping, and the mechanical gates.

The entry rules live here as named constants with their sources, because this
is exactly what got fumbled when the process was ad hoc:

- Buy zone is pivot to +5% (scan.py ``buy_zone`` — "no chasing > +5%"; SEPA doc
  "Entry within 5% of pivot"). The +10% in vcp.BUY_ZONE_PCT is only the Tier-A
  *screening* tolerance and is never an entry bound.
- The RS floor is 70 (Step-1 checklist; the app's Min-RS default).
"""
from __future__ import annotations

import csv
import datetime as _dt
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.stock_screener.cockpit.cache import CACHE_DIR, WATCHLIST_JSON

# ---- rules (sources in the module docstring) ------------------------------ #
MIN_RS = 70                    # Step-1 floor / app Min-RS default
BUY_ZONE_MAX_PCT = 5.0         # pivot .. +5% = the entry range (scan.py buy_zone)
APPROACH_MIN_PCT = -3.0        # within 3% below pivot = "approaching"
VOL_CONFIRM_RATIO = 1.4        # close above pivot on >=40% above-average volume
EARNINGS_BLOCK_DAYS = 21       # no entry with earnings inside ~3 weeks (doc Step-4)
MAX_SCAN_AGE_DAYS = 3.0        # hunt must run off a weekend-fresh scan

_LAST_SCAN_PKL = CACHE_DIR / "last_scan.pkl"
_PERSIST_VERSION = 1           # scan_worker._PERSIST_VERSION — bump together
HUNT_DIR = CACHE_DIR / "hunt"

VERDICTS = ("PASS", "PASS-", "FAIL")


class HuntError(RuntimeError):
    """Pipeline preconditions failed; message says what to do about it."""


@dataclass
class ScanBundle:
    result: object            # ScanResult (candidates / payloads / regime / ...)
    completed_wall: float
    key: tuple


# --------------------------------------------------------------------------- #
def load_scan(path: Optional[Path] = None) -> ScanBundle:
    """Load the persisted last scan, refusing stale/missing/foreign pickles."""
    path = Path(path) if path else _LAST_SCAN_PKL
    if not path.exists():
        raise HuntError(f"{path} not found — run a scan from the cockpit first "
                        "(the Sat 10:30 scheduled refresh normally provides it).")
    with open(path, "rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict) or d.get("version") != _PERSIST_VERSION or d.get("result") is None:
        raise HuntError(f"{path} has an unexpected shape/version — re-run the scan.")
    bundle = ScanBundle(d["result"], float(d.get("completed_wall") or 0.0),
                        tuple(d.get("key") or ()))
    age_days = (_dt.datetime.now().timestamp() - bundle.completed_wall) / 86400.0
    if age_days > MAX_SCAN_AGE_DAYS:
        raise HuntError(f"scan is {age_days:.1f} days old (> {MAX_SCAN_AGE_DAYS:.0f}) — "
                        "refresh it from the cockpit before hunting.")
    return bundle


def status(bundle: ScanBundle) -> dict:
    cand = bundle.result.candidates
    n_tier_a = int((cand["tier"] == "A").sum()) if len(cand) else 0
    n_eligible = int(((cand["tier"] == "A") & (cand["rs"] >= MIN_RS)).sum()) if len(cand) else 0
    return {
        "scan_time": _dt.datetime.fromtimestamp(bundle.completed_wall).isoformat(sep=" "),
        "key": list(bundle.key),
        "regime": bundle.result.regime,
        "n_scanned": bundle.result.n_scanned,
        "n_passed_template": bundle.result.n_passed,
        "n_tier_a": n_tier_a,
        "n_eligible": n_eligible,      # tier A AND rs >= MIN_RS — the review set
        "min_rs": MIN_RS,
    }


def candidates(bundle: ScanBundle, min_rs: int = MIN_RS) -> pd.DataFrame:
    """Tier A ∩ RS floor, in scan order (tier, quality, fund, rs)."""
    cand = bundle.result.candidates
    out = cand[(cand["tier"] == "A") & (cand["rs"] >= min_rs)].reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
def _watchlist_tickers() -> List[str]:
    import json
    try:
        entries = json.loads(WATCHLIST_JSON.read_text())
        return [e["ticker"] for e in entries if isinstance(e, dict) and e.get("ticker")]
    except (OSError, ValueError):
        return []


def diagnostics(bundle: ScanBundle, cand: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate: everything the review + gates + report need.

    Purely derived from the scan payloads — nothing here refetches prices.
    """
    wl = set(_watchlist_tickers())
    rows = []
    for rank, (_, c) in enumerate(cand.iterrows(), start=1):
        t = c["ticker"]
        p = bundle.result.payloads[t]
        df, lev, v = p["df"], p["levels"], p["vcp"]
        close = df["Close"].to_numpy()
        vol = df["Volume"].to_numpy()
        piv = float(lev["pivot"])
        vs_pivot = (float(close[-1]) / piv - 1.0) * 100.0

        vs50 = pd.Series(vol).rolling(50).mean().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            vr = vol[-25:] / np.where(vs50[-25:] > 0, vs50[-25:], np.nan)
        ret1 = np.diff(np.log(close))[-25:]
        dist_days = int(np.nansum((ret1 < -0.002) & (vr > 1.25)))

        opens = df["Open"].to_numpy()
        gaps = np.abs(opens[1:] / close[:-1] - 1.0)
        gaps8 = int((gaps[-120:] > 0.08).sum())

        depths = [cn["drawdown_pct"] for cn in v["contractions"]]
        s2 = p.get("step2") or {}
        checks = s2.get("checks") or {}
        fu = p.get("fundamentals") or {}
        rows.append({
            "rank": rank, "ticker": t, "wl": int(t in wl),
            "q": float(c["vcp_quality"]), "rs": int(c["rs"]),
            "fund": int(c["fund_score"]),
            "close": round(float(close[-1]), 2), "pivot": round(piv, 2),
            "stop": round(float(lev["stop"]), 2),
            "vs_pivot_pct": round(vs_pivot, 2),
            "adv_musd": round(float((df["Close"] * df["Volume"]).tail(20).mean()) / 1e6, 2),
            "dist_days": dist_days, "gaps8": gaps8,
            "depths": "->".join(f"{d:.0f}" for d in depths),
            "n_legs": len(depths),
            "earnings_in": p.get("earnings_in"),
            "breakout_today": int(bool(lev.get("breakout_today"))),
            "volume_ratio": round(float(lev.get("volume_ratio") or 0.0), 2),
            # step-2 detail (booleans as 0/1 so the CSV round-trips cleanly)
            "f_rev": int(bool(checks.get("revenue_growth"))),
            "f_eps": int(bool(checks.get("eps_growth"))),
            "f_accel": int(bool(checks.get("eps_accelerating"))),
            "f_margin": int(bool(checks.get("margin_expanding"))),
            "rev_yoy": fu.get("revenue_yoy"), "eps_yoy": fu.get("eps_yoy"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def bucket(vs_pivot_pct: float) -> str:
    """Position vs the ENTRY rules (not the Tier-A screening tolerance)."""
    if vs_pivot_pct > BUY_ZONE_MAX_PCT:
        return "past_entry"
    if vs_pivot_pct >= 0.0:
        return "buy_zone"
    if vs_pivot_pct >= APPROACH_MIN_PCT:
        return "approaching"
    return "below"


def _earnings_blocked(earnings_in) -> bool:
    try:
        return 0 <= int(float(earnings_in)) <= EARNINGS_BLOCK_DAYS
    except (TypeError, ValueError):
        return False


def gates(diag: pd.DataFrame, verdicts: Dict[str, dict], min_fund: int = 0) -> dict:
    """Mechanical gates over the PASS names. min_fund is a *parameter* the user
    chooses per run — never silently applied to the review itself."""
    out = {"buy_zone": [], "approaching": [], "below": [], "past_entry": [],
           "earnings_blocked": [], "volume_confirmed": [], "min_fund": min_fund}
    for _, r in diag.iterrows():
        v = verdicts.get(r["ticker"])
        if not v or v["verdict"] != "PASS":
            continue
        if int(r["fund"]) < min_fund:
            continue
        row = {"ticker": r["ticker"], "vs_pivot_pct": float(r["vs_pivot_pct"]),
               "fund": int(r["fund"]), "q": float(r["q"]), "rs": int(r["rs"])}
        if _earnings_blocked(r["earnings_in"]):
            out["earnings_blocked"].append({**row, "earnings_in": int(float(r["earnings_in"]))})
            continue                      # blocked names appear nowhere else
        out[bucket(row["vs_pivot_pct"])].append(row)
        if bool(r["breakout_today"]) and float(r["volume_ratio"]) >= VOL_CONFIRM_RATIO:
            out["volume_confirmed"].append({**row, "volume_ratio": float(r["volume_ratio"])})
    for k in ("buy_zone", "approaching", "below", "past_entry"):
        out[k].sort(key=lambda x: -x["q"])
    return out


def watchlist_audit(diag: pd.DataFrame, verdicts: Dict[str, dict]) -> List[dict]:
    """Status card per pinned name — including the not-Tier-A / sub-RS states."""
    by_t = {r["ticker"]: r for _, r in diag.iterrows()}
    cards = []
    for t in _watchlist_tickers():
        r = by_t.get(t)
        if r is None:
            cards.append({"ticker": t, "state": "not_eligible",
                          "note": "not Tier A with RS >= %d in this scan" % MIN_RS})
        else:
            v = verdicts.get(t) or {}
            cards.append({"ticker": t, "state": v.get("verdict", "unreviewed"),
                          "note": v.get("notes", ""),
                          "vs_pivot_pct": float(r["vs_pivot_pct"])})
    return cards


# ---- verdict bookkeeping (crash-safe: append batches, validate at the end) - #
def hunt_dir(date: Optional[str] = None) -> Path:
    d = HUNT_DIR / (date or _dt.date.today().isoformat())
    d.mkdir(parents=True, exist_ok=True)
    return d


def append_verdicts(path: Path, rows: List[dict]) -> int:
    """Append a reviewed batch. rows: {ticker, verdict, notes}."""
    for r in rows:
        if r.get("verdict") not in VERDICTS:
            raise HuntError(f"bad verdict {r.get('verdict')!r} for {r.get('ticker')!r} "
                            f"(must be one of {VERDICTS})")
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "verdict", "notes"])
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ("ticker", "verdict", "notes")})
    return len(rows)


def read_verdicts(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["ticker"]: r for r in csv.DictReader(f)}


def validate_verdicts(path: Path, diag: pd.DataFrame) -> List[str]:
    """Every candidate exactly once; no strays; no bad labels. [] == clean."""
    problems: List[str] = []
    seen: Dict[str, int] = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                seen[r["ticker"]] = seen.get(r["ticker"], 0) + 1
                if r["verdict"] not in VERDICTS:
                    problems.append(f"{r['ticker']}: bad verdict {r['verdict']!r}")
    expected = set(diag["ticker"])
    for t in sorted(expected - set(seen)):
        problems.append(f"{t}: missing verdict")
    for t in sorted(set(seen) - expected):
        problems.append(f"{t}: verdict for a non-candidate")
    for t, n in sorted(seen.items()):
        if n > 1:
            problems.append(f"{t}: {n} verdicts (want 1)")
    return problems
