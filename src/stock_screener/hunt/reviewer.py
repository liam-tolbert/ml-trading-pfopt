"""Per-ticker chart review: Claude answers a fixed form, code decides the verdict.

For each candidate Claude sees two charts (``charts.render_review_images``: weekly for
the run, daily for the base) and the base's measurements (``fact_sheet``), and answers
``FORM``: eight closed questions about the base. It never writes a verdict.
``verdict_from_form`` turns the answers, plus the book rules that are plain numbers,
into PASS / PASS- / FAIL and the codes in ``pipeline.NOTE_CODES``.

Each ticker is asked twice in separate calls, and up to ``votes`` times when the first
two answers differ on any question. Each question takes the majority answer. A ticker a
measurement already fails (``measured_fail``) is not asked. ``needs_check`` lists what
sends a name to the user: the votes' verdicts differ, a vote was unsure, or an answer
contradicts the measurements.

The reviewer MUST NOT see RS, F, the quality score, the distance to the pivot, days to
earnings or dollar volume. The gates apply those, and what the reviewer can't see it
can't weigh. The ticker is hidden for the same reason.

Plain stdlib + numpy + pandas: the deploy gate runs the tests of this module on the Pi.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.stock_screener.cockpit.doctrine import VOL_AVG_DAYS
from src.stock_screener.cockpit.indicators import prior_volume_average

from . import pipeline as pl

# The model and effort every weekly run uses until the user changes them on purpose.
REVIEW_MODEL = "claude-opus-5-5"
REVIEW_EFFORT = "medium"
REVIEW_VOTES = 3
# Raised by one whenever the verdict table in verdict_from_form changes meaning.
TABLE_VERSION = 1
# A pullback counts as wider only when it is more than this many times as deep as the one
# before. Halving is the books' rule of thumb, not a disqualifier (HANDOFF §6.94).
EXPAND_RATIO = 1.10

REVIEW_SUBDIR = "review"
VOTES_FILE = "votes.jsonl"
FORM_FILE = "form.csv"

# Each question's answers, mildest first. The prompt explains each one.
FORM: Dict[str, Tuple[str, ...]] = {
    "pattern": ("vcp", "cup_with_handle", "flat_base", "double_bottom", "high_tight_flag",
                "no_base"),
    "swings_match": ("yes", "no"),
    "right_side": ("orderly_tightening", "wider_leg", "v_shape"),
    "final_area": ("tight", "loose", "not_formed"),
    "volume": ("drying_up", "mixed", "heavy_selling"),
    "pivot": ("clear", "several", "none"),
    "character": ("quick_recoveries", "grinding", "wild_swings"),
    "base_count": ("first_or_second", "third", "fourth_or_later", "cant_tell"),
}

# When the votes split three ways, the answer that weighs most against the base wins, so
# a split never produces a milder verdict than one of its votes.
_SEVERITY = {
    "no_base": 2, "no": 1, "wider_leg": 2, "v_shape": 2, "loose": 1, "not_formed": 1,
    "mixed": 1, "heavy_selling": 2, "several": 1, "none": 1, "grinding": 1,
    "wild_swings": 2, "fourth_or_later": 1,
}

# A down day on heavy volume: the same definition as the diagnostics' dist_days.
_DOWN_RET = -0.002
_HEAVY_VOL = 1.25


# ---- what Claude is shown ------------------------------------------------- #
def fact_sheet(payload: dict, row: dict) -> dict:
    """The base's measurements for one candidate: its scan ``payload`` and diagnostics
    ``row``.

    Returns a dict that is JSON-safe. ``swings`` are the detector's peak and trough
    points lettered A, B, C… in date order; ``legs`` pair each peak with its trough. A
    value that can't be measured is None. Nothing here may be RS, F, the quality score,
    the distance to the pivot, days to earnings or dollar volume."""
    df = payload["df"]
    cons = sorted(payload["vcp"].get("contractions") or [],
                  key=lambda c: pd.Timestamp(c["peak_date"]))
    swings, legs = [], []
    for i, c in enumerate(cons):
        a, b = chr(ord("A") + 2 * i), chr(ord("A") + 2 * i + 1)
        swings.append({"label": a, "kind": "peak", "date": _day(c["peak_date"]),
                       "price": round(float(c["peak_price"]), 2)})
        swings.append({"label": b, "kind": "trough", "date": _day(c["trough_date"]),
                       "price": round(float(c["trough_price"]), 2)})
        legs.append({"from": a, "to": b, "depth_pct": round(float(c["drawdown_pct"]), 1),
                     **_leg_volume(df, c["peak_date"], c["trough_date"])})
    depths = [lg["depth_pct"] for lg in legs]
    start = pd.Timestamp(cons[0]["peak_date"]) if cons else None
    base = df[df.index >= start] if start is not None else df.iloc[:0]
    return {
        "swings": swings,
        "legs": legs,
        "leg_ratios": _ratios(depths),
        "final_leg_pct": depths[-1] if depths else None,
        "base_weeks": (round(len(base) / 5.0, 1) if len(base) else None),
        "dryup_ratio": _num(row.get("dryup_ratio")),
        "quiet_days": _int(row.get("quiet_days")),
        "heavy_down_days_in_base": _heavy_down_days(df, start),
        "up_down_volume_in_base": _up_down_volume(base),
        "closed_below_200_in_base": _below_200(df, start),
        "gaps_over_8pct_120d": _int(row.get("gaps8")),
        "typical_day_pct": _num(row.get("day_range")),
        "stop_room_days": _num(row.get("stop_room")),
        "depth_vs_market": _num(row.get("depth_vs_spy")),
    }


def facts_text(facts: dict) -> str:
    """``facts`` as the lines the prompt shows, labelled with the chart's letters."""
    lines = []
    if facts["swings"]:
        lines.append("Swing points marked on the daily chart:")
        for s in facts["swings"]:
            lines.append(f"  {s['label']}  {s['kind']:6s} {s['date']}  {s['price']:.2f}")
        lines.append("Legs (peak to trough):")
        for lg in facts["legs"]:
            vol = (f", volume {lg['volume_x50']:.2f}x its 50-day average"
                   if lg.get("volume_x50") is not None else "")
            lines.append(f"  {lg['from']}->{lg['to']}  -{lg['depth_pct']:.1f}% over "
                         f"{lg['bars']} days{vol}")
        if facts["leg_ratios"]:
            lines.append("Each leg's depth over the one before: "
                         + ", ".join(f"{r:.2f}" for r in facts["leg_ratios"]))
    else:
        lines.append("The detector marked no swing points.")
    rows = (("base_weeks", "Base length (weeks, from A)", "{:.1f}"),
            ("dryup_ratio", "Final tight area's volume over its 50-day average", "{:.2f}x"),
            ("quiet_days", "Near-silent days in the final tight area", "{}"),
            ("heavy_down_days_in_base", "Down days on heavy volume since A", "{}"),
            ("up_down_volume_in_base", "Up-day volume over down-day volume since A", "{:.2f}"),
            ("closed_below_200_in_base", "Closed below the 200-day since A", "{}"),
            ("gaps_over_8pct_120d", "Gaps over 8% in the last 120 days", "{}"),
            ("typical_day_pct", "Typical day's range (%)", "{:.2f}"),
            ("stop_room_days", "Stop distance in typical days", "{:.1f}"),
            ("depth_vs_market", "Base depth over the market's fall in the same stretch",
             "{:.1f}x"))
    for key, label, fmt in rows:
        v = facts.get(key)
        if v is None:
            continue
        lines.append(f"{label}: {('yes' if v else 'no') if isinstance(v, bool) else fmt.format(v)}")
    return "\n".join(lines)


_PROMPT = """You are reviewing one stock chart for a Minervini SEPA trader. The stock already \
passed the trend template. Your job is only to describe the base: is it a well-formed \
volatility contraction pattern (VCP) or another sound base? You do not give a verdict, \
and you do not judge where the price is now relative to the buy point. Judge the base \
even if the price has since moved above or below it.

Read these two images with the Read tool, then answer. Do not read any other file.
  Weekly chart, about two years: {weekly}
  Daily chart of the base, volume below: {daily}
The daily chart marks the detector's swing points with letters. Candles are green up, \
red down. Lines: blue 50-day, orange 150-day, purple 200-day (weekly: 10, 30, 40 weeks). \
The blue line on the volume pane is the 50-day average volume.

Measured from the price data (exact; trust these over pixel estimates):
{facts}

What a sound base looks like (Minervini): after an advance the stock pulls back two to \
six times, each pullback smaller than the one before, roughly half (25 -> 12 -> 6 -> 3%). \
Halving is a rule of thumb, not a requirement: pullbacks that shrink slowly, or level off \
at about the same depth, are still a sound base. Volume dries up as the swings shrink \
and is lowest in the final tight area. Higher lows; quick bounces off lows and closes in \
the upper half of the day's range. The pivot is the high of the last tight area. Warning \
signs: pullbacks getting clearly larger, heavy volume on down days, wide erratic daily \
swings, a V-shaped run up the right side with no pause, breaking the 200-day. The first \
and second bases of an advance work best.

Answer every question with exactly one of its options:
- pattern: what is this? vcp | cup_with_handle | flat_base | double_bottom | \
high_tight_flag | no_base (a stair-step advance, a V, sideways chop, or no pullbacks \
toward a pivot).
- swings_match: do the lettered points sit on the real swing highs and lows of the \
base? yes | no. If no, list the peaks and troughs you see in swings_seen as dates \
(YYYY-MM-DD) with kind peak or trough; otherwise leave swings_seen empty.
- right_side: how do the pullbacks behave from left to right? orderly_tightening (each \
pullback smaller than, or about as deep as, the one before; shrinking slowly or not \
halving still counts) | wider_leg (a later pullback clearly deeper than the one before: \
more than {expand:.2f} times its depth, on the measured legs or, if the lettered points \
are wrong, on the swings you see) | v_shape (a straight run up the right side with no \
pause).
- final_area: the last tight area before the pivot. tight (narrow days, closes stacked) \
| loose (wide days or closes) | not_formed (no tight area yet).
- volume: through the base. drying_up | mixed | heavy_selling (clusters of heavy volume \
on down days).
- pivot: one clear level to break | several (competing highs) | none.
- character: the daily action. quick_recoveries (quick bounces, closes in the upper \
half) | grinding (slow, heavy recoveries) | wild_swings (normal days wider than a \
sensible stop).
- base_count: which base of the advance is this, from the weekly chart? \
first_or_second | third | fourth_or_later | cant_tell.

For each question write one short line of evidence in evidence (what on the chart or in \
the numbers decided it). Set sure to false only if, on pattern, right_side, volume or \
character, two answers were about equally likely. A close call on any other question is \
not a reason."""


def build_prompt(weekly: Path, daily: Path, facts: dict) -> str:
    return _PROMPT.format(weekly=weekly, daily=daily, facts=facts_text(facts),
                          expand=EXPAND_RATIO)


def schema() -> dict:
    """The JSON schema every answer MUST match."""
    return {
        "type": "object", "additionalProperties": False,
        "required": [*FORM, "swings_seen", "evidence", "sure"],
        "properties": {
            **{k: {"type": "string", "enum": list(v)} for k, v in FORM.items()},
            "swings_seen": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["date", "kind"],
                "properties": {"date": {"type": "string"},
                               "kind": {"type": "string", "enum": ["peak", "trough"]}}}},
            "evidence": {"type": "object", "additionalProperties": False,
                         "required": list(FORM),
                         "properties": {k: {"type": "string"} for k in FORM}},
            "sure": {"type": "boolean"},
        },
    }


def prompt_hash() -> str:
    """Short hash of the prompt, the schema and the verdict table: one value per review
    method, recorded with every run."""
    blob = (_PROMPT + json.dumps(schema(), sort_keys=True) + f"table{TABLE_VERSION}"
            + f"expand{EXPAND_RATIO}")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def validate_form(d) -> List[str]:
    """Problems with one answer ``d``; empty means it matches ``schema()``."""
    if not isinstance(d, dict):
        return ["not an object"]
    problems = [f"{k}: {d.get(k)!r} is not one of {list(v)}"
                for k, v in FORM.items() if d.get(k) not in v]
    ev = d.get("evidence")
    if not isinstance(ev, dict) or any(not isinstance(ev.get(k), str) for k in FORM):
        problems.append("evidence: one string per question")
    sw = d.get("swings_seen")
    if not isinstance(sw, list) or any(not isinstance(s, dict) or s.get("kind") not in
                                       ("peak", "trough") for s in sw):
        problems.append("swings_seen: a list of {date, kind}")
    if not isinstance(d.get("sure"), bool):
        problems.append("sure: true or false")
    return problems


# ---- what code decides ---------------------------------------------------- #
_DQ = (("pattern", "no_base", "DQ:no-base"), ("right_side", "wider_leg", "DQ:expanding"),
       ("right_side", "v_shape", "DQ:v-shape"), ("volume", "heavy_selling", "DQ:distribution"),
       ("character", "wild_swings", "DQ:bronco"))
_CAV = (("final_area", "loose", "CAV:loose"), ("final_area", "not_formed", "CAV:pivot"),
        ("pivot", "several", "CAV:pivot"), ("pivot", "none", "CAV:pivot"),
        ("volume", "mixed", "CAV:no-dryup"), ("base_count", "fourth_or_later", "CAV:late"))


_MEASURED = {"DQ:below-200": "closed below the 200-day during the base",
             "DQ:bronco": "stop inside one typical day"}


def measured_fail(facts: dict, row: dict) -> Optional[str]:
    """The DQ code when a measurement fails the base whatever the form says, else None:
    a close below the 200-day during the base, then a stop inside one typical day."""
    if facts.get("closed_below_200_in_base"):
        return "DQ:below-200"
    room = _num(row.get("stop_room"))
    if room is not None and room < 1.0:
        return "DQ:bronco"
    return None


def verdict_from_form(form: dict, facts: dict, row: dict) -> Tuple[str, List[str], str]:
    """The verdict table (version ``TABLE_VERSION``).

    Returns ``(verdict, codes, question)``: ``question`` is the form key whose answer
    decided it, or "" when a measurement did. FAIL takes the first disqualifier in this
    order: closed below the 200-day during the base, a stop inside one typical day, then
    the form's no base, wider leg, V, heavy selling, wild swings. Otherwise each caveat
    adds a code (a dry-up ratio of 1.0 or more and a capped order size count too) and
    makes it PASS-. Otherwise PASS."""
    code = measured_fail(facts, row)
    if code:
        return "FAIL", [code], ""
    for q, a, code in _DQ:
        if form.get(q) == a:
            return "FAIL", [code], q
    codes, first = [], ""
    for q, a, code in _CAV:
        if form.get(q) == a and code not in codes:
            codes.append(code)
            first = first or q
    du = _num(facts.get("dryup_ratio"))
    if du is not None and du >= 1.0 and "CAV:no-dryup" not in codes:
        codes.append("CAV:no-dryup")
    if _int(row.get("size_capped")) == 1:
        codes.append("CAV:thin")
    if codes:
        return "PASS-", codes, first
    return "PASS", ["OK"], ""


def conflicts(form: dict, facts: dict, df: Optional[pd.DataFrame] = None) -> List[str]:
    """Where an answer contradicts the measurements. With ``swings_match`` = no and
    ``df`` given, the legs are re-measured from the points Claude named."""
    depths = [lg["depth_pct"] for lg in facts.get("legs") or []]
    if form.get("swings_match") == "no" and df is not None:
        mine = [lg["depth_pct"] for lg in measure_swings(df, form.get("swings_seen") or [])]
        depths = mine or depths
    ratios = _ratios(depths)
    out = []
    side = form.get("right_side")
    if side == "orderly_tightening" and any(r > EXPAND_RATIO for r in ratios):
        out.append("tightening, but a measured leg is clearly deeper than the one before")
    if side == "wider_leg" and ratios and all(r <= EXPAND_RATIO for r in ratios):
        out.append("wider leg, but no measured leg is clearly deeper than the one before")
    du = _num(facts.get("dryup_ratio"))
    if form.get("volume") == "drying_up" and du is not None and du >= 1.0:
        out.append(f"drying up, but the final area trades {du:.2f}x its average")
    return out


def measure_swings(df: pd.DataFrame, points: List[dict]) -> List[dict]:
    """Legs measured from the peaks and troughs in ``points`` ({date, kind}).

    Each date snaps to the highest high (peak) or lowest low (trough) within two bars.
    A peak pairs with the next trough. Unreadable dates and points outside ``df`` are
    dropped. Returns ``[{peak_date, trough_date, depth_pct}]`` in date order."""
    snapped = []
    for p in points:
        try:
            t = pd.Timestamp(p["date"])
        except (KeyError, ValueError, TypeError):
            continue
        if df.empty or t < df.index[0] or t > df.index[-1]:
            continue
        i = int(df.index.get_indexer([t], method="nearest")[0])
        win = df.iloc[max(0, i - 2):i + 3]
        if p.get("kind") == "peak":
            snapped.append(("peak", win["High"].idxmax(), float(win["High"].max())))
        elif p.get("kind") == "trough":
            snapped.append(("trough", win["Low"].idxmin(), float(win["Low"].min())))
    snapped.sort(key=lambda s: s[1])
    legs, peak = [], None
    for kind, when, price in snapped:
        if kind == "peak":
            peak = (when, price)
        elif peak is not None and price < peak[1]:
            legs.append({"peak_date": _day(peak[0]), "trough_date": _day(when),
                         "depth_pct": round((1.0 - price / peak[1]) * 100.0, 1)})
            peak = None
    return legs


def combine(votes: List[dict]) -> Tuple[dict, bool]:
    """One answer from several valid ``votes``: each question takes the most common
    answer, and a tie goes to the answer that weighs most against the base, then to the
    earliest vote. Returns ``(form, all_sure)``."""
    form: Dict[str, object] = {}
    for q in FORM:
        answers = [v[q] for v in votes]
        counts = Counter(answers)
        top = max(counts.values())
        tied = [a for a in dict.fromkeys(answers) if counts[a] == top]
        form[q] = max(tied, key=lambda a: _SEVERITY.get(a, 0))
    form["evidence"] = {q: next(v["evidence"].get(q, "") for v in votes if v[q] == form[q])
                        for q in FORM}
    form["swings_seen"] = next((v.get("swings_seen") or [] for v in votes
                                if v["swings_match"] == form["swings_match"]), [])
    sure = all(v.get("sure") for v in votes)
    form["sure"] = sure
    return form, sure


def settled(votes: List[dict]) -> bool:
    """True when the first two answers agree on every question. A third answer can't
    change any majority then, so the ticker needs no more."""
    return len(votes) >= 2 and all(votes[0][q] == votes[1][q] for q in FORM)


def decide(votes: List[dict], facts: dict, row: dict, df: Optional[pd.DataFrame] = None,
           wanted: int = REVIEW_VOTES) -> dict:
    """The verdict for one ticker from its valid ``votes``.

    Returns ``{verdict, codes, form, needs_check, notes}``. ``needs_check`` is empty when
    nothing calls for the user's eye."""
    form, sure = combine(votes)
    verdict, codes, q = verdict_from_form(form, facts, row)
    each = {verdict_from_form(v, facts, row)[0] for v in votes}
    checks = []
    if len(votes) < wanted:
        checks.append(f"only {len(votes)} of {wanted} answers")
    if len(each) > 1:
        checks.append("answers split " + "/".join(sorted(each)))
    if not sure:
        checks.append("unsure")
    checks += conflicts(form, facts, df)
    return {"verdict": verdict, "codes": codes, "form": form, "needs_check": checks,
            "notes": _note(verdict, codes, form, facts, q, checks)}


def _note(verdict: str, codes: List[str], form: dict, facts: dict, q: str,
          checks: List[str]) -> str:
    if verdict == "PASS-":
        head = "CAV:" + ",".join(c.split(":", 1)[1] for c in codes)
    else:
        head = codes[0]
    legs = "-".join(f"{lg['depth_pct']:.0f}" for lg in facts.get("legs") or [])
    parts = [form["pattern"].replace("_", " ")]
    if legs:
        parts.append(f"{legs} legs")
    if facts.get("dryup_ratio") is not None:
        parts.append(f"DU {facts['dryup_ratio']:.2f}x")
    text = ", ".join(parts)
    why = (form["evidence"].get(q) or "").strip() if q else ""
    if why:
        text += f". {why[:90]}"
    if checks:
        text += ". Check: " + ", ".join(checks)
    return f"{head}; {text.replace(';', ',')}"


# ---- running it ----------------------------------------------------------- #
Caller = Callable[[str, dict], dict]


def claude_cli(claude_bin: str, model: str = REVIEW_MODEL, effort: str = REVIEW_EFFORT,
               timeout_s: int = 600, cwd: Optional[Path] = None) -> Caller:
    """A ``Caller`` that runs one headless Claude Code call per answer.

    The call gets the Read tool only, no project files (``--safe-mode``) and the answer
    schema. The returned function gives ``{form, error, model, cost_usd, seconds}``;
    ``form`` is None when the call failed or returned no structured answer."""
    sch = json.dumps(schema(), separators=(",", ":"))

    def call(prompt: str, _schema: dict) -> dict:
        argv = [claude_bin, "-p", "--safe-mode", "--tools", "Read", "--allowedTools", "Read",
                "--permission-mode", "dontAsk", "--no-session-persistence",
                "--output-format", "json", "--json-schema", sch, "--model", model]
        if effort:
            argv += ["--effort", effort]
        t0 = time.time()
        try:
            p = subprocess.run(argv, input=prompt, capture_output=True, text=True,
                               encoding="utf-8", timeout=timeout_s, cwd=cwd)
        except subprocess.TimeoutExpired:
            return {"form": None, "error": f"timed out after {timeout_s}s"}
        except OSError as e:
            return {"form": None, "error": f"could not start {claude_bin}: {e}"}
        secs = round(time.time() - t0, 1)
        try:
            out = json.loads(p.stdout)
        except ValueError:
            tail = (p.stderr or p.stdout or "").strip()[-300:]
            return {"form": None, "error": f"exit {p.returncode}: {tail}", "seconds": secs}
        form = out.get("structured_output")
        ok = isinstance(form, dict)
        return {"form": form if ok else None,
                "error": None if ok else str(out.get("result") or "no answer")[-300:],
                "model": ",".join(sorted(out.get("modelUsage") or {})) or None,
                "cost_usd": out.get("total_cost_usd"), "seconds": secs}
    return call


def review_paths(hunt_path: Path, ticker: str) -> Tuple[Path, Path, Path]:
    """``(weekly png, daily png, facts json)`` for ``ticker`` in a hunt folder."""
    d = hunt_path / REVIEW_SUBDIR
    return d / f"{ticker}_weekly.png", d / f"{ticker}_daily.png", d / f"{ticker}.json"


def read_votes(hunt_path: Path) -> Dict[str, List[dict]]:
    """Valid answers per ticker from ``votes.jsonl``, in file order. Lines that don't
    parse, and answers that fail ``validate_form``, are skipped."""
    path = hunt_path / REVIEW_SUBDIR / VOTES_FILE
    out: Dict[str, List[dict]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and not validate_form(rec.get("form")):
            out.setdefault(rec["ticker"], []).append(rec["form"])
    return out


def run_review(hunt_path: Path, diag: pd.DataFrame, call: Caller, *,
               votes: int = REVIEW_VOTES, jobs: int = 4,
               frames: Optional[Dict[str, pd.DataFrame]] = None,
               log: Callable[[str], None] = print) -> dict:
    """Review every candidate in ``diag`` that has no verdict yet, and record verdicts.

    Needs ``review-charts`` to have written each ticker's images and facts. A ticker
    ``measured_fail`` fails gets its FAIL without a call. Every other ticker is asked
    for two valid answers. A ticker that got some answer but isn't ``settled`` is then
    asked until it has ``votes``. A failed call is retried once. Every answer is
    appended to ``votes.jsonl`` as it arrives, so a rerun resumes. Then writes
    ``form.csv`` (asked tickers only) and appends one coded verdict per ticker that has a
    measured FAIL or at least one answer. ``frames`` (ticker -> daily OHLCV) lets
    ``conflicts`` re-measure Claude's swings.

    Returns ``{verdicts, needs_check, no_answer, measured}``: the counts, the tickers to
    look at with their reasons, the tickers left without a verdict, and the tickers that
    failed on a measurement."""
    vpath = hunt_path / "verdicts.csv"
    rdir = hunt_path / REVIEW_SUBDIR
    rdir.mkdir(exist_ok=True)
    have = pl.read_verdicts(vpath)
    rows = {r["ticker"]: r for r in diag.to_dict("records")}
    todo = [t for t in rows if t not in have]
    for t in todo:
        for p in review_paths(hunt_path, t):
            if not p.exists():
                raise pl.HuntError(f"{p} not found — run `review-charts` first.")

    def read_facts(t: str) -> dict:
        return json.loads(review_paths(hunt_path, t)[2].read_text(encoding="utf-8"))

    facts = {t: read_facts(t) for t in todo}
    measured = {t: measured_fail(facts[t], rows[t]) for t in todo}
    measured = {t: c for t, c in measured.items() if c}
    asked = [t for t in todo if t not in measured]
    got = read_votes(hunt_path)
    lock = threading.Lock()
    sch = schema()

    def ask(t: str) -> None:
        weekly, daily, fpath = review_paths(hunt_path, t)
        facts = json.loads(fpath.read_text(encoding="utf-8"))
        prompt = build_prompt(weekly.resolve(), daily.resolve(), facts)
        res = {}
        for _ in range(2):
            res = call(prompt, sch)
            if res.get("form") is not None and not validate_form(res["form"]):
                break
            res["error"] = res.get("error") or "; ".join(validate_form(res.get("form")))
        rec = {"ticker": t, "form": res.get("form"), "error": res.get("error"),
               "model": res.get("model"), "cost_usd": res.get("cost_usd"),
               "seconds": res.get("seconds")}
        with lock:
            with open(rdir / VOTES_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            if rec["error"]:
                log(f"{t}: {rec['error']}")

    def run(want: Callable[[str], int], what: str) -> None:
        tasks = [t for t in asked for _ in range(max(0, want(t) - len(got.get(t, []))))]
        log(f"review: {what}, {len(tasks)} calls, {jobs} at a time")
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            list(ex.map(ask, tasks))

    log(f"review: {len(todo)} tickers, {len(measured)} failed on a measurement")
    run(lambda t: min(votes, 2), f"{len(asked)} tickers asked")
    got = read_votes(hunt_path)
    open_ = {t for t in asked if got.get(t) and not settled(got[t])}
    run(lambda t: votes if t in open_ else 0, f"{len(open_)} not settled by two answers")

    got = read_votes(hunt_path)
    decided, new_rows, no_answer = {}, [], []
    for t in rows:
        if t in measured:
            new_rows.append({"ticker": t, "verdict": "FAIL",
                             "notes": f"{measured[t]}; {_MEASURED[measured[t]]}, not asked"})
            continue
        if not got.get(t):
            if t not in have:
                no_answer.append(t)
            continue
        v = got[t][:votes]
        df = (frames or {}).get(t)
        need = min(votes, 2) if settled(v) else votes
        decided[t] = decide(v, facts.get(t) or read_facts(t), rows[t], df, need)
        if t not in have:
            new_rows.append({"ticker": t, "verdict": decided[t]["verdict"],
                             "notes": decided[t]["notes"]})
    _write_form(rdir / FORM_FILE, decided, {t: len(v) for t, v in got.items()})
    pl.append_verdicts(vpath, new_rows)
    counts = Counter(d["verdict"] for d in decided.values())
    counts["FAIL"] += len(measured)
    return {"verdicts": {k: n for k, n in counts.items() if n},
            "needs_check": {t: d["needs_check"] for t, d in decided.items()
                            if d["needs_check"]},
            "no_answer": no_answer, "measured": sorted(measured)}


FORM_COLUMNS = ["ticker", "verdict", "codes", *FORM, "sure", "swings_seen", "needs_check",
                "answers"]


def _write_form(path: Path, decided: Dict[str, dict], n_answers: Dict[str, int]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FORM_COLUMNS)
        w.writeheader()
        for t, d in decided.items():
            fm = d["form"]
            w.writerow({"ticker": t, "verdict": d["verdict"], "codes": ",".join(d["codes"]),
                        **{q: fm[q] for q in FORM}, "sure": int(bool(fm["sure"])),
                        "swings_seen": json.dumps(fm.get("swings_seen") or []),
                        "needs_check": " | ".join(d["needs_check"]),
                        "answers": n_answers.get(t, 0)})


# ---- measuring agreement with the user ------------------------------------ #
LABEL_COLUMNS = ["ticker", "verdict", *FORM]


def write_label_template(path: Path, tickers: List[str]) -> None:
    """A CSV with one blank row per ticker for the user's own answers. Any column MAY
    stay blank; blanks are left out of the scores."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        w.writeheader()
        for t in tickers:
            w.writerow({"ticker": t})


def read_labels(path: Path) -> Dict[str, dict]:
    """``ticker -> {column: value}`` from a labels or ``form.csv`` file, blanks dropped.
    Raises HuntError when the file is missing or has no ticker column, or a cell holds
    an answer that isn't one of its question's options."""
    if not path.exists():
        raise pl.HuntError(f"{path} not found.")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "ticker" not in (reader.fieldnames or []):
            raise pl.HuntError(f"{path} needs a ticker column.")
        out = {}
        for r in reader:
            row = {k: (r.get(k) or "").strip() for k in LABEL_COLUMNS if (r.get(k) or "").strip()}
            for k, v in row.items():
                if k in FORM and v not in FORM[k]:
                    raise pl.HuntError(f"{path}: {r['ticker']} {k}={v!r} is not one of "
                                       f"{list(FORM[k])}")
                if k == "verdict" and v not in pl.VERDICTS:
                    raise pl.HuntError(f"{path}: {r['ticker']} verdict {v!r}")
            out[row.get("ticker", "")] = row
    out.pop("", None)
    return out


def score(mine: Dict[str, dict], theirs: Dict[str, dict]) -> dict:
    """How often ``theirs`` (the reviewer, or a re-label) agrees with ``mine`` (the user).

    Each question counts only tickers both answered. Returns per-question agreement, the
    verdict agreement with Cohen's kappa and the confusion table (``mine -> theirs ->
    count``), and the two costly errors: ``false_pass`` (theirs PASS where mine is not)
    and ``missed_pass`` (mine PASS where theirs is not)."""
    out: dict = {"questions": {}}
    for q in ("verdict", *FORM):
        both = [t for t in mine if q in mine[t] and q in (theirs.get(t) or {})]
        agree = sum(mine[t][q] == theirs[t][q] for t in both)
        out["questions"][q] = {"n": len(both),
                               "agree_pct": round(100.0 * agree / len(both), 1) if both else None}
    both = [t for t in mine if "verdict" in mine[t] and "verdict" in (theirs.get(t) or {})]
    conf = {a: {b: 0 for b in pl.VERDICTS} for a in pl.VERDICTS}
    for t in both:
        conf[mine[t]["verdict"]][theirs[t]["verdict"]] += 1
    n = len(both)
    po = sum(conf[v][v] for v in pl.VERDICTS) / n if n else None
    pe = (sum(sum(conf[v].values()) * sum(conf[a][v] for a in pl.VERDICTS)
              for v in pl.VERDICTS) / n ** 2 if n else None)
    out["verdict"] = {
        "n": n, "confusion": conf,
        "kappa": (round((po - pe) / (1 - pe), 3) if n and pe < 1 else None),
        "false_pass": [t for t in both if theirs[t]["verdict"] == "PASS"
                       and mine[t]["verdict"] != "PASS"],
        "missed_pass": [t for t in both if mine[t]["verdict"] == "PASS"
                        and theirs[t]["verdict"] != "PASS"],
    }
    return out


# ---- small helpers -------------------------------------------------------- #
def _day(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%d")


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _int(v) -> Optional[int]:
    f = _num(v)
    return None if f is None else int(f)


def _ratios(depths: List[float]) -> List[float]:
    return [round(b / a, 2) for a, b in zip(depths, depths[1:]) if a]


def _leg_volume(df: pd.DataFrame, peak, trough) -> dict:
    leg = df[(df.index >= pd.Timestamp(peak)) & (df.index <= pd.Timestamp(trough))]
    avg = prior_volume_average(df["Volume"], VOL_AVG_DAYS)
    base = avg.loc[leg.index[0]] if len(leg) else np.nan
    x = float(leg["Volume"].mean() / base) if len(leg) and base and base > 0 else None
    return {"bars": int(len(leg)), "volume_x50": (round(x, 2) if x is not None else None)}


def _heavy_down_days(df: pd.DataFrame, start) -> Optional[int]:
    if start is None:
        return None
    avg = prior_volume_average(df["Volume"], VOL_AVG_DAYS)
    ret = np.log(df["Close"]).diff()
    m = (df.index >= start) & (ret < _DOWN_RET) & (df["Volume"] > _HEAVY_VOL * avg)
    return int(m.sum())


def _up_down_volume(base: pd.DataFrame) -> Optional[float]:
    if len(base) < 2:
        return None
    chg = base["Close"].diff()
    down = float(base["Volume"][chg < 0].sum())
    return round(float(base["Volume"][chg > 0].sum()) / down, 2) if down > 0 else None


def _below_200(df: pd.DataFrame, start) -> Optional[bool]:
    if start is None:
        return None
    sma = df["Close"].rolling(200).mean()
    m = (df.index >= start) & sma.notna()
    if not m.any():
        return None
    return bool((df["Close"][m] < sma[m]).any())
