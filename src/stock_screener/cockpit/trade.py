"""Paper-trade the cockpit watchlist through Alpaca.

Each name is sized by a chosen mode (see :func:`build_buy_plan`): a % of equity, a flat $
amount, an explicit share count, or **risk-to-stop** —
``shares = floor((equity × risk%) / (price − stop))``, Minervini's position sizer, so a
stop-out costs ≈ risk% of the account (the idea the Step-4 panel shows, sized on the live
fill price rather than the pivot). Orders are plain market BUYs on the **paper** account
(``alpaca_trader.connect()`` forces ``paper=True``); the guardrails apply — a per-order
floor of $50 (dollar-denominated modes) and a 10%-of-equity single-order cap (the risk mode
clamps to it and flags ``capped``; the other modes skip an over-cap name — never fatal).

The plan builder (:func:`build_buy_plan`) is pure and network-free so it's unit-tested;
:func:`submit_buy_plan` is the thin side-effecting wrapper that talks to Alpaca and is
imported lazily (so the cockpit still loads when ``alpaca-py`` isn't installed).
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

MIN_TRADE_USD = 50.0        # mirrors alpaca_trader.MIN_TRADE_USD (kept here so the pure
                            # plan builder needn't import alpaca-py)
MAX_ORDER_PCT = 0.10        # mirrors alpaca_trader.MAX_ORDER_PCT — single-order cap as a
                            # fraction of equity; the risk mode clamps to it (submit re-checks)
STALE_PLAN_BARS = 2         # skip a plan name whose freshest daily bar is more than this many
                            # *trading* days old rather than size on stale data (the scan memo
                            # has no time-based invalidation). 2 absorbs a weekend + a holiday.
# When a frozen judged_pivot drives the plan, mirror scan._entry_levels: default stop 7.5% below
# the pivot, hard-floored at 10% below it (Minervini's 7-8% ideal / 10% max).
DEFAULT_STOP_FROM_PIVOT = 0.075
MAX_STOP_FROM_PIVOT = 0.10

# --- Positions-page stop management (Minervini exit rules) ---------------------------------- #
INITIAL_STOP_PCT = 0.08     # ~8% initial stop below the entry (buy point)
BREAKEVEN_GAIN = 0.16       # gain past which the stop should be at least breakeven (~2x initial risk)
TRAIL_GAIN = 0.20           # gain past which, "well in profit", trail the 50-day SMA
SELL_STRENGTH_GAIN = 0.20   # gain past which to consider selling part into strength
HEAVY_VOL_RATIO = 1.5       # latest volume vs its 50-day average = a heavy-volume day
EARNINGS_SOON_DAYS = 21     # mirrors triggers.EARNINGS_SOON_DAYS (kept here so this module
                            # needn't import the pandas-heavy triggers/scan stack at load time)
EARNINGS_CUSHION_MIN = 0.08  # min profit cushion to comfortably hold a position through a report
# Suggested-stop bases for the re-arm action; "auto" picks per position by its gain.
STOP_BASES = ("auto", "initial", "breakeven", "sma50")

# --- Progressive exposure (risk-% guidance off recent closed trades) ------------------------ #
RISK_GUIDE_LAST_N = 10      # recent form, not lifetime stats, drives exposure
RISK_GUIDE_MIN_TRADES = 5   # below this the sample says nothing — stay at base
RISK_PCT_PILOT = 0.5        # cold numbers -> half the base unit until they improve
RISK_PCT_BASE = 1.0         # the widget default — neutral evidence
RISK_PCT_STRONG = 1.25      # proven recent edge -> press modestly, never double

# --- Sell pillars (P1 breakout-holding thresholds; P2-P4 reuse constants above) ------------- #
P1_CUSHION_PCT = 0.03       # real breakouts pay quickly — want ~3% by the cushion day
P1_CUSHION_DAYS = 10        # trading days; no cushion by here -> sell into strength
P1_STALL_DAYS = 15          # flat-to-red by here -> exit (calibration, not scripture)
DECISIVE_BELOW_PIVOT_PCT = 0.02  # a close >2% below the pivot is decisive, not noise

# --- Progressive-exposure gate + free-roll -------------------------------------------------- #
GATE_HALF_SIZE_AFTER = 2    # consecutive losing closed trades -> advise half-size probes
FREE_ROLL_R = 2.0           # R-multiple where selling part + breakeven stop is advised
FREE_ROLL_FRACTION = 0.5    # fraction the free-roll action sells by default

# The cockpit trades a SEPARATE Alpaca paper account from the All-Weather mirror (which owns
# the shared ALPACA_API_KEY/SECRET pair). Each paper account has its own key pair, so prefer
# the dedicated "Minervini Trader" keys, falling back to the shared pair only if unset. The
# dedicated names must match .env EXACTLY — a mismatch silently resolves to None and the code
# then trades the SHARED account — so keep these two constants in sync with .env.
MINERVINI_KEY_ENVS = "ALPACA_API_KEY_MINERVINI"
MINERVINI_SECRET_ENVS = "ALPACA_API_KEY_SECRET_MINERVINI"
# The shared pair accepts either spelling (_first_env takes the first non-empty in the tuple).
SHARED_KEY_ENVS = ("ALPACA_API_KEY", "ALPACA_API_KEY_PAPER1")
SHARED_SECRET_ENVS = ("ALPACA_API_SECRET", "ALPACA_API_SECRET_PAPER1")


def _first_env(names: "Sequence[str] | str") -> Optional[str]:
    """Return the value of the first env var named in ``names`` that is set and non-empty.

    Accepts a single name (``str``) or a sequence of candidate names. A bare string is
    treated as ONE name — never iterated character-by-character, which would silently
    resolve a stray one-char env var (e.g. ``$_``) instead of the intended key.
    """
    if isinstance(names, str):
        names = (names,)
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


class TradeUnavailable(RuntimeError):
    """Alpaca can't be reached — package missing, or credentials absent from .env."""


def _connect_paper():
    """Return ``(paper TradingClient, using_dedicated)`` for the cockpit's account.

    Prefers the dedicated Minervini keys; falls back to the shared ALPACA_* pair. Always
    ``paper=True``. Raises :class:`TradeUnavailable` if alpaca-py or credentials are missing.
    """
    try:
        from alpaca.trading.client import TradingClient
    except ImportError as e:
        raise TradeUnavailable(
            "alpaca-py is not installed — run `pip install alpaca-py` "
            "(or `conda env update -f environment.yml`).") from e
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    ded_key = _first_env(MINERVINI_KEY_ENVS)
    ded_secret = _first_env(MINERVINI_SECRET_ENVS)
    key = ded_key or _first_env(SHARED_KEY_ENVS)
    secret = ded_secret or _first_env(SHARED_SECRET_ENVS)
    if not key or not secret:
        raise TradeUnavailable(
            "No Alpaca credentials in .env. Add the Minervini Trader paper account's keys as "
            "ALPACA_API_KEY_MINERVINI / ALPACA_API_KEY_SECRET_MINERVINI (each Alpaca paper "
            "account has its own key pair), or a shared ALPACA_API_KEY / ALPACA_API_SECRET pair.")
    return TradingClient(key, secret, paper=True), bool(ded_key and ded_secret)


def fetch_account_summary() -> dict:
    """Connect and read the target account so the UI can confirm *which* account will be
    traded before any order is sent. Returns ``{account_number, equity, cash,
    using_dedicated}``; raises :class:`TradeUnavailable` on missing package/credentials."""
    client, using_dedicated = _connect_paper()
    acct = client.get_account()
    return {
        "account_number": getattr(acct, "account_number", "?"),
        "equity": float(acct.equity),
        "cash": float(acct.cash),
        "using_dedicated": using_dedicated,
    }


def fetch_held_shares() -> Dict[str, int]:
    """``{symbol: whole shares held}`` on the cockpit's paper account (same ``int(float(qty))``
    convention as :func:`submit_buy_plan`). Lets the Build-plan preview mark already-held names
    ('stop re-arm only, no buy') since :func:`build_buy_plan` is holdings-blind. Raises
    :class:`TradeUnavailable` on missing package/credentials — the caller treats that as 'unknown'
    (no held annotations)."""
    client, _ = _connect_paper()
    return {p.symbol: int(float(p.qty)) for p in client.get_all_positions()}


def _pos_float(p, name) -> Optional[float]:
    try:
        v = getattr(p, name, None)
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def position_symbols() -> List[str]:
    """Just the symbols currently held on the paper account — no price history, no journal.

    Deliberately lighter than :func:`fetch_positions` and :func:`fetch_gate_inputs`: the
    refresh job calls this to decide WHICH prices to download, and ``fetch_positions``
    downloads price history itself, which would make that circular. A held name that has
    fallen off the watchlist still needs fresh bars — the Positions page and the sell
    pillars go blind on a position they cannot price. Raises :class:`TradeUnavailable`
    when credentials are absent; the caller degrades to watchlist-only."""
    client, _ = _connect_paper()
    return sorted({s for s in (getattr(p, "symbol", None)
                               for p in client.get_all_positions()) if s})


def fetch_gate_inputs() -> dict:
    """Light inputs for :func:`gate_status`: Alpaca positions only (NO price-history
    fetch — the unrealized figures come from the broker) plus the journal's open/closed
    episodes. Raises :class:`TradeUnavailable` on any failure; the manual trade panel
    treats that as gate UNKNOWN (open — the human judges), the unattended morning
    executor as CLOSED. The asymmetric fail direction is deliberate."""
    client, _ = _connect_paper()
    positions = []
    for p in client.get_all_positions():
        positions.append({"symbol": getattr(p, "symbol", None),
                          "qty": int(_pos_float(p, "qty") or 0),
                          "avg_entry": _pos_float(p, "avg_entry_price"),
                          "current_price": _pos_float(p, "current_price"),
                          "unrealized_pl": _pos_float(p, "unrealized_pl"),
                          "unrealized_plpc": _pos_float(p, "unrealized_plpc")})
    j = build_trade_journal(fetch_order_fills()["fills"])
    return {"positions": positions, "open_episodes": j["open"],
            "closed_episodes": j["closed"]}


def gate_status(positions: List[dict], open_episodes: List[dict],
                closed_episodes: List[dict]) -> dict:
    """Progressive-exposure gate: may a NEW position be opened right now? Pure.

    Scope is cockpit-TAGGED positions only — an Alpaca position counts only when a
    tagged OPEN journal episode exists for its symbol, so manual/legacy holdings never
    poison the gate (a book of untagged names reads as flat). Rules:

    * flat (no tagged positions) -> OPEN — the first pilot is always allowed;
    * otherwise every position in the NEWEST set (all tagged positions sharing the max
      entry DATE — same-open fills differ only by milliseconds, so day granularity)
      must be at breakeven or better, AND the tagged book's net unrealized P&L must be
      >= 0;
    * ``consecutive_losses`` counts losing TAGGED closed trades from the most recent
      backwards (re-sorted by ``exit_date`` — the journal groups by symbol; ``pl >= 0``
      including a $0 scratch breaks the streak); at ``GATE_HALF_SIZE_AFTER`` the
      ``probe_size_factor`` drops to 0.5. ADVISORY ONLY — surfaced in captions, never
      auto-applied to quantities.

    Sells and stop re-arms are never gated (risk-reducing). A position whose entry
    date can't be resolved counts as newest (conservative). Returns
    ``{open, reason, probe_size_factor, consecutive_losses}``."""
    import pandas as pd

    tagged_open = {e.get("symbol"): e for e in (open_episodes or []) if e.get("tagged")}
    scoped = [p for p in (positions or [])
              if p.get("symbol") in tagged_open and int(p.get("qty") or 0) > 0]

    streak = 0
    for t in sorted([t for t in (closed_episodes or []) if t.get("tagged")],
                    key=lambda t: t.get("exit_date"), reverse=True):
        pl = t.get("pl")
        if pl is not None and pl < 0:
            streak += 1
        else:
            break
    factor = 0.5 if streak >= GATE_HALF_SIZE_AFTER else 1.0
    half_note = (f"; {streak} consecutive losses — half-size probe advised"
                 if factor < 1.0 else "")

    if not scoped:
        return {"open": True,
                "reason": "book flat (no cockpit positions) — first pilot allowed"
                          + half_note,
                "probe_size_factor": factor, "consecutive_losses": streak}

    def _entry_day(sym):
        try:
            ts = pd.Timestamp(tagged_open[sym].get("entry_date"))
            if ts.tzinfo is not None:
                ts = ts.tz_convert("America/New_York").tz_localize(None)
            return ts.normalize()
        except Exception:
            return None

    days = {p["symbol"]: _entry_day(p["symbol"]) for p in scoped}
    known = [d for d in days.values() if d is not None]
    newest_day = max(known) if known else None
    newest = [p for p in scoped if days[p["symbol"]] in (None, newest_day)]

    losers = [p["symbol"] for p in newest
              if p.get("unrealized_plpc") is None or p["unrealized_plpc"] < 0]
    if losers:
        return {"open": False,
                "reason": "newest position below breakeven: " + ", ".join(sorted(losers))
                          + " — let the probe prove itself first" + half_note,
                "probe_size_factor": factor, "consecutive_losses": streak}
    net = sum(p.get("unrealized_pl") or 0.0 for p in scoped)
    if net < 0:
        return {"open": False,
                "reason": f"net open P&L ${net:,.0f} < 0 — work the book before adding"
                          + half_note,
                "probe_size_factor": factor, "consecutive_losses": streak}
    return {"open": True,
            "reason": "newest position at breakeven+ and net open P&L >= 0" + half_note,
            "probe_size_factor": factor, "consecutive_losses": streak}


SIZING_MODES = ("pct", "dollars", "shares", "risk")


def stop_is_valid(stop_price, price) -> bool:
    """A protective sell-stop is valid only strictly BELOW the reference price.

    Alpaca rejects a sell stop at/above the market (it would trigger instantly) and an OTO
    stop-loss leg that isn't below the entry. Used by the UI (live per-keystroke check) and
    re-checked in :func:`submit_buy_plan` against the last close.
    """
    return bool(stop_price and price and stop_price > 0 and stop_price < price)


def suggest_stop(*, avg_entry: Optional[float], current_price: Optional[float],
                 sma_50: Optional[float], current_stop: Optional[float],
                 gain_pct: Optional[float], basis: str = "auto"
                 ) -> Tuple[Optional[float], str]:
    """Minervini stop suggestion for a held position under a chosen ``basis``.

    Basis levels: ``initial`` = ``avg_entry × (1 - INITIAL_STOP_PCT)`` (~8% below entry);
    ``breakeven`` = ``avg_entry``; ``sma50`` = ``sma_50 × 0.99`` (just under the 50-day). ``auto``
    picks by gain (the position's stage): well in profit (``gain_pct >= TRAIL_GAIN``) with a 50-day
    available → trail the SMA; working (``gain_pct >= BREAKEVEN_GAIN``) → at least breakeven; else
    the initial 8% stop.

    Returns ``(suggested_price_or_None, effective_basis_label)``. The suggestion is floored at the
    current in-force stop — and, once the trade is working, at breakeven — so it is ratchet-safe
    (never proposes LOWER than what's in force, never gives back a working trade below breakeven).
    Returns ``None`` when no basis input is available or the result isn't strictly below
    ``current_price`` (underwater / already stopped-out territory → leave for a manual edit)."""
    initial_val = avg_entry * (1.0 - INITIAL_STOP_PCT) if avg_entry else None
    breakeven_val = float(avg_entry) if avg_entry else None
    sma_val = sma_50 * 0.99 if sma_50 else None

    if basis == "auto":
        if gain_pct is not None and gain_pct >= TRAIL_GAIN and sma_val is not None:
            eff = "sma50"
        elif gain_pct is not None and gain_pct >= BREAKEVEN_GAIN and breakeven_val is not None:
            eff = "breakeven"
        else:
            eff = "initial"
    else:
        eff = basis

    base_val = {"initial": initial_val, "breakeven": breakeven_val, "sma50": sma_val}.get(eff)
    # Ratchet-safe floor: never below the in-force stop. In AUTO mode also never below breakeven
    # once the trade is working (so trailing the 50-day can't give back a won trade) — an EXPLICIT
    # basis is honored as chosen (still floored at the in-force stop).
    floors = [v for v in (base_val, current_stop) if v is not None]
    if (basis == "auto" and gain_pct is not None and gain_pct >= BREAKEVEN_GAIN
            and breakeven_val is not None):
        floors.append(breakeven_val)
    if not floors:
        return None, eff
    cand = round(max(floors), 2)
    return (cand, eff) if stop_is_valid(cand, current_price) else (None, eff)


def position_stage(gain_pct: Optional[float]) -> Optional[str]:
    """The position's stage on the Minervini stop ladder, from its gain. Pure. Uses the SAME
    thresholds as :func:`suggest_stop`'s auto basis so the label and the suggested stop agree."""
    if gain_pct is None:
        return None
    if gain_pct < 0:
        return "underwater"
    if gain_pct < BREAKEVEN_GAIN:
        return "fresh"
    if gain_pct < TRAIL_GAIN:
        return "working"
    return "well in profit"


def r_multiple(avg_entry, current_price, pivot=None) -> Tuple[Optional[float], bool]:
    """The position's gain as a multiple of its reconstructed initial risk. Pure.

    The entry-time stop is not persisted anywhere, so risk is reconstructed: a frozen
    pivot whose derived stop (``pivot × (1 - DEFAULT_STOP_FROM_PIVOT)``) sits below the
    entry reproduces the level the OTO actually attached — exact, ``approximate=False``.
    Otherwise ``INITIAL_STOP_PCT`` off the entry (``approximate=True`` — also the path
    when the pivot sits at/above the entry, where pivot-derived risk would be <= 0).
    Returns ``(r, approximate)``; ``(None, True)`` on missing/degenerate inputs."""
    try:
        e, c = float(avg_entry), float(current_price)
    except (TypeError, ValueError):
        return None, True
    if e <= 0:
        return None, True
    risk = None
    approx = True
    if pivot:
        try:
            stop = float(pivot) * (1.0 - DEFAULT_STOP_FROM_PIVOT)
            if 0.0 < stop < e:
                risk, approx = e - stop, False
        except (TypeError, ValueError):
            pass
    if risk is None:
        risk = e * INITIAL_STOP_PCT
    if risk <= 0:
        return None, True
    return (c - e) / risk, approx


def position_advisories(pos: dict) -> List[str]:
    """Display-only Minervini exit advisories derived from a :func:`fetch_positions` dict. Pure.

    Note the "×initial-risk" rule (#4) approximates the initial risk at ``INITIAL_STOP_PCT`` (8%),
    because the entry-time stop distance isn't persisted anywhere — so it's a nudge, not exact.
    The earnings-cushion rules fire only for a KNOWN upcoming report (``earnings_in`` 0..21) with
    a KNOWN gain — a just-reported name (negative days) or missing data stays silent."""
    out: List[str] = []
    gain = pos.get("gain_pct")
    avg_entry = pos.get("avg_entry")
    cur_stop = pos.get("current_stop")

    if not pos.get("has_stop"):
        out.append("⚠ No protective stop armed — arm one.")
    ei = pos.get("earnings_in")
    if ei is not None and 0 <= ei <= EARNINGS_SOON_DAYS and gain is not None:
        # A stop can't protect against an earnings gap — the exit decision must come BEFORE
        # the report unless the position has already built a cushion.
        if gain < 0:
            out.append(f"⚠ Earnings in {int(ei)}d with a loss — no cushion; "
                       "exit or reduce before the report.")
        elif gain < EARNINGS_CUSHION_MIN:
            out.append(f"⚠ Earnings in {int(ei)}d with only a {gain * 100:.0f}% cushion — "
                       "consider trimming before the report.")
    if gain is not None and gain >= SELL_STRENGTH_GAIN:
        out.append(f"Up {gain * 100:.0f}% — consider selling part into strength.")
    if pos.get("below_sma50"):
        vr = pos.get("volume_ratio")
        heavy = " on heavy volume" if (vr is not None and vr >= HEAVY_VOL_RATIO) else ""
        out.append(f"Closed below the 50-day SMA{heavy} — exit signal.")
    if (gain is not None and gain >= BREAKEVEN_GAIN and avg_entry
            and (cur_stop is None or cur_stop < avg_entry)):
        out.append("Up ≥ 2× initial risk — raise stop to at least breakeven.")
    return out


def _trading_days_since(last, today) -> Optional[int]:
    """Number of *trading* days between a bar date ``last`` and ``today`` — 0 when ``last`` is
    the freshest possible bar (e.g. Friday's bar read on the weekend, or today's own bar).
    Business-day based; holidays aren't modelled, so a holiday reads as one extra day and the
    caller's tolerance absorbs it. Returns None if either date can't be parsed, so the caller
    skips the check rather than blocking a trade on a parse error."""
    try:
        import pandas as pd
        a = pd.Timestamp(last).normalize()
        b = pd.Timestamp(today).normalize()
        if b <= a:
            return 0
        return len(pd.bdate_range(a, b)) - 1
    except Exception:
        return None


def sell_pillars(pos: dict, *, entry_date=None, pivot=None, regime=None,
                 spy_note=None, today=None) -> dict:
    """The sell doctrine's four thesis pillars for ONE holding. Pure, display-only.

    Any pillar failing kills the trade — the stop is only the disaster floor for what
    happens between checks. Reads SETTLED closes (``pos["last_close"]`` / ``pos["df"]``),
    never the live print. Tolerates missing inputs everywhere: each pillar degrades to
    ``unknown`` rather than raising, and a bare position dict (no new keys) yields four
    unknowns. Returns ``{"P1".."P4": {"status": "ok"|"warn"|"fail"|"unknown",
    "detail": str}}``.

    * P1 breakout holding — needs ``entry_date`` (the journal open-episode's first buy;
      tz-aware timestamps are read in exchange time) and ideally ``pivot`` (the
      watchlist's frozen level; without it only the laggard clock runs). Day-0 close
      back below the pivot, a decisive close below it (>2% or a 2nd consecutive), or a
      close below the breakout bar's low fail outright; a stalled clock warns at day
      ``P1_CUSHION_DAYS`` without a ``P1_CUSHION_PCT`` cushion and fails flat-to-red at
      day ``P1_STALL_DAYS``.
    * P2 template — STRICT: anything under 8/8 fails (user decision; expect occasional
      one-day red flips when a knife-edge SMA criterion wobbles).
    * P3 tape — the scan regime dict when available, else the trigger report's SPY-only
      read (partial: ok/warn), else unknown.
    * P4 earnings — inside the ``EARNINGS_SOON_DAYS`` window a loss or a thin cushion
      (< ``EARNINGS_CUSHION_MIN``) fails; a real cushion still warns (trim to
      hold-through size). ``earnings_in`` None reads unknown — "no report scheduled"
      and "data missing" are indistinguishable upstream."""
    import pandas as pd

    def _pill(status, detail):
        return {"status": status, "detail": detail}

    # ---- P1: the breakout itself -------------------------------------------------- #
    last_close = pos.get("last_close")
    gain = pos.get("gain_pct")
    df = pos.get("df")
    if entry_date is None:
        p1 = _pill("unknown", "no journal episode — refresh the Journal page")
    else:
        try:
            e = pd.Timestamp(entry_date)
            if e.tzinfo is not None:
                e = e.tz_convert("America/New_York").tz_localize(None)
            e = e.normalize()
        except Exception:
            e = None
        if e is None:
            p1 = _pill("unknown", "unreadable entry date")
        else:
            if today is None:
                t_now = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
            else:
                t_now = pd.Timestamp(today).normalize()
            day_n = _trading_days_since(e, t_now)
            fails: List[str] = []
            warns: List[str] = []
            if pivot and last_close is not None:
                if day_n == 0 and last_close < pivot:
                    fails.append("Day-0 close back below the pivot — the breakout "
                                 "never happened; sell next open")
                if last_close < pivot * (1.0 - DECISIVE_BELOW_PIVOT_PCT):
                    fails.append(f"closed {(1 - last_close / pivot) * 100:.1f}% below "
                                 "the pivot — decisive break")
                elif (df is not None and len(df) >= 2
                        and float(df["Close"].iloc[-1]) < pivot
                        and float(df["Close"].iloc[-2]) < pivot):
                    fails.append("second consecutive close below the pivot")
            if df is not None and len(df) and last_close is not None:
                try:
                    bo = df[df.index <= e]
                    if len(bo):
                        bo_low = float(bo["Low"].iloc[-1])
                        if last_close < bo_low:
                            fails.append("closed below the breakout bar's low — "
                                         "no grace day")
                except Exception:
                    pass
            if day_n is not None and gain is not None:
                if day_n >= P1_STALL_DAYS and gain <= 0:
                    fails.append(f"day {day_n} and flat-to-red — exit")
                elif day_n >= P1_CUSHION_DAYS and gain < P1_CUSHION_PCT:
                    warns.append(f"day {day_n}, no {P1_CUSHION_PCT * 100:.0f}% cushion "
                                 "— sell into strength")
            note = "" if pivot else " (no frozen pivot — clock only)"
            if fails:
                p1 = _pill("fail", "; ".join(fails) + note)
            elif warns:
                p1 = _pill("warn", "; ".join(warns) + note)
            elif day_n is None and last_close is None:
                p1 = _pill("unknown", "no bars to judge" + note)
            else:
                p1 = _pill("ok", f"day {day_n}, holding{note}")

    # ---- P2: Stage-2 structure ---------------------------------------------------- #
    tc = pos.get("template_criteria")
    if tc is None:
        p2 = _pill("unknown", "no template read (no bars)")
    elif int(tc) >= 8:
        p2 = _pill("ok", "8/8 trend template")
    else:
        p2 = _pill("fail", f"{int(tc)}/8 — template broken")

    # ---- P3: the tape ------------------------------------------------------------- #
    if isinstance(regime, dict) and regime.get("regime") is not None:
        if regime.get("should_generate_buys"):
            p3 = _pill("ok", str(regime.get("regime")))
        else:
            p3 = _pill("fail", f"{regime.get('regime')} — risk-off: demote yellow "
                               "flags to red, laggards go first")
    elif isinstance(spy_note, dict) and spy_note.get("trend"):
        trend = str(spy_note["trend"])
        if trend.lower().startswith("bull"):
            p3 = _pill("ok", f"SPY {trend} (trigger-report read, no breadth)")
        else:
            p3 = _pill("warn", f"SPY {trend} (trigger-report read, no breadth)")
    else:
        p3 = _pill("unknown", "no scan or trigger report yet")

    # ---- P4: the earnings window -------------------------------------------------- #
    ei = pos.get("earnings_in")
    if ei is None:
        p4 = _pill("unknown", "no earnings date")
    elif ei < 0:
        p4 = _pill("ok", f"reported {-int(ei)}d ago")
    elif ei > EARNINGS_SOON_DAYS:
        p4 = _pill("ok", f"next report {int(ei)}d out")
    elif gain is None:
        p4 = _pill("warn", f"report in {int(ei)}d — cushion unknown")
    elif gain < 0:
        p4 = _pill("fail", f"report in {int(ei)}d with a loss — a loss is never "
                           "carried into a report")
    elif gain < EARNINGS_CUSHION_MIN:
        p4 = _pill("fail", f"report in {int(ei)}d, cushion {gain * 100:.0f}% < "
                           f"{EARNINGS_CUSHION_MIN * 100:.0f}% — trim/exit first")
    else:
        p4 = _pill("warn", f"report in {int(ei)}d — trim to hold-through size")

    return {"P1": p1, "P2": p2, "P3": p3, "P4": p4}


def _stop_only_entry(t, price, pivot, frozen, buy_hi, stop, payload) -> dict:
    """A zero-share plan row for a HELD name whose buy failed the sizing gates: submit's
    held path sends no buy anyway and only re-arms the GTC stop, so this row exists purely
    to carry ``stop_price`` there instead of the name silently losing stop maintenance."""
    return {"ticker": t, "shares": 0, "price": round(price, 2),
            "pivot": round(float(pivot), 2) if pivot else None,
            "pivot_frozen": bool(frozen),
            "est_value": 0.0,
            "extended": bool(buy_hi and price > buy_hi),
            "capped": False, "stop_only": True,
            "stop_price": round(float(stop), 2),
            "limit_price": None,                # stop re-arm only — no buy, no limit
            "earnings_in": payload.get("earnings_in")}


def build_buy_plan(tickers: Sequence[str], payloads: Dict[str, dict], *,
                   mode: str, amount: float,
                   equity: Optional[float] = None, asof=None,
                   max_bar_age_days: Optional[float] = None,
                   pivots: Optional[Dict[str, float]] = None,
                   held: Optional[Dict[str, int]] = None,
                   order_type: str = "market") -> Tuple[List[dict], List[dict]]:
    """Size a BUY for EACH watchlisted name by the chosen ``mode``:

    * ``"pct"``     — ``amount`` % of the account ``equity`` per name (needs ``equity``);
    * ``"dollars"`` — ``amount`` dollars per name;
    * ``"shares"``  — exactly ``amount`` (whole) shares per name;
    * ``"risk"``    — ``amount`` % of ``equity`` risked to the stop (needs ``equity`` + a stop):
      ``shares = (equity × amount%) / (price − stop)`` — Minervini's position sizer, so a
      stop-out costs ≈ ``amount``% of the account. Sized on the current price (the real fill),
      not the pivot, so the risk figure is honest for the order actually sent.

    Returns ``(plan, skipped)``. Each plan entry has ticker / shares / price / pivot /
    est_value / extended / stop_price / capped; each skipped entry is ``{ticker, reason}``. A
    name is skipped when it isn't in the current scan, has no current price, sizes to < 1 share,
    or (for every dollar-denominated mode — pct/dollars/risk) rounds to a notional under the $50
    floor; the ``"shares"`` mode is exempt from that floor since the count is explicit. The
    ``"risk"`` mode additionally skips a name with no stop, or a stop not below the price (a
    non-positive risk-per-share). ``extended`` flags a price already above the no-chase buy zone
    (> pivot × 1.05); the caller surfaces it as a warning rather than skipping. ``capped`` is
    True only in ``"risk"`` mode when the risk-sized quantity would exceed the 10%-of-equity
    single-order cap and was clamped down to it (so the realized risk falls BELOW the target —
    the caller labels it); it's always False for the other modes. ``stop_price`` is the
    app-computed protective stop (``levels["stop"]``, ~7-8% below pivot) or ``None`` if
    unavailable — the caller may edit it before submit; note that editing it after build does
    NOT re-scale a risk-sized quantity. ``earnings_in`` (calendar days to the next scheduled
    report, from the scan payload; None = unknown) is carried through untouched so the caller
    can warn about buying into an imminent report — advisory only, never a skip.

    When ``max_bar_age_days`` is set, a name whose freshest bar is more than that many
    *trading* days old (relative to ``asof`` or today) is skipped as stale rather than sized
    on days-old data — the app pairs this with :func:`freshen_prices` at Build so ordinary
    names read fresh. Both default to off, keeping the builder pure and unchanged for callers
    (and unit tests) that omit them.

    ``pivots`` maps ticker -> a FROZEN judged_pivot (the level the watchlist trigger fired on).
    The detected scan pivot drifts every scan, so for a name with a frozen pivot the buy zone,
    ``extended`` flag, default ``stop_price``, and risk sizing all key off the frozen level
    instead of the current payload's ``levels`` (default stop 7.5% below it, hard-floored 10%
    below — mirroring ``scan._entry_levels``; a tighter engine stop below the frozen pivot is
    kept). Each plan entry carries ``pivot_frozen`` (True when its pivot came from ``pivots``).
    Omit ``pivots`` and every name uses its scan-derived levels as before.

    ``held`` (optional ``{ticker: shares}``) closes the build-time stop gap: a HELD
    name whose buy fails a sizing gate (rounds < 1 share, or under the $50 floor) is
    emitted as a zero-share ``stop_only=True`` row instead of skipped, so submit's held
    path still re-arms its protective stop. Names skipped before levels exist (not in
    scan / no price / stale) or without a computable stop still skip — there's no level
    to arm. Omit ``held`` (the default) and behavior is unchanged.

    ``order_type="limit"`` plans limit BUYs instead of market: each entry gets a
    ``limit_price`` defaulting to its buy-zone TOP (effective pivot × 1.05 — the no-chase
    cap, so a name that gaps past the zone simply doesn't fill; a name with no pivot falls
    back to the last close, a marketable cap). Sizing, the risk-per-share, the $50 floor,
    and the 10% cap all use the LIMIT as the basis — the worst-case fill for a buy limit
    is the limit itself (fills lower, never higher) — so ``est_value`` is the honest
    maximum. The risk mode requires ``stop < limit``. ``"market"`` (the default) leaves
    every existing behavior byte-identical; ``limit_price`` is then ``None``.
    """
    if mode not in SIZING_MODES:
        raise ValueError(f"mode must be one of {SIZING_MODES}, got {mode!r}")
    if order_type not in ("market", "limit"):
        raise ValueError(f"order_type must be 'market' or 'limit', got {order_type!r}")
    _stale_ref = None
    if max_bar_age_days is not None:
        import pandas as pd
        _stale_ref = (pd.Timestamp(asof).normalize() if asof is not None
                      else pd.Timestamp.today().normalize())
    plan: List[dict] = []
    skipped: List[dict] = []
    for t in dict.fromkeys(tickers):
        payload = payloads.get(t)
        if not payload:
            skipped.append({"ticker": t, "reason": "not in the current scan"})
            continue
        lv = payload.get("levels", {}) or {}
        df = payload.get("df")
        price = (float(df["Close"].iloc[-1])
                 if df is not None and len(df) else None)
        if not price or price <= 0:
            skipped.append({"ticker": t, "reason": "no current price"})
            continue
        if _stale_ref is not None:
            _age = _trading_days_since(df.index[-1], _stale_ref)
            if _age is not None and _age > max_bar_age_days:
                skipped.append({"ticker": t, "reason":
                                f"stale price data (last bar "
                                f"{pd.Timestamp(df.index[-1]).date()}, ~{_age} trading days "
                                f"old) — Re-scan to refresh"})
                continue

        # Effective entry levels (read early — the risk mode sizes against the stop). A frozen
        # judged_pivot overrides the drifted scan pivot: buy zone, extended flag, and the default
        # stop all key off it, so the order the user submits matches the level the trigger fired on.
        frozen = None
        if pivots:
            _fp = pivots.get(t)
            frozen = float(_fp) if _fp and _fp > 0 else None
        if frozen:
            pivot, buy_hi = frozen, frozen * 1.05
            _eng = lv.get("stop")                        # keep a tighter engine stop below the pivot
            _raw = (_eng if (_eng and _eng > 0 and _eng < frozen)
                    else frozen * (1.0 - DEFAULT_STOP_FROM_PIVOT))
            stop = max(_raw, frozen * (1.0 - MAX_STOP_FROM_PIVOT))
        else:
            bz = lv.get("buy_zone") or (None, None)
            pivot, buy_hi = bz[0], bz[1]
            stop = lv.get("stop")
        capped = False

        # Sizing basis: the worst-case fill. Market orders fill ~at the current price; a buy
        # LIMIT fills at or below its limit, so the limit itself is the honest basis for
        # share counts, the $50 floor, and the 10% cap.
        limit = None
        basis = price
        if order_type == "limit":
            # A stop AT/ABOVE the current price means the base broke down below its
            # pivot: the zone-top limit would be MARKETABLE, fill ~at the price, and the
            # OTO stop leg would arm above the market — an instant stop-out. The market
            # path rejects the same numbers at submit; reject here at the source.
            if stop and stop > 0 and price and stop >= price:
                skipped.append({"ticker": t, "reason":
                                "stop not below the current price — the base has broken "
                                "down below its pivot; re-judge or remove the name"})
                continue
            limit = float(buy_hi) if buy_hi and buy_hi > 0 else price
            basis = limit

        if mode == "pct":
            if not equity or equity <= 0:
                skipped.append({"ticker": t, "reason": "account equity unavailable"})
                continue
            shares = int((equity * amount / 100.0) / basis)          # floor
        elif mode == "dollars":
            shares = int(amount / basis)                             # floor
        elif mode == "risk":
            if not equity or equity <= 0:
                skipped.append({"ticker": t, "reason": "account equity unavailable"})
                continue
            if not stop or stop <= 0:
                skipped.append({"ticker": t, "reason": "no stop to risk-size against"})
                continue
            if stop >= basis:
                skipped.append({"ticker": t, "reason":
                                f"stop not below {'limit' if limit else 'price'} — "
                                "can't risk-size"})
                continue
            shares = int((equity * amount / 100.0) / (basis - stop))  # floor
            # Risk sizing yields position% ≈ risk% / stop-distance%, which routinely exceeds the
            # 10% single-order cap (1% risk / 8% stop = 12.5%). Clamp to the cap rather than skip;
            # the realized risk then sits below target and the caller flags it via ``capped``.
            cap_shares = int(MAX_ORDER_PCT * equity / basis)
            if shares > cap_shares:
                shares, capped = cap_shares, True
        else:                                                        # "shares"
            shares = int(amount)

        # A held name failing a sizing gate still needs its stop maintained — emit a
        # stop-only row (shares=0) instead of dropping it (needs a valid stop).
        _held_fallback = bool(held and held.get(t, 0) > 0 and stop and stop > 0)
        if shares < 1:
            if _held_fallback:
                plan.append(_stop_only_entry(t, price, pivot, frozen, buy_hi, stop, payload))
            else:
                skipped.append({"ticker": t, "reason": "sizing rounds to < 1 share"})
            continue
        est_value = shares * basis
        if mode != "shares" and est_value < MIN_TRADE_USD:
            if _held_fallback:
                plan.append(_stop_only_entry(t, price, pivot, frozen, buy_hi, stop, payload))
            else:
                skipped.append({"ticker": t,
                                "reason": f"under the ${MIN_TRADE_USD:.0f} order minimum"})
            continue

        plan.append({
            "ticker": t, "shares": shares, "price": round(price, 2),
            "pivot": round(float(pivot), 2) if pivot else None,
            "pivot_frozen": bool(frozen),
            "est_value": round(est_value, 2),
            "extended": bool(buy_hi and price > buy_hi),
            "capped": capped,
            "stop_price": round(float(stop), 2) if stop and stop > 0 else None,
            "limit_price": round(limit, 2) if limit else None,
            "earnings_in": payload.get("earnings_in"),
        })
    return plan, skipped


def freshen_prices(tickers: Sequence[str], payloads: Dict[str, dict]) -> Dict[str, dict]:
    """Re-pull the latest daily bars for the given watchlist ``tickers`` and return a small
    payloads dict with each name's ``df`` replaced by the freshest available frame, so
    :func:`build_buy_plan` sizes on current prices instead of the possibly days-old closes
    frozen in the scan memo (the scan result lives in session_state with no time-based
    invalidation and the trigger fragment keeps the tab alive for days).

    Uses the cheap incremental top-up (``max_age_days=0`` — the same path refresh_job
    uses to fetch the finalized close without a full 2y refetch); only the handful of
    watchlist names are fetched, never the universe. Any name the refresh can't reach keeps
    its existing frame — :func:`build_buy_plan`'s ``max_bar_age_days`` guard then skips a
    genuinely stale one. Tickers absent from ``payloads`` are dropped (the builder already
    reports those as 'not in the current scan'). ``levels`` and ``earnings_in`` are carried
    through untouched; only the price frame is refreshed. ``data_feed`` is imported lazily so
    the cockpit still loads without its optional deps, and a total fetch failure degrades to
    the original frames rather than raising."""
    want = [t for t in dict.fromkeys(tickers) if payloads.get(t)]
    if not want:
        return {}
    try:
        from . import data_feed
        fresh = data_feed.get_many_prices(want, max_age_days=0.0) or {}
    except Exception:
        fresh = {}
    out: Dict[str, dict] = {}
    for t in want:
        f = fresh.get(t)
        out[t] = {**payloads[t], "df": f} if (f is not None and len(f)) else payloads[t]
    return out


def _open_sell_stops(client, ticker: str, *, GetOrdersRequest, QueryOrderStatus,
                     OrderSide, OrderType) -> List:
    """This ticker's OPEN sell **stop** orders (STOP / STOP_LIMIT / TRAILING_STOP).

    A manual limit sell isn't a stop, so it's excluded. Alpaca surfaces a triggered OTO stop
    leg as its own top-level SELL order too, so this flat query catches both standalone stops
    and prior OTO legs. Returns the order objects (each carries a ``stop_price``) so the caller
    can read the current stop level for the ratchet AND cancel them when raising.
    """
    stop_types = {OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP}
    try:
        opens = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN, side=OrderSide.SELL, symbols=[ticker]))
    except Exception:
        return []
    out = []
    for od in opens or []:
        if getattr(od, "symbol", None) != ticker:
            continue
        otype = getattr(od, "type", None) or getattr(od, "order_type", None)
        if otype in stop_types:
            out.append(od)
    return out


def _stop_price_of(order) -> Optional[float]:
    """Best-effort read of an order's stop trigger price (alpaca-py ``Order.stop_price``) as a
    float, or None if absent/unparseable. Real stops always carry one; None just means we
    can't compare, so the caller falls back to replacing rather than ratcheting."""
    v = getattr(order, "stop_price", None)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _order_qty(order) -> int:
    """Whole-share qty of an order (alpaca-py ``Order.qty``), or 0 if absent/unparseable — used
    to check whether the in-force stop(s) still cover the whole position after it grew."""
    v = getattr(order, "qty", None)
    if v is None:
        return 0
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _cancel_orders(client, orders) -> List[str]:
    """Cancel each order by id, independently guarded so one stuck order never blocks the rest.
    Returns the cancelled ids."""
    cancelled: List[str] = []
    for od in orders or []:
        try:
            client.cancel_order_by_id(od.id)
            cancelled.append(str(getattr(od, "id", "?")))
        except Exception:
            pass
    return cancelled


def _open_sell_stops_by_symbol(client, *, GetOrdersRequest, QueryOrderStatus,
                               OrderSide, OrderType) -> Dict[str, List]:
    """ALL open sell STOP/STOP_LIMIT/TRAILING_STOP orders in ONE query, grouped by symbol.

    Same type filter as :func:`_open_sell_stops` but omits the ``symbols=`` filter, so the whole
    account's protective stops come back in a single round-trip (the positions page needs every
    symbol's stop at once). Returns ``{symbol: [order, ...]}``; empty dict on any error."""
    stop_types = {OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP}
    try:
        opens = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN, side=OrderSide.SELL))
    except Exception:
        return {}
    out: Dict[str, List] = {}
    for od in opens or []:
        sym = getattr(od, "symbol", None)
        if not sym:
            continue
        otype = getattr(od, "type", None) or getattr(od, "order_type", None)
        if otype in stop_types:
            out.setdefault(sym, []).append(od)
    return out


def _open_cockpit_buy_orders(client, *, GetOrdersRequest, QueryOrderStatus,
                             OrderSide) -> list:
    """OPEN cockpit BUY orders (``client_order_id`` starts ``SEPA``) in ONE query — the
    order OBJECTS, for callers that need ids (cancel) as well as symbols (skip). Side=BUY
    already excludes ``SEPAstop-`` sells. Empty list on any error (fail-open)."""
    try:
        opens = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN, side=OrderSide.BUY))
    except Exception:
        return []
    return [od for od in (opens or [])
            if getattr(od, "symbol", None)
            and str(getattr(od, "client_order_id", None) or "").startswith("SEPA")]


def _open_cockpit_buys(client, *, GetOrdersRequest, QueryOrderStatus, OrderSide) -> set:
    """Symbols with an OPEN cockpit BUY order.

    The documented cadence submits after the close, so a queued BUY (or a resting GTC
    limit) has no position yet — :func:`submit_buy_plan`'s only 'already invested'
    guard is ``get_all_positions()``, which wouldn't see it. Skipping these on a
    re-submit prevents a second BUY (double position, double risk)."""
    return {od.symbol for od in _open_cockpit_buy_orders(
        client, GetOrdersRequest=GetOrdersRequest, QueryOrderStatus=QueryOrderStatus,
        OrderSide=OrderSide)}


def cancel_pending_buys() -> dict:
    """Cancel every OPEN cockpit BUY order (``SEPA…`` tags only — never sells, stops, or
    other tools' orders). THE control for a resting GTC limit whose setup broke (limits
    rest until filled or canceled; the pending-buy guard blocks re-submits but cannot
    cancel). Canceling an unfilled OTO parent cancels its held stop leg too — nothing
    was bought, so there is nothing left to protect.

    Returns ``{"cancelled": [{ticker, id}], "errors": [{ticker, id, error}]}``; one
    failed cancel never aborts the rest. Raises :class:`TradeUnavailable` only for
    missing package/credentials (the panel catches it)."""
    client, _using = _connect_paper()
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import OrderSide, QueryOrderStatus
    except ImportError as e:
        raise TradeUnavailable(str(e)) from e
    cancelled, errors = [], []
    for od in _open_cockpit_buy_orders(client, GetOrdersRequest=GetOrdersRequest,
                                       QueryOrderStatus=QueryOrderStatus,
                                       OrderSide=OrderSide):
        ref = {"ticker": od.symbol, "id": str(getattr(od, "id", "?"))}
        try:
            client.cancel_order_by_id(od.id)
            cancelled.append(ref)
        except Exception as e:                  # one stuck order shouldn't abort the rest
            errors.append({**ref, "error": str(e)})
    return {"cancelled": cancelled, "errors": errors}


def _rearm_gtc_stop(client, symbol: str, held_shares: int, desired_stop, price, existing, *,
                    OrderSide, TimeInForce, StopOrderRequest) -> dict:
    """Minervini's one-way GTC stop ratchet for a held position — the single source of truth,
    called by both :func:`submit_buy_plan` (held-name branch) and :func:`rearm_stops`.

    ``existing`` is that symbol's open sell-stop orders (the caller fetches them, per-ticker or
    batched). The current in-force stop is ``max(_stop_price_of(existing))``. Returns a PARTIAL
    result dict — ``{status, detail}`` plus ``stop_price`` when a level is set — where status is
    ``"stop_only"`` (placed/raised), ``"stop_kept"`` (existing kept — a would-be lower/equal or
    an invalid new stop), ``"skipped"`` (no valid stop and none in force) or ``"failed"``
    (replacement rejected). NEVER lowers a stop: it only cancels + replaces to RAISE. GTC so
    the stop persists across sessions.

    Cancel-before-place is guarded (same pattern as
    :func:`submit_position_sell`): if the replacement submit fails AFTER the old stop was
    cancelled, the previous stop is RESTORED at its old level for the full held quantity —
    the position is never silently left unprotected. A failed restore is loudly reported."""
    prices = [p for p in (_stop_price_of(od) for od in existing) if p is not None]
    cur = max(prices) if prices else None                # current stop level, if any
    new_stop = round(float(desired_stop), 2) if desired_stop else None

    if not stop_is_valid(new_stop, price):
        # New stop isn't below the price. If a valid stop is already in force the position stays
        # protected — keep it; otherwise there's nothing to place.
        if cur is not None:
            return {"status": "stop_kept", "stop_price": cur,
                    "detail": f"kept existing stop @ {cur:.2f} (new stop not below price)"}
        return {"status": "skipped",
                "detail": "no valid stop (must be > 0 and < current price)"}

    err = {}

    def _try_place(level):
        try:
            return client.submit_order(order_data=StopOrderRequest(
                symbol=symbol, qty=held_shares, side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC, stop_price=level,
                client_order_id=f"SEPAstop-{symbol}-{int(time.time() * 1000)}"))
        except Exception as e:
            err["msg"] = str(e)
            return None

    def _failed(level, cancelled):
        detail = f"stop placement FAILED @ {level:.2f}: {err.get('msg', '?')}"
        out = {"status": "failed", "detail": detail}
        if cancelled and cur is not None:                # one restore attempt, never a loop
            if _try_place(cur) is not None:
                out["stop_price"] = cur
                out["detail"] += f"; previous stop restored @ {cur:.2f}"
            else:
                out["detail"] += "; stop restore FAILED — arm a stop manually"
        return out

    # Ratchet: only replace to RAISE the stop; a lower-or-equal one is kept — UNLESS the in-force
    # stop under-covers the position (it grew via a manual pyramid buy Alpaca-side). Then re-place
    # at the SAME (never-lower) level for the full held qty so the added shares aren't left
    # unprotected while the UI reports a stop in force. Only acts on positive evidence of
    # under-coverage (0 < covered < held), so an unreadable qty never triggers needless churn.
    if cur is not None and new_stop <= cur:
        covered = sum(_order_qty(od) for od in existing)
        if not (0 < covered < held_shares):
            return {"status": "stop_kept", "stop_price": cur,
                    "detail": f"kept existing stop @ {cur:.2f} — not lowering to {new_stop:.2f}"}
        cancelled = _cancel_orders(client, existing)     # under-covered -> re-place at cur, full qty
        resp = _try_place(cur)
        if resp is None:
            return _failed(cur, cancelled)
        return {"status": "stop_only", "stop_price": cur,
                "detail": f"GTC stop re-placed for full {held_shares} sh @ {cur:.2f} "
                          f"(was {covered} sh; id {getattr(resp, 'id', '?')})"}

    cancelled = _cancel_orders(client, existing)         # replace the lower stop(s)
    resp = _try_place(new_stop)
    if resp is None:
        return _failed(new_stop, cancelled)
    verb = "raised" if cur is not None else "placed"
    detail = (f"GTC stop {verb}: SELL {held_shares} @ {new_stop:.2f} "
              f"(id {getattr(resp, 'id', '?')})")
    if cur is not None:
        detail += f" (was {cur:.2f})"
    return {"status": "stop_only", "stop_price": new_stop, "detail": detail}


def submit_buy_plan(plan: List[dict], *, attach_stop: bool = True) -> dict:
    """Submit each planned order on the cockpit's Alpaca **paper** account (Minervini Trader
    keys preferred — see :func:`_connect_paper`), attaching a protective stop when
    ``attach_stop`` is set.

    Per name:

    * **already held** in the account — no buy is sent; a **GTC** sell-stop protects the WHOLE
      held position, managed as Minervini's one-way ratchet (never lower a stop, only raise it):
      if no stop is open it's placed at ``stop_price``; if one already is, it's replaced only to
      RAISE it — a would-be lower-or-equal stop is left untouched (result ``"stop_kept"``). GTC
      so it persists across sessions instead of expiring each close. Exempt from the $50 floor /
      10%-cap since a protective stop is risk-reducing.
    * **not held** — a BUY. With ``attach_stop`` it's an OTO order carrying a stop-loss
      leg (the stop activates only after the buy fills, so it works even when the buy is queued
      to the next open). The whole OTO is **GTC** end-to-end: the stop leg is held until
      the buy fills, then rests as a GTC stop that SURVIVES the close — a DAY leg would
      expire at that day's close, leaving an intraday fill unprotected overnight. The
      held-name ratchet manages (only ever raises) the same stop from the next re-arm
      on. Without ``attach_stop``, a plain market BUY.

      An entry carrying a positive ``limit_price`` (a ``build_buy_plan(order_type="limit")``
      plan) is sent as a **limit** BUY instead: with ``attach_stop`` a GTC OTO limit + stop
      leg — same GTC-end-to-end shape, so a fill on ANY later day is protected the
      moment it happens, and an unfilled limit rests until filled or canceled (the pending-buy
      guard blocks re-submits meanwhile); without ``attach_stop`` a DAY limit that expires at
      the close. Stop validity is checked against the worst-case fill. No ``limit_price``
      → the market behavior above, unchanged.

      **Build-time intent is binding:** an entry stamped ``rearm_only`` (held when the plan
      was built — the preview showed it with no checkbox and "no buy") or ``stop_only``
      (zero-share stop carrier), whose position has since closed, is SKIPPED — never
      converted into a buy the user did not consent to. Rebuild the plan to buy such a name.

    Reuses ``alpaca_trader``'s tradability check and 10%-of-equity order cap (buys only).
    Returns ``{equity, cash, account_number, using_dedicated, results}`` where each result is
    the plan entry plus a ``status`` ("submitted" / "stop_only" / "stop_kept" / "skipped" /
    "failed") and a ``detail`` string. Raises :class:`TradeUnavailable` if alpaca-py or
    credentials are missing.
    """
    client, using_dedicated = _connect_paper()          # paper=True enforced inside
    try:
        from alpaca.trading.requests import (
            MarketOrderRequest, LimitOrderRequest, StopOrderRequest, StopLossRequest,
            GetOrdersRequest)
        from alpaca.trading.enums import (
            OrderSide, TimeInForce, OrderClass, QueryOrderStatus, OrderType)
        from src.portfolio_experimentation.alpaca_trader import (
            validate_tradable, MAX_ORDER_PCT)
    except ImportError as e:                             # alpaca-py present for _connect but
        raise TradeUnavailable(str(e)) from e            # a submodule/name somehow isn't

    acct = client.get_account()
    equity, cash = float(acct.equity), float(acct.cash)
    account_number = getattr(acct, "account_number", "?")
    # {ticker: whole shares held} — mirrors get_account_state's int(float(qty)) convention.
    held = {p.symbol: int(float(p.qty)) for p in client.get_all_positions()}
    # Cockpit BUYs still queued (submitted after the close, not yet filled) don't show as
    # positions — skip a re-buy on those so a re-submit can't double the position.
    pending_buys = _open_cockpit_buys(
        client, GetOrdersRequest=GetOrdersRequest,
        QueryOrderStatus=QueryOrderStatus, OrderSide=OrderSide)
    tradable = validate_tradable(client, [o["ticker"] for o in plan])
    max_allowed = MAX_ORDER_PCT * equity

    results: List[dict] = []
    for o in plan:
        t = o["ticker"]
        stop = o.get("stop_price")
        held_shares = held.get(t, 0)
        if t not in tradable:
            results.append({**o, "status": "skipped",
                            "detail": "not tradable on Alpaca"})
            continue
        try:
            if held_shares > 0:
                # Already invested — no buy; manage a GTC protective stop for the whole position
                # via the shared one-way ratchet (never lower, only raise — see _rearm_gtc_stop).
                if not attach_stop:
                    results.append({**o, "status": "skipped",
                                    "detail": f"already held ({held_shares} sh); "
                                              "stop attach disabled"})
                    continue
                existing = _open_sell_stops(
                    client, t, GetOrdersRequest=GetOrdersRequest,
                    QueryOrderStatus=QueryOrderStatus, OrderSide=OrderSide, OrderType=OrderType)
                res = _rearm_gtc_stop(client, t, held_shares, stop, o["price"], existing,
                                      OrderSide=OrderSide, TimeInForce=TimeInForce,
                                      StopOrderRequest=StopOrderRequest)
                results.append({**o, **res})
                continue

            # Not held — a BUY (market, or limit when the entry carries a limit_price),
            # with an OTO protective stop when attach_stop is on.
            #
            # Build-time-intent guard: a row the plan preview showed as "already held —
            # stop re-arm only, no buy" (``rearm_only``, stamped from BUILD-time
            # holdings) or a zero-share ``stop_only`` row must NEVER convert into a buy
            # just because the position closed between Build and Submit (its GTC stop
            # firing is enough) — the user was shown no checkbox and consented to no
            # buy. Same guard kills qty<1 rows before they reach the API as noise.
            if o.get("rearm_only") or o.get("stop_only") or int(o.get("shares", 0)) < 1:
                results.append({**o, "status": "skipped",
                                "detail": "position closed since the plan was built — "
                                          "no buy sent (rebuild the plan to buy it)"})
                continue
            # Server-side backstop for the progressive-exposure gate: the panel stamps
            # blocked buy rows so a stale client can't slip one through.
            if o.get("gate_blocked"):
                results.append({**o, "status": "skipped",
                                "detail": "progressive-exposure gate closed — no buy"})
                continue
            if t in pending_buys:
                results.append({**o, "status": "skipped",
                                "detail": "a cockpit BUY is already queued (pending fill) — "
                                          "not re-submitting"})
                continue
            _lim = o.get("limit_price")
            # None = a market plan; a PRESENT but non-positive limit is an edit error — skip
            # rather than silently falling back to an uncapped market buy.
            if _lim is not None and not (float(_lim) > 0):
                results.append({**o, "status": "skipped",
                                "detail": "invalid limit price — set a limit > 0"})
                continue
            limit = float(_lim) if _lim else None
            # The 10% cap binds on the worst-case fill: for a limit row RECOMPUTE from
            # the (possibly user-EDITED) limit — the entry's est_value is build-time and
            # an upward edit would otherwise slip past the cap. Market rows keep the
            # build value (the price isn't editable).
            _est = o["shares"] * limit if limit else o["est_value"]
            if _est > max_allowed:
                results.append({**o, "status": "skipped",
                                "detail": f"exceeds 10% of equity (${max_allowed:,.0f} cap)"})
                continue
            if attach_stop:
                # Stop validity is against the WORST fill either way: a limit BUY can
                # fill anywhere at or below the limit — including ~the current price
                # when the limit is marketable — so the stop must clear BOTH.
                _worst = min(limit, o["price"]) if limit else o["price"]
                if not stop_is_valid(stop, _worst):
                    results.append({**o, "status": "skipped",
                                    "detail": "stop not below entry — fix stop or turn off "
                                              "Attach stop"})
                    continue
                # GTC OTO: the stop leg inherits GTC ("held" until the fill), so the
                # protective stop persists past the close — a DAY leg expires AT the
                # close and can leave an intraday fill unprotected overnight. The same
                # shape covers a limit that fills days later: its stop arms on the fill.
                if limit:
                    req = LimitOrderRequest(
                        symbol=t, qty=o["shares"], side=OrderSide.BUY,
                        limit_price=round(limit, 2),
                        time_in_force=TimeInForce.GTC, order_class=OrderClass.OTO,
                        stop_loss=StopLossRequest(stop_price=round(float(stop), 2)),
                        client_order_id=f"SEPAoto-{t}-{int(time.time() * 1000)}")
                    detail_head = (f"limit buy {o['shares']} @ {limit:.2f} + stop @ "
                                   f"{stop:.2f} (GTC — rests until filled/canceled)")
                else:
                    req = MarketOrderRequest(
                        symbol=t, qty=o["shares"], side=OrderSide.BUY,
                        time_in_force=TimeInForce.GTC, order_class=OrderClass.OTO,
                        stop_loss=StopLossRequest(stop_price=round(float(stop), 2)),
                        client_order_id=f"SEPAoto-{t}-{int(time.time() * 1000)}")
                    detail_head = f"buy {o['shares']} + stop @ {stop:.2f}"
            else:
                if limit:
                    req = LimitOrderRequest(
                        symbol=t, qty=o["shares"], side=OrderSide.BUY,
                        limit_price=round(limit, 2),
                        time_in_force=TimeInForce.DAY,
                        client_order_id=f"SEPAcockpit-{t}-{int(time.time() * 1000)}")
                    detail_head = f"limit buy {o['shares']} @ {limit:.2f} (no stop, DAY)"
                else:
                    req = MarketOrderRequest(
                        symbol=t, qty=o["shares"], side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                        client_order_id=f"SEPAcockpit-{t}-{int(time.time() * 1000)}")
                    detail_head = f"buy {o['shares']} (no stop)"
            resp = client.submit_order(order_data=req)
            results.append({**o, "status": "submitted",
                            "detail": f"{detail_head} (id {getattr(resp, 'id', '?')})"})
        except Exception as e:                          # one bad symbol shouldn't abort the rest
            results.append({**o, "status": "failed", "detail": str(e)})

    return {"equity": equity, "cash": cash, "account_number": account_number,
            "using_dedicated": using_dedicated, "results": results}


def _attr_float(obj, attr: str) -> Optional[float]:
    """``getattr`` + float-coerce-or-None. alpaca-py ``Position`` P&L fields are read
    defensively — any absent/odd-typed field reads as None and the view renders without it."""
    v = getattr(obj, attr, None)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch_positions() -> dict:
    """Read the cockpit's Alpaca **paper** account (Minervini keys preferred) and return holdings
    enriched with P&L, in-force stop level, 50-day SMA, earnings date (for the cushion
    advisories), stage on the stop ladder, and Minervini exit advisories.

    Returns ``{"account": {account_number, equity, cash, using_dedicated, positions_count,
    total_unrealized_pl}, "positions": [ per-position dict ]}``. Raises :class:`TradeUnavailable`
    on missing package/credentials (the page catches it). Every per-position numeric field degrades
    to ``None`` (never raises) when a Position attribute is absent or price history is < 50 bars."""
    client, using_dedicated = _connect_paper()
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import OrderSide, QueryOrderStatus, OrderType
    except ImportError as e:
        raise TradeUnavailable(str(e)) from e

    acct = client.get_account()
    equity, cash = float(acct.equity), float(acct.cash)
    account_number = getattr(acct, "account_number", "?")
    raw = list(client.get_all_positions())
    stops_by_sym = _open_sell_stops_by_symbol(
        client, GetOrdersRequest=GetOrdersRequest, QueryOrderStatus=QueryOrderStatus,
        OrderSide=OrderSide, OrderType=OrderType)

    symbols = [s for s in (getattr(p, "symbol", None) for p in raw) if s]
    # One batched price pull for the 50-day SMA + SMA-cross advisory; lazy imports so trade.py
    # still loads without yfinance / the vendored screening package present.
    frames: dict = {}
    data_feed = None
    if symbols:
        try:
            from . import data_feed as _df_mod
            data_feed = _df_mod
            # NEVER queue the Positions page behind the bulk pipeline: _YF_LOCK
            # serializes every in-process download, so while a scan/refresh is mid-sweep
            # this small fetch would wait on it — and the page (with its SELL controls)
            # waits on this fetch. Selling needs only Alpaca data; current_price comes
            # from the Position objects either way, and the SMA-50/volume advisories
            # tolerate a cache bar.
            frames = data_feed.get_many_prices(
                symbols, allow_network=not data_feed.network_busy())
        except Exception:
            frames = {}
    try:
        from src.stock_screener.minervini_screener.screening import calculate_sma
    except Exception:
        calculate_sma = None
    import pandas as pd

    # Earnings dates for the cushion advisories (best-effort; weekly JSON cache per ticker,
    # serial — position counts are small and the page's cache_data absorbs the cost).
    earn: Dict[str, tuple] = {}                     # sym -> (next_earnings, earnings_in)
    try:
        from . import data_feed as _fund_mod
        from .scan import _days_to_earnings
        for sym in symbols:
            try:
                f = _fund_mod.get_fundamentals(sym)
                earn[sym] = ((f or {}).get("next_earnings"), _days_to_earnings(f))
            except Exception:
                pass
    except Exception:
        pass

    positions: List[dict] = []
    for p in raw:
        sym = getattr(p, "symbol", None)
        if not sym:
            continue
        try:
            qty = int(float(getattr(p, "qty", 0) or 0))
        except (TypeError, ValueError):
            qty = 0
        avg_entry = _attr_float(p, "avg_entry_price")
        price = _attr_float(p, "current_price")

        stop_prices = [q for q in (_stop_price_of(od) for od in stops_by_sym.get(sym, []))
                       if q is not None]
        current_stop = max(stop_prices) if stop_prices else None

        sma_50 = last_close = volume_ratio = None
        df = frames.get(sym)
        if df is None and data_feed is not None:
            df = frames.get(data_feed.normalize(sym))
        if df is not None and len(df):
            last_close = float(df["Close"].iloc[-1])
            if price is None:
                price = last_close
            if calculate_sma is not None and len(df) >= 50:
                s = calculate_sma(df["Close"], 50)
                if len(s) and pd.notna(s.iloc[-1]):
                    sma_50 = float(s.iloc[-1])
            if "Volume" in df.columns and len(df) >= 51:
                # Prior 50 bars, EXCLUDING today — matches triggers._volume_ratio so the shared
                # 1.5× heavy-volume exit gate reads the same on both surfaces.
                avg_vol = float(df["Volume"].iloc[-51:-1].mean())
                if avg_vol > 0:
                    volume_ratio = float(df["Volume"].iloc[-1]) / avg_vol

        gain_pct = _attr_float(p, "unrealized_plpc")
        if gain_pct is None and avg_entry and price:
            gain_pct = (price - avg_entry) / avg_entry
        below_sma50 = bool(sma_50 is not None and last_close is not None and last_close < sma_50)

        template_criteria = None
        if df is not None and len(df):
            try:
                from .scan import template_chain
                _chain = template_chain(df)
                if _chain is not None:
                    template_criteria = int(_chain[0].get("criteria_passed", 0))
            except Exception:
                template_criteria = None

        next_earnings, earnings_in = earn.get(sym, (None, None))
        pos = {
            "symbol": sym, "qty": qty, "avg_entry": avg_entry, "current_price": price,
            "market_value": _attr_float(p, "market_value"),
            "cost_basis": _attr_float(p, "cost_basis"),
            "unrealized_pl": _attr_float(p, "unrealized_pl"),
            "unrealized_plpc": _attr_float(p, "unrealized_plpc"),
            "lastday_price": _attr_float(p, "lastday_price"),
            "current_stop": current_stop, "has_stop": current_stop is not None,
            "sma_50": sma_50, "last_close": last_close, "volume_ratio": volume_ratio,
            "gain_pct": gain_pct, "below_sma50": below_sma50,
            "next_earnings": next_earnings, "earnings_in": earnings_in,
            "stage": position_stage(gain_pct),
            "template_criteria": template_criteria, "df": df,
        }
        pos["advisories"] = position_advisories(pos)
        positions.append(pos)

    total_pl = sum(p["unrealized_pl"] for p in positions if p["unrealized_pl"] is not None)
    account = {
        "account_number": account_number, "equity": equity, "cash": cash,
        "using_dedicated": using_dedicated, "positions_count": len(positions),
        "total_unrealized_pl": total_pl,
    }
    return {"account": account, "positions": positions}


def rearm_stops(targets: List[dict]) -> dict:
    """Raise/place GTC protective stops for already-held names via the shared one-way ratchet
    (:func:`_rearm_gtc_stop`) — never lowering a stop. Each target: ``{ticker, stop_price, price}``
    (``price`` = the reference the stop must sit below). A ticker not held in the account is
    skipped. Returns ``{equity, cash, account_number, using_dedicated, results}`` — the same shape
    and status vocabulary as :func:`submit_buy_plan`. Raises :class:`TradeUnavailable`."""
    client, using_dedicated = _connect_paper()
    try:
        from alpaca.trading.requests import StopOrderRequest, GetOrdersRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderType
    except ImportError as e:
        raise TradeUnavailable(str(e)) from e

    acct = client.get_account()
    equity, cash = float(acct.equity), float(acct.cash)
    account_number = getattr(acct, "account_number", "?")
    held = {p.symbol: int(float(p.qty)) for p in client.get_all_positions()}
    by_sym = _open_sell_stops_by_symbol(
        client, GetOrdersRequest=GetOrdersRequest, QueryOrderStatus=QueryOrderStatus,
        OrderSide=OrderSide, OrderType=OrderType)

    results: List[dict] = []
    for tgt in targets:
        t = tgt.get("ticker")
        held_shares = held.get(t, 0)
        if held_shares <= 0:
            results.append({**tgt, "status": "skipped", "detail": "not held in this account"})
            continue
        try:
            res = _rearm_gtc_stop(client, t, held_shares, tgt.get("stop_price"),
                                  tgt.get("price"), by_sym.get(t, []),
                                  OrderSide=OrderSide, TimeInForce=TimeInForce,
                                  StopOrderRequest=StopOrderRequest)
            results.append({**tgt, **res})
        except Exception as e:                          # one bad symbol shouldn't abort the rest
            results.append({**tgt, "status": "failed", "detail": str(e)})
    return {"equity": equity, "cash": cash, "account_number": account_number,
            "using_dedicated": using_dedicated, "results": results}


def submit_position_sell(symbol: str, qty: int, *,
                         remainder_stop: Optional[float] = None) -> dict:
    """Manual market SELL of part/all of a held position (paper account), stop-aware.

    Shares covered by an open GTC sell-stop are RESERVED at Alpaca, so the flow is:
    cancel the symbol's open sell-stops → submit the market SELL (DAY, tagged
    ``SEPAsell-``) → re-place a GTC stop for any REMAINING shares at
    ``max(old level, remainder_stop)`` — the ratchet never lowers, and with no prior
    stop a given ``remainder_stop`` places one (the free-roll's stop→breakeven move
    rides this). If the market sell fails AFTER the cancel, the previous stop is
    restored for the full held quantity at the OLD level — the position is never silently left
    unprotected (the cancel-before-place gap done right).

    No $50 floor / 10%-cap / tradability gate: like the stop re-arm path, a sell is
    risk-reducing. ``qty`` above the held count clamps to it (a stale page is the only
    way there — the UI caps the input). A cancel that silently failed can still bounce
    the market sell on reserved shares; that lands in the failure/restore path and the
    duplicate restore is itself reported. Returns a flat dict for the page's icon
    renderer: ``{status: submitted|skipped|failed, detail, symbol, sold_qty, remaining,
    stop_price, account_number, equity}``. Raises :class:`TradeUnavailable`."""
    client, _using_dedicated = _connect_paper()
    try:
        from alpaca.trading.requests import (MarketOrderRequest, StopOrderRequest,
                                             GetOrdersRequest)
        from alpaca.trading.enums import (OrderSide, TimeInForce, QueryOrderStatus,
                                          OrderType)
    except ImportError as e:
        raise TradeUnavailable(str(e)) from e

    acct = client.get_account()
    base = {"symbol": symbol, "sold_qty": 0, "remaining": 0, "stop_price": None,
            "account_number": getattr(acct, "account_number", "?"),
            "equity": float(acct.equity)}

    held = 0
    for p in client.get_all_positions():
        if getattr(p, "symbol", None) == symbol:
            try:
                held = int(float(getattr(p, "qty", 0) or 0))
            except (TypeError, ValueError):
                held = 0
            break
    try:
        qty = int(qty)
    except (TypeError, ValueError):
        qty = 0
    if held < 1:
        return {**base, "status": "skipped", "detail": "not held in this account"}
    if qty < 1:
        return {**base, "status": "skipped", "detail": "sell quantity must be ≥ 1"}
    clamp_note = ""
    if qty > held:
        clamp_note = f" (requested {qty}, clamped to the {held} held)"
        qty = held
    remaining = held - qty
    base.update(sold_qty=qty, remaining=remaining)

    existing = _open_sell_stops(client, symbol, GetOrdersRequest=GetOrdersRequest,
                                QueryOrderStatus=QueryOrderStatus, OrderSide=OrderSide,
                                OrderType=OrderType)
    levels = [q for q in (_stop_price_of(od) for od in existing) if q is not None]
    old_level = max(levels) if levels else None
    base["stop_price"] = old_level
    cancelled = _cancel_orders(client, existing)

    # Ratchet: a requested remainder_stop only ever RAISES the remainder's level.
    if remainder_stop is None:
        eff_level = old_level
    elif old_level is None:
        eff_level = float(remainder_stop)
    else:
        eff_level = max(old_level, float(remainder_stop))

    def _place_stop(stop_qty: int, level: float) -> bool:
        try:
            client.submit_order(order_data=StopOrderRequest(
                symbol=symbol, qty=stop_qty, side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC, stop_price=round(level, 2),
                client_order_id=f"SEPAstop-{symbol}-{int(time.time() * 1000)}"))
            return True
        except Exception:
            return False

    try:
        client.submit_order(order_data=MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
            client_order_id=f"SEPAsell-{symbol}-{int(time.time() * 1000)}"))
    except Exception as e:
        detail = f"market sell failed: {e}"
        if cancelled and old_level is not None:
            # Restore at the OLD level: the sell never happened, so a raised
            # remainder_stop has no business being in force.
            detail += (f"; previous stop restored @ {old_level:.2f}"
                       if _place_stop(held, old_level)
                       else "; stop restore FAILED — arm a stop manually")
        return {**base, "status": "failed", "detail": detail + clamp_note}

    detail = f"market SELL {qty}/{held} sh (DAY)"
    if remaining > 0:
        if eff_level is not None:
            if _place_stop(remaining, eff_level):
                base["stop_price"] = eff_level
                detail += (f"; stop re-placed @ {eff_level:.2f} for the remaining "
                           f"{remaining} sh")
                if old_level is not None and eff_level > old_level:
                    detail += f" (raised from {old_level:.2f})"
            else:
                detail += (f"; stop re-place FAILED — remaining {remaining} sh "
                           "unprotected, re-arm manually")
        else:
            detail += f"; remaining {remaining} sh have no stop — arm one"
    elif cancelled:
        detail += "; protective stop cancelled (no shares remain)"
    return {**base, "status": "submitted", "detail": detail + clamp_note}


# --------------------------------------------------------------------------- #
# Trade journal — "know your numbers" (Think & Trade Like a Champion)
# --------------------------------------------------------------------------- #
SEPA_TAG_PREFIXES = ("SEPAoto-", "SEPAstop-", "SEPAcockpit-", "SEPAsell-")
_ORDERS_PAGE_LIMIT = 500        # Alpaca's max order-history page size
_MAX_ORDER_PAGES = 40           # pagination ceiling (40 × 500 = 20k orders), not a real limit


def _fill_time(fill: dict):
    """A fill's timestamp as a tz-aware UTC ``pd.Timestamp``. Missing/unparseable times map
    to epoch 0 so undated fills sort first deterministically instead of raising."""
    import pandas as pd
    t = pd.to_datetime(fill.get("time"), utc=True, errors="coerce")
    return t if pd.notna(t) else pd.Timestamp(0, tz="UTC")


def build_trade_journal(fills: List[dict]) -> dict:
    """Reconstruct round-trip trades from raw order fills. Pure — no network.

    ``fills``: ``{symbol, side ("buy"/"sell"), qty, price, time, client_order_id}`` dicts
    (:func:`fetch_order_fills` produces them; any time parseable by pandas works). Fills are
    sorted by time and grouped per symbol into POSITION EPISODES — flat → long → flat closes
    one trade — so scale-ins and partial sells aggregate into a single round trip (avg entry
    vs avg exit), which is what the "know your numbers" stats count as ONE trade.

    Returns ``{"closed": [...], "open": [...], "unmatched_sells": [...]}``:

    * ``closed`` — fully-exited episodes: ``{symbol, entry_date, exit_date, hold_days,
      shares, avg_entry, avg_exit, cost, proceeds, pl, pl_pct, n_fills, tagged}``
      (``pl_pct`` is a fraction of cost, like the positions page's ``gain_pct``);
    * ``open`` — episodes still holding shares: ``{symbol, entry_date, shares_open,
      avg_entry, realized_pl, n_fills, tagged}`` — ``realized_pl`` books partial sells at
      the episode's average cost; open episodes are EXCLUDED from the closed-trade stats;
    * ``unmatched_sells`` — a sell fill (or the excess part of one) with no prior buy in
      the supplied history (pre-history / transferred shares): recorded, never guessed at.

    ``tagged`` is True when ANY fill in the episode carries a cockpit client_order_id
    (``SEPA_TAG_PREFIXES``) — the entry tag alone is enough, because a triggered OTO stop
    leg exits under an Alpaca-generated id, not the parent's SEPA one.
    """
    eps = 1e-9
    by_sym: Dict[str, List[dict]] = {}
    for f in sorted(fills, key=_fill_time):
        sym = f.get("symbol")
        if sym:
            by_sym.setdefault(sym, []).append(f)

    closed: List[dict] = []
    open_eps: List[dict] = []
    unmatched: List[dict] = []
    for sym in sorted(by_sym):
        ep = None                                       # the in-flight episode, if any
        for f in by_sym[sym]:
            try:
                qty = float(f.get("qty") or 0.0)
                price = float(f.get("price") or 0.0)
            except (TypeError, ValueError):
                continue
            if qty <= 0 or price <= 0:
                continue
            side = str(f.get("side", "")).lower()
            tagged = str(f.get("client_order_id") or "").startswith(SEPA_TAG_PREFIXES)
            t = _fill_time(f)

            if side == "buy":
                if ep is None:
                    ep = {"entry": t, "buy_qty": 0.0, "buy_cost": 0.0, "sell_qty": 0.0,
                          "proceeds": 0.0, "open": 0.0, "n": 0, "tagged": False}
                ep["buy_qty"] += qty
                ep["buy_cost"] += qty * price
                ep["open"] += qty
                ep["n"] += 1
                ep["tagged"] = ep["tagged"] or tagged
            elif side == "sell":
                if ep is None or ep["open"] <= eps:
                    unmatched.append({"symbol": sym, "qty": qty, "price": price, "time": t,
                                      "client_order_id": f.get("client_order_id", "")})
                    continue
                matched = min(qty, ep["open"])
                if qty - matched > eps:                 # the excess has nothing to close
                    unmatched.append({"symbol": sym, "qty": qty - matched, "price": price,
                                      "time": t,
                                      "client_order_id": f.get("client_order_id", "")})
                ep["sell_qty"] += matched
                ep["proceeds"] += matched * price
                ep["open"] -= matched
                ep["n"] += 1
                ep["tagged"] = ep["tagged"] or tagged
                if ep["open"] <= eps:                   # flat again -> one closed round trip
                    pl = ep["proceeds"] - ep["buy_cost"]
                    closed.append({
                        "symbol": sym, "entry_date": ep["entry"], "exit_date": t,
                        "hold_days": max((t - ep["entry"]).days, 0),
                        "shares": ep["buy_qty"],
                        "avg_entry": ep["buy_cost"] / ep["buy_qty"],
                        "avg_exit": ep["proceeds"] / ep["sell_qty"],
                        "cost": ep["buy_cost"], "proceeds": ep["proceeds"],
                        "pl": pl, "pl_pct": pl / ep["buy_cost"],
                        "n_fills": ep["n"], "tagged": ep["tagged"],
                    })
                    ep = None
        if ep is not None and ep["open"] > eps:
            avg_cost = ep["buy_cost"] / ep["buy_qty"]
            open_eps.append({
                "symbol": sym, "entry_date": ep["entry"], "shares_open": ep["open"],
                "avg_entry": avg_cost,
                "realized_pl": ep["proceeds"] - avg_cost * ep["sell_qty"],
                "n_fills": ep["n"], "tagged": ep["tagged"],
            })
    return {"closed": closed, "open": open_eps, "unmatched_sells": unmatched}


def journal_stats(closed: List[dict]) -> dict:
    """Minervini's "know your numbers" from :func:`build_trade_journal` closed trades. Pure.

    Returns ``{n, wins, losses, scratches, batting_avg, avg_win_pct, avg_loss_pct,
    win_loss_ratio, expectancy_pct, total_pl, avg_hold_days_win, avg_hold_days_loss}``.
    Percent fields are FRACTIONS (0.15 = +15%). ``batting_avg`` = wins / all closed
    (a $0 scratch counts against the average but is neither win nor loss);
    ``expectancy_pct`` = mean ``pl_pct`` across ALL closed trades, i.e. batting × avg win
    + (1 − batting) × avg loss with scratches at 0 — the per-trade edge that gates
    progressive exposure. Every ratio degrades to ``None`` (never raises) when its inputs
    are empty, so a fresh account renders as "—" rather than a crash."""
    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    n = len(closed)
    wins = [t for t in closed if t["pl"] > 0]
    losses = [t for t in closed if t["pl"] < 0]
    avg_win = _mean([t["pl_pct"] for t in wins])
    avg_loss = _mean([t["pl_pct"] for t in losses])
    return {
        "n": n, "wins": len(wins), "losses": len(losses),
        "scratches": n - len(wins) - len(losses),
        "batting_avg": (len(wins) / n) if n else None,
        "avg_win_pct": avg_win, "avg_loss_pct": avg_loss,
        "win_loss_ratio": (avg_win / abs(avg_loss)) if (avg_win is not None and avg_loss)
                          else None,
        "expectancy_pct": _mean([t["pl_pct"] for t in closed]),
        "total_pl": sum(t["pl"] for t in closed),
        "avg_hold_days_win": _mean([t["hold_days"] for t in wins]),
        "avg_hold_days_loss": _mean([t["hold_days"] for t in losses]),
    }


def suggest_risk_pct(closed: List[dict], last_n: int = RISK_GUIDE_LAST_N) -> dict:
    """Progressive-exposure risk-% suggestion from the LAST ``last_n`` closed trades. Pure.

    Trades smaller after losses, earns the right to size up: batting ~.300 at the ~2:1
    payoff the 7-8% stop discipline targets is roughly breakeven, so below that — or with
    outright negative expectancy — the recent read is not working and the unit halves
    (``RISK_PCT_PILOT``). Batting ≥ .500 with positive expectancy steps up ONE notch
    (``RISK_PCT_STRONG`` — progressive exposure is stepwise, and the 10% single-order cap
    still clamps position size). A thin sample (< ``RISK_GUIDE_MIN_TRADES``) stays at
    ``RISK_PCT_BASE`` and never raises. ``closed`` rows come from
    :func:`build_trade_journal`, which groups by SYMBOL — this function re-sorts by
    ``exit_date`` so "last N" means the most recent, not an alphabetical accident.

    Returns ``{risk_pct, reason, n, wins, losses, batting_avg, expectancy_pct}`` with the
    numbers baked into ``reason`` for display at the point of sizing."""
    recent = sorted(closed or [], key=lambda t: t["exit_date"])[-last_n:]
    s = journal_stats(recent)
    n, wins, losses = s["n"], s["wins"], s["losses"]
    batting, expectancy = s["batting_avg"], s["expectancy_pct"]
    base = {"n": n, "wins": wins, "losses": losses,
            "batting_avg": batting, "expectancy_pct": expectancy}
    if n < RISK_GUIDE_MIN_TRADES:
        return {**base, "risk_pct": RISK_PCT_BASE,
                "reason": (f"only {n} closed trade(s) — default sizing until the "
                           "sample grows")}
    exp_s = f"{expectancy * 100:+.1f}%"
    form = f"last {n} closed: {wins}W/{losses}L, expectancy {exp_s}"
    if expectancy <= 0 or batting < 0.3:
        return {**base, "risk_pct": RISK_PCT_PILOT, "reason": f"{form} — pilot size"}
    if batting >= 0.5:
        return {**base, "risk_pct": RISK_PCT_STRONG,
                "reason": f"{form} — press modestly"}
    return {**base, "risk_pct": RISK_PCT_BASE, "reason": f"{form} — normal size"}


def fetch_order_fills() -> dict:
    """Pull the cockpit account's FULL closed-order history from Alpaca and normalize the
    filled ones into the fill dicts :func:`build_trade_journal` consumes.

    Pages backwards through ``GetOrdersRequest(status=CLOSED)`` using ``until`` = the
    oldest ``submitted_at`` seen (Alpaca's ``until`` is exclusive, so pages don't overlap;
    orders are deduped by id anyway as cheap insurance). Orders that never filled
    (cancelled/expired, ``filled_qty`` 0) are dropped; partial fills are kept at their
    ``filled_qty`` × ``filled_avg_price``. Returns ``{"account": {account_number, equity,
    cash, using_dedicated}, "fills": [...]}`` with fills sorted oldest-first. Raises
    :class:`TradeUnavailable` on missing package/credentials."""
    client, using_dedicated = _connect_paper()
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
    except ImportError as e:
        raise TradeUnavailable(str(e)) from e
    try:
        from alpaca.common.enums import Sort
        direction = Sort.DESC                          # newest-first pages; until walks back
    except Exception:
        direction = None                               # older alpaca-py: server default (desc)

    acct = client.get_account()
    account = {
        "account_number": getattr(acct, "account_number", "?"),
        "equity": float(acct.equity), "cash": float(acct.cash),
        "using_dedicated": using_dedicated,
    }

    orders: List = []
    until = None
    for _ in range(_MAX_ORDER_PAGES):
        kw = {"status": QueryOrderStatus.CLOSED, "limit": _ORDERS_PAGE_LIMIT}
        if direction is not None:
            kw["direction"] = direction
        if until is not None:
            kw["until"] = until
        page = list(client.get_orders(filter=GetOrdersRequest(**kw)) or [])
        orders.extend(page)
        if len(page) < _ORDERS_PAGE_LIMIT:
            break
        stamps = [s for s in (getattr(o, "submitted_at", None) for o in page)
                  if s is not None]
        nxt = min(stamps) if stamps else None
        if nxt is None or (until is not None and nxt >= until):
            break                                      # no progress -> stop, don't spin
        until = nxt

    fills: List[dict] = []
    seen = set()
    for o in orders:
        oid = str(getattr(o, "id", "") or "")
        if oid and oid in seen:
            continue
        seen.add(oid)
        sym = getattr(o, "symbol", None)
        qty = _attr_float(o, "filled_qty")
        price = _attr_float(o, "filled_avg_price")
        if not sym or not qty or qty <= 0 or not price or price <= 0:
            continue
        raw_side = getattr(o, "side", "")
        side = str(getattr(raw_side, "value", raw_side)).lower()
        side = "buy" if "buy" in side else ("sell" if "sell" in side else None)
        if side is None:
            continue
        fills.append({
            "symbol": sym, "side": side, "qty": qty, "price": price,
            "time": getattr(o, "filled_at", None) or getattr(o, "submitted_at", None),
            "order_id": oid,
            "client_order_id": str(getattr(o, "client_order_id", "") or ""),
        })
    fills.sort(key=_fill_time)
    return {"account": account, "fills": fills}
