"""Paper-trade the cockpit watchlist through Alpaca.

:func:`build_buy_plan` sizes each name by a chosen mode: a % of equity, a flat $ amount,
an explicit share count, or **risk-to-stop**, Minervini's position sizer:
``shares = floor((equity × risk%) / (price − stop))``. A stop-out then costs ≈ risk% of the
account. Sizing uses the live price (or the limit), not the pivot.

Orders go to the **paper** account only: :func:`_connect_paper` forces ``paper=True``.
Guardrails: a $50 per-order floor in the dollar-denominated modes and a 10%-of-equity
single-order cap. The risk mode clamps to the cap and flags ``capped``; submit skips any
other over-cap buy.

:func:`build_buy_plan` is pure and network-free. The Alpaca functions import alpaca-py
lazily, so the cockpit still loads without it.
"""
from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

# doctrine is constants-only and imports nothing, so this module stays import-light.
from .doctrine import (DEFAULT_STOP_FROM_PIVOT, DERIVED_STOP_FLOOR, DERIVED_STOP_MIN_WINS,
                       DERIVED_STOP_WIN_FRACTION, EARNINGS_SOON_DAYS, MAX_LOSS_FROM_FILL,
                       NO_CHASE_PCT, VOL_AVG_DAYS, VOL_CONFIRM_RATIO)

MIN_TRADE_USD = 50.0        # MUST match alpaca_trader.MIN_TRADE_USD. Copied so the pure
                            # plan builder needn't import alpaca-py.
MAX_ORDER_PCT = 0.10        # single-order cap, a fraction of equity. MUST match
                            # alpaca_trader.MAX_ORDER_PCT, which submit re-checks against.
STALE_PLAN_BARS = 2         # skip a plan name whose freshest bar is more than this many
                            # trading days old. The scan memo never expires. 2 absorbs a
                            # weekend plus a holiday.
ALPACA_TIMEOUT_S = (5.0, 15.0)  # (connect, read) seconds on every Alpaca request. alpaca-py
                                # sets none, so a dropped connection would block the caller
                                # forever. Alpaca answers in < 1 s.

# --- Positions-page stop management (Minervini exit rules) ---------------------------------- #
INITIAL_STOP_PCT = 0.08     # ~8% initial stop below the entry (buy point)
BREAKEVEN_GAIN = 0.16       # gain past which the auto stop is at least breakeven (~2x initial risk)
TRAIL_GAIN = 0.20           # gain past which, "well in profit", trail the 50-day SMA
SELL_STRENGTH_GAIN = 0.20   # gain past which to consider selling part into strength
HEAVY_VOL_RATIO = VOL_CONFIRM_RATIO   # a heavy-volume day IS the breakout-confirmation bar
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

# The cockpit trades its own Alpaca paper account, separate from the All-Weather mirror that
# owns the shared ALPACA_API_KEY/SECRET pair. Each paper account has its own key pair. The
# dedicated "Minervini Trader" keys win; the shared pair is the fallback. These two names MUST
# match .env exactly: a mismatch reads None and silently trades the shared account.
MINERVINI_KEY_ENVS = "ALPACA_API_KEY_MINERVINI"
MINERVINI_SECRET_ENVS = "ALPACA_API_KEY_SECRET_MINERVINI"
# Either spelling works for the shared pair; the first non-empty one wins.
SHARED_KEY_ENVS = ("ALPACA_API_KEY", "ALPACA_API_KEY_PAPER1")
SHARED_SECRET_ENVS = ("ALPACA_API_SECRET", "ALPACA_API_SECRET_PAPER1")


def _first_env(names: "Sequence[str] | str") -> Optional[str]:
    """The value of the first env var in ``names`` that is set and non-empty, else None.

    ``names`` is one name or a sequence of names. A bare string is ONE name: iterating it
    would resolve a stray one-char var such as ``$_`` instead of the key.
    """
    if isinstance(names, str):
        names = (names,)
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


class TradeUnavailable(RuntimeError):
    """Alpaca can't be reached: alpaca-py is missing, .env has no credentials, or a read
    timed out."""


def _with_timeout(client, timeout=ALPACA_TIMEOUT_S):
    """Give every request on ``client``'s HTTP session a default ``timeout``. Returns ``client``.

    alpaca-py builds its own ``requests.Session`` (``client._session``) and passes no
    timeout, so the hook is the session's transport adapter. It fills in the timeout on each
    send unless the caller set one. A client without ``_session`` is returned untouched, not
    refused: ``test_connect_paper_installs_timeout`` then fails the deploy gate instead of the
    app failing closed at runtime."""
    from requests.adapters import HTTPAdapter

    class _DefaultTimeout(HTTPAdapter):
        def send(self, request, **kwargs):
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = timeout
            return super().send(request, **kwargs)

    session = getattr(client, "_session", None)
    if session is not None:
        adapter = _DefaultTimeout()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
    return client


def _alpaca_timeout(err) -> TradeUnavailable:
    """A read timeout as the pages' "can't reach Alpaca" error. The Positions page shows a
    TradeUnavailable as a warning and never memoizes it, so Refresh retries."""
    return TradeUnavailable(f"Alpaca didn't answer within {ALPACA_TIMEOUT_S[1]:.0f}s — "
                            f"press Refresh to retry. ({type(err).__name__})")


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
    return _with_timeout(TradingClient(key, secret, paper=True)), bool(ded_key and ded_secret)


def fetch_account_summary() -> dict:
    """The target account, so the UI can confirm *which* account will trade before any
    order is sent. Returns ``{account_number, equity, cash, using_dedicated}``; raises
    :class:`TradeUnavailable` on missing package/credentials."""
    client, using_dedicated = _connect_paper()
    acct = client.get_account()
    return {
        "account_number": getattr(acct, "account_number", "?"),
        "equity": float(acct.equity),
        "cash": float(acct.cash),
        "using_dedicated": using_dedicated,
    }


def fetch_held_shares() -> Dict[str, int]:
    """``{symbol: whole shares held}`` on the cockpit's paper account, truncated by
    ``int(float(qty))`` as in :func:`submit_buy_plan`. Feeds :func:`build_buy_plan`'s
    ``held`` and the Build preview's "stop re-arm only, no buy" marks. Raises
    :class:`TradeUnavailable` on missing package/credentials; the caller reads that as
    unknown (no held marks)."""
    client, _ = _connect_paper()
    return {p.symbol: int(float(p.qty)) for p in client.get_all_positions()}


def _pos_float(p, name) -> Optional[float]:
    try:
        v = getattr(p, name, None)
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def position_symbols() -> List[str]:
    """The sorted symbols held on the paper account. No price history, no journal.

    The refresh job calls this to decide which prices to download, so it MUST NOT download
    prices itself as :func:`fetch_positions` does. A held name off the watchlist still needs
    fresh bars, or the Positions page and the sell pillars cannot price it. Raises
    :class:`TradeUnavailable` on missing package/credentials; the caller falls back to the
    watchlist alone."""
    client, _ = _connect_paper()
    return sorted({s for s in (getattr(p, "symbol", None)
                               for p in client.get_all_positions()) if s})


def fetch_gate_inputs() -> dict:
    """Inputs for :func:`gate_status`: Alpaca positions plus the journal's open and closed
    episodes. No price history: the unrealized figures come from the broker.

    Returns ``{positions, open_episodes, closed_episodes}``. Raises
    :class:`TradeUnavailable` on missing package/credentials; other broker errors
    propagate. On any failure the manual trade panel reads the gate as UNKNOWN (open; the
    human judges) and the unattended morning executor as CLOSED. The asymmetry is
    deliberate."""
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

    Only cockpit-TAGGED positions count: an Alpaca position needs a tagged OPEN journal
    episode for its symbol. Manual or legacy holdings never affect the gate; a book of
    untagged names reads as flat. Rules:

    * flat (no tagged positions) -> OPEN; the first pilot is always allowed;
    * otherwise OPEN only when every position in the NEWEST set is at breakeven or better
      (an unknown P&L fails), AND the tagged book's net unrealized P&L is >= 0. The
      newest set is the tagged positions sharing the latest entry DATE: same-open fills
      differ only by milliseconds;
    * ``consecutive_losses`` counts losing TAGGED closed trades back from the latest
      ``exit_date``. Any ``pl >= 0``, a $0 scratch included, ends the streak. At
      ``GATE_HALF_SIZE_AFTER`` the ``probe_size_factor`` drops to 0.5. ADVISORY ONLY:
      shown in captions, never applied to quantities.

    Sells and stop re-arms are never gated; they reduce risk. A position whose entry date
    can't be read counts as newest. Returns ``{open, reason, probe_size_factor,
    consecutive_losses}``."""
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
    """True when ``stop_price`` is positive and strictly BELOW the reference ``price``.

    Alpaca rejects a sell stop at or above the market, which would trigger at once, and an
    OTO stop-loss leg that isn't below the entry.
    """
    return bool(stop_price and price and stop_price > 0 and stop_price < price)


def stop_below_fill(fill, pct) -> float:
    """A stop ``pct`` (fraction) below ``fill``, rounded UP to the cent.

    Rounding down could emit a stop that :func:`stop_within_max_loss` refuses. The inner
    ``round`` strips float noise: 42 × 0.9 × 100 is 3780.0000000000005."""
    return math.ceil(round(float(fill) * (1.0 - float(pct)) * 100.0, 6)) / 100.0


def fill_floor(fill) -> float:
    """The lowest stop the max loss allows for a fill at ``fill``."""
    return stop_below_fill(fill, MAX_LOSS_FROM_FILL)


def stop_within_max_loss(stop, fill) -> bool:
    """True when ``stop`` loses at most ``MAX_LOSS_FROM_FILL`` of a buy filled at ``fill``.
    Missing inputs read False."""
    try:
        s, f = float(stop), float(fill)
    except (TypeError, ValueError):
        return False
    return s > 0 and f > 0 and s >= f * (1.0 - MAX_LOSS_FROM_FILL) - 1e-6


def suggest_stop(*, avg_entry: Optional[float], current_price: Optional[float],
                 sma_50: Optional[float], current_stop: Optional[float],
                 gain_pct: Optional[float], basis: str = "auto",
                 initial_pct: Optional[float] = None
                 ) -> Tuple[Optional[float], str]:
    """Minervini stop suggestion for a held position under a chosen ``basis``.

    Basis levels: ``initial`` = ``avg_entry × (1 - INITIAL_STOP_PCT)`` (~8% below entry;
    ``initial_pct`` overrides the 8%, e.g. with :func:`derived_stop_pct`);
    ``breakeven`` = ``avg_entry``; ``sma50`` = ``sma_50 × 0.99``, just under the 50-day.
    ``auto`` picks by gain: ``gain_pct >= TRAIL_GAIN`` with a 50-day trails the SMA;
    ``gain_pct >= BREAKEVEN_GAIN`` is at least breakeven; else ``initial``.

    Returns ``(price_or_None, effective_basis)``. The price is floored at ``current_stop``,
    so it never proposes a lower stop. ``auto`` also floors it at breakeven once
    ``gain_pct >= BREAKEVEN_GAIN``; an explicit basis is honoured as chosen. The price is
    None when there is neither a basis level nor a ``current_stop``, or when the result
    isn't strictly below ``current_price``; that case is left for a manual edit."""
    initial_val = (avg_entry * (1.0 - (INITIAL_STOP_PCT if initial_pct is None else initial_pct))
                   if avg_entry else None)
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
    # The breakeven floor stops an auto 50-day trail giving back a working trade.
    floors = [v for v in (base_val, current_stop) if v is not None]
    if (basis == "auto" and gain_pct is not None and gain_pct >= BREAKEVEN_GAIN
            and breakeven_val is not None):
        floors.append(breakeven_val)
    if not floors:
        return None, eff
    cand = round(max(floors), 2)
    return (cand, eff) if stop_is_valid(cand, current_price) else (None, eff)


def position_stage(gain_pct: Optional[float]) -> Optional[str]:
    """The position's stage on the stop ladder from its gain: ``underwater``, ``fresh``,
    ``working`` or ``well in profit``; None without a gain. Pure. The thresholds MUST match
    :func:`suggest_stop`'s auto basis so the label and the suggested stop agree."""
    if gain_pct is None:
        return None
    if gain_pct < 0:
        return "underwater"
    if gain_pct < BREAKEVEN_GAIN:
        return "fresh"
    if gain_pct < TRAIL_GAIN:
        return "working"
    return "well in profit"


def r_multiple(avg_entry, current_price, pivot=None, current_stop=None,
               stop_pct=None) -> Tuple[Optional[float], bool]:
    """The position's gain as a multiple of its reconstructed initial risk. Pure.

    The entry stop isn't stored, so it is rebuilt from the stops the plan builder could
    have attached for a frozen ``pivot``. Candidates, in order:

    * with ``stop_pct``: ``stop_pct`` below a market fill (≈ the entry) and below a
      zone-top limit, each raised to the pivot default;
    * the pivot default raised to the max-loss floor of those same two fills;
    * the bare pivot default, which older entries carry.

    A candidate within a cent of ``current_stop`` is exact (``approximate=False``).
    Otherwise the first candidate is an estimate. With no usable pivot, risk is
    ``stop_pct`` (else ``INITIAL_STOP_PCT``) off the entry. Returns ``(r, approximate)``;
    ``(None, True)`` on missing or degenerate inputs."""
    try:
        e, c = float(avg_entry), float(current_price)
    except (TypeError, ValueError):
        return None, True
    if e <= 0:
        return None, True
    cands: List[float] = []
    if pivot:
        try:
            ps = float(pivot) * (1.0 - DEFAULT_STOP_FROM_PIVOT)
            zone_top = float(pivot) * (1.0 + NO_CHASE_PCT)
            if stop_pct:
                cands = [max(ps, stop_below_fill(e, stop_pct)),
                         max(ps, stop_below_fill(zone_top, stop_pct))]
            cands += [max(ps, fill_floor(e)), max(ps, fill_floor(zone_top)), ps]
            cands = [s for s in cands if 0.0 < s < e]
        except (TypeError, ValueError):
            cands = []
    risk, approx = None, True
    try:
        cs = float(current_stop) if current_stop is not None else None
    except (TypeError, ValueError):
        cs = None
    if cs is not None:
        for s in cands:
            if abs(s - cs) <= 0.01:
                risk, approx = e - s, False
                break
    if risk is None:
        risk = e - cands[0] if cands else e * (stop_pct or INITIAL_STOP_PCT)
    if risk <= 0:
        return None, True
    return (c - e) / risk, approx


def position_advisories(pos: dict) -> List[str]:
    """Display-only Minervini exit advisories for a :func:`fetch_positions` dict, as a list
    of messages. Pure.

    The "2× initial risk" rule assumes an ``INITIAL_STOP_PCT`` stop because the entry stop
    isn't stored, so it is a nudge, not exact. The earnings rules fire only for a known
    report within ``EARNINGS_SOON_DAYS`` (``earnings_in`` >= 0) and a known gain."""
    out: List[str] = []
    gain = pos.get("gain_pct")
    avg_entry = pos.get("avg_entry")
    cur_stop = pos.get("current_stop")

    if not pos.get("has_stop"):
        out.append("⚠ No protective stop armed — arm one.")
    elif (cur_stop and avg_entry and cur_stop < avg_entry
            and not stop_within_max_loss(cur_stop, avg_entry)):
        # A gap-up market fill lands above the price the plan floored the stop for.
        out.append(f"⚠ Stop {(1 - cur_stop / avg_entry) * 100:.1f}% below your cost "
                   f"(max {MAX_LOSS_FROM_FILL * 100:.0f}%) — raise it to ≥ "
                   f"${fill_floor(avg_entry):,.2f}.")
    ei = pos.get("earnings_in")
    if ei is not None and 0 <= ei <= EARNINGS_SOON_DAYS and gain is not None:
        # A stop can't protect against an earnings gap. Without a cushion, the exit comes
        # before the report.
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
    """Business days from bar date ``last`` to ``today``; 0 when ``last`` is the freshest
    possible bar (today's, or Friday's read on a weekend). Holidays aren't modelled, so one
    reads as an extra day; the caller's tolerance absorbs it. None if either date can't be
    parsed, so the caller skips the check rather than block a trade on a parse error."""
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

    Any failing pillar kills the trade; the stop is only the disaster floor between
    checks. Reads SETTLED closes (``pos["last_close"]`` / ``pos["df"]``), never the live
    print. Missing inputs never raise; a pillar without its inputs reads ``unknown``, and
    a bare position dict with no other arguments yields four unknowns. Returns
    ``{"P1".."P4": {"status": "ok"|"warn"|"fail"|"unknown", "detail": str}}``.

    * P1 breakout holding — needs ``entry_date``, the journal open episode's first buy
      (tz-aware timestamps are read in exchange time). ``pivot`` is the watchlist's frozen
      level; without it the pivot checks are skipped. A Day-0 close back below the pivot,
      a decisive close below it (> ``DECISIVE_BELOW_PIVOT_PCT`` or a 2nd in a row), or a
      close below the breakout bar's low fails outright. A stalled clock warns at day
      ``P1_CUSHION_DAYS`` without a ``P1_CUSHION_PCT`` cushion and fails flat-to-red at
      day ``P1_STALL_DAYS``. Post-breakout violations
      (:func:`advisories.post_breakout_read`) warn. With ``doctrine.VIOLATIONS_CAN_FAIL``
      on, ``VIOLATION_FAIL_COUNT`` of them fail.
    * P2 template — STRICT: anything under 8/8 fails, by the user's choice. Expect an
      occasional one-day red flip when an SMA criterion sits on the edge.
    * P3 tape — the scan ``regime`` dict when available, else the trigger report's
      SPY-only ``spy_note`` (partial: ok/warn), else unknown.
    * P4 earnings — inside the ``EARNINGS_SOON_DAYS`` window a loss or a thin cushion
      (< ``EARNINGS_CUSHION_MIN``) fails; a real cushion still warns (trim to
      hold-through size). ``earnings_in`` None reads unknown: upstream can't tell "no
      report scheduled" from "data missing"."""
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
            if df is not None and len(df):
                from . import advisories, doctrine
                read = advisories.post_breakout_read(
                    df, e, avg_entry=pos.get("avg_entry"),
                    below_sma50=bool(pos.get("below_sma50")),
                    volume_ratio=pos.get("volume_ratio"), today=today)
                viol = (read or {}).get("violations") or []
                if (viol and doctrine.VIOLATIONS_CAN_FAIL
                        and len(viol) >= doctrine.VIOLATION_FAIL_COUNT):
                    fails.append(f"{len(viol)} post-breakout violations: " + "; ".join(viol))
                elif viol:
                    warns.append("violations: " + "; ".join(viol))
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
    """A zero-share ``stop_only`` plan row for a HELD name whose buy failed a sizing gate.
    Submit's held path sends no buy and only re-arms the GTC stop, so the row carries
    ``stop_price`` there and the name keeps its stop maintenance."""
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
                   order_type: str = "market",
                   stop_pct: Optional[float] = None) -> Tuple[List[dict], List[dict]]:
    """Size a BUY for EACH watchlisted name by the chosen ``mode``:

    * ``"pct"``     — ``amount`` % of the account ``equity`` per name (needs ``equity``);
    * ``"dollars"`` — ``amount`` dollars per name;
    * ``"shares"``  — exactly ``amount`` (whole) shares per name;
    * ``"risk"``    — ``amount`` % of ``equity`` risked to the stop (needs ``equity`` + a stop):
      ``shares = (equity × amount%) / (price − stop)``, Minervini's position sizer. A
      stop-out costs ≈ ``amount``% of the account. Sized on the price of the order sent (the
      last close, or the limit), not the pivot, so the risk figure holds for that order.

    Returns ``(plan, skipped)``; raises ValueError for an unknown ``mode`` or
    ``order_type``. Each plan entry has ticker, shares, price, pivot, pivot_frozen,
    est_value, extended, capped, stop_price, stop_floored, stop_derived, limit_price,
    earnings_in and day_range_pct. Each skipped entry is ``{ticker, reason}``. A name is
    skipped when it isn't in the current scan, has no current price, sizes to < 1 share, or
    rounds to a notional under the $50 floor in a dollar-denominated mode (pct, dollars,
    risk). ``"shares"`` is exempt from the floor: the count is explicit. pct and risk skip
    every name without ``equity``. ``"risk"`` also skips a name with no stop, or a stop not
    below the basis (no positive risk per share).

    ``extended`` flags a price above the buy zone's top; the caller warns, never skips.
    ``capped`` is True only in ``"risk"`` mode, when the quantity was clamped to the
    10%-of-equity single-order cap; the realized risk is then below target. ``stop_price``
    is the protective stop (``levels["stop"]``, ~7-8% below the pivot) or None. The caller
    may edit it before submit, but an edit does NOT re-scale a risk-sized quantity.
    ``earnings_in`` (calendar days to the next report; None = unknown) passes through for
    an advisory warning, never a skip. ``day_range_pct`` is the name's typical daily range
    (%), for judging the stop against noise.

    With ``max_bar_age_days`` set, a name whose freshest bar is more than that many trading
    days older than ``asof`` (default today) is skipped as stale. The app pairs this with
    :func:`freshen_prices` at Build. Omit ``max_bar_age_days`` and the builder never reads
    the clock.

    ``pivots`` maps ticker -> a FROZEN judged_pivot, the level the watchlist trigger fired
    on. The scan's detected pivot drifts, so a frozen name's buy zone, ``extended`` flag,
    default ``stop_price`` and risk sizing all key off the frozen level. Its stop is the
    engine's stop when that is below the frozen pivot, else ``DEFAULT_STOP_FROM_PIVOT``
    below it, floored at ``MAX_LOSS_FROM_FILL`` below it, as in ``scan._entry_levels``.
    ``pivot_frozen`` marks rows whose pivot came from ``pivots``. Without ``pivots``, every
    name uses its scan levels.

    **Max loss from the fill.** A non-held name's stop is raised to :func:`fill_floor` of
    its worst-case fill: the limit, or the last price for a market plan. Paying more
    therefore buys a tighter stop. A raised stop at or above the current price skips the
    name. Held names are exempt: their stop is a re-arm target, and raising it would start
    a trailing stop unasked. A name with no stop keeps none; submit refuses it when the
    stop is attached. ``stop_floored`` marks rows where the floor bound.

    ``stop_pct`` (:func:`derived_stop_pct`; None = off) puts a non-held stop that fraction
    below the same worst-case fill. A tighter existing stop is kept. ``stop_derived`` marks
    rows where the derived stop bound.

    ``held`` (``{ticker: shares}``) turns a HELD name whose buy fails a sizing gate (< 1
    share, or under the $50 floor) into a zero-share ``stop_only=True`` row instead of a
    skip, so submit's held path still re-arms its protective stop. Every other skip stands,
    and a name without a stop still skips: there is no level to arm.

    ``order_type="limit"`` plans limit BUYs. Each ``limit_price`` defaults to the buy
    zone's top, the no-chase cap, so a name that gaps past the zone doesn't fill. With no
    zone top it is the last close, a marketable cap. Sizing, the risk per share, the $50
    floor and the 10% cap all use the limit as the basis: a buy limit never fills higher,
    so ``est_value`` is the maximum. A stop at or above the current price skips the name.
    ``"market"`` (the default) sizes on the last close; ``limit_price`` is then None.
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

        # A frozen judged_pivot overrides the drifted scan pivot, so the order matches the
        # level the trigger fired on.
        frozen = None
        if pivots:
            _fp = pivots.get(t)
            frozen = float(_fp) if _fp and _fp > 0 else None
        if frozen:
            pivot, buy_hi = frozen, frozen * (1.0 + NO_CHASE_PCT)
            _eng = lv.get("stop")                        # used if below the pivot
            _raw = (_eng if (_eng and _eng > 0 and _eng < frozen)
                    else frozen * (1.0 - DEFAULT_STOP_FROM_PIVOT))
            stop = max(_raw, frozen * (1.0 - MAX_LOSS_FROM_FILL))
        else:
            bz = lv.get("buy_zone") or (None, None)
            pivot, buy_hi = bz[0], bz[1]
            stop = lv.get("stop")
        capped = False

        # Size on the worst-case fill: ~the current price for a market order, the limit for a
        # buy limit, which never fills higher.
        limit = None
        basis = price
        if order_type == "limit":
            # A stop at or above the price means the base broke below its pivot. The
            # zone-top limit would fill at once and the OTO stop would arm above the market,
            # an instant stop-out. Submit rejects the same numbers for a market plan.
            if stop and stop > 0 and price and stop >= price:
                skipped.append({"ticker": t, "reason":
                                "stop not below the current price — the base has broken "
                                "down below its pivot; re-judge or remove the name"})
                continue
            limit = float(buy_hi) if buy_hi and buy_hi > 0 else price
            basis = limit

        # The OTO stop is fixed at submit, so a limit order MUST carry the stop for its
        # limit, even if it fills lower.
        stop_floored = stop_derived = False
        if not (held and held.get(t, 0) > 0):
            if stop_pct:
                _d = stop_below_fill(basis, stop_pct)
                if not (stop and stop > 0) or _d > stop:
                    stop, stop_derived = _d, True
            if stop and stop > 0:
                _floor = fill_floor(basis)
                if stop < _floor:
                    stop, stop_floored = _floor, True
            if (stop_floored or stop_derived) and stop >= price:
                _rule = (f"{stop_pct * 100:.1f}% (½ your average win)" if stop_derived
                         else f"max {MAX_LOSS_FROM_FILL * 100:.0f}%")
                skipped.append({"ticker": t, "reason":
                                f"a limit at {basis:,.2f} needs a stop ≥ {stop:,.2f} "
                                f"({_rule} below the price paid), above the current price "
                                f"{price:,.2f} — lower the limit or wait for the name to "
                                "reach its zone"})
                continue

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
            # Position% ≈ risk% / stop%, which often exceeds the 10% cap (1% / 8% = 12.5%).
            # Clamp rather than skip; ``capped`` flags the below-target risk.
            cap_shares = int(MAX_ORDER_PCT * equity / basis)
            if shares > cap_shares:
                shares, capped = cap_shares, True
        else:                                                        # "shares"
            shares = int(amount)

        # A held name that fails a sizing gate still needs its stop maintained.
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
            "stop_floored": stop_floored,
            "stop_derived": stop_derived,
            "limit_price": round(limit, 2) if limit else None,
            "earnings_in": payload.get("earnings_in"),
            "day_range_pct": (payload.get("vcp") or {}).get("median_tr_pct"),
        })
    return plan, skipped


def freshen_prices(tickers: Sequence[str], payloads: Dict[str, dict]) -> Dict[str, dict]:
    """A payloads dict for ``tickers`` with each ``df`` replaced by the freshest daily bars.

    The scan memo lives in session_state and never expires, and the trigger fragment keeps
    the tab alive for days, so its closes can be days old at Build. Network call: the
    incremental top-up (``max_age_days=0``, as refresh_job uses) for these names only,
    never the universe. Only ``df`` changes; the other payload keys pass through. A name
    the fetch can't reach keeps its frame, and :func:`build_buy_plan`'s
    ``max_bar_age_days`` then skips it if stale. Tickers absent from ``payloads`` are
    dropped; the builder reports them. A total fetch failure returns the original frames
    and never raises. ``data_feed`` is imported lazily, so the cockpit loads without its
    optional deps."""
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

    A limit sell isn't a stop and is excluded. Alpaca lists a triggered OTO stop leg as its
    own top-level SELL order, so this flat query catches OTO legs as well as standalone
    stops. Returns the order objects, which carry ``stop_price``, for the caller to read
    and cancel; [] on any error.
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
    """An order's ``stop_price`` as a float, or None if absent or unparseable. Callers
    drop None, so a stop with no readable level is replaced rather than ratcheted."""
    v = getattr(order, "stop_price", None)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _order_qty(order) -> int:
    """An order's ``qty`` in whole shares, or 0 if absent or unparseable."""
    v = getattr(order, "qty", None)
    if v is None:
        return 0
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _cancel_orders(client, orders) -> List[str]:
    """Cancel each order by id; one failed cancel never blocks the rest. Returns the
    cancelled ids."""
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

    Same type filter as :func:`_open_sell_stops` without ``symbols=``, so every stop on the
    account comes back in one round trip. Returns ``{symbol: [order, ...]}``; an empty dict
    on any error except a timeout, which raises :class:`TradeUnavailable`."""
    from requests.exceptions import Timeout
    stop_types = {OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP}
    try:
        opens = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN, side=OrderSide.SELL))
    except Timeout as e:
        # "No answer" is not "no stops": an empty dict here renders a protected position as
        # unprotected, and rearm_stops would then place a second stop on the same shares.
        raise _alpaca_timeout(e) from e
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
    """OPEN cockpit BUY orders (``client_order_id`` starts ``SEPA``), in one query. Returns
    the order objects: callers need ids to cancel and symbols to skip. Side=BUY excludes
    ``SEPAstop-`` sells. [] on any error (fail-open)."""
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

    A BUY queued after the close, or a resting GTC limit, has no position yet, so
    ``get_all_positions()`` misses it. :func:`submit_buy_plan` skips these symbols so a
    re-submit can't send a second BUY and double the risk."""
    return {od.symbol for od in _open_cockpit_buy_orders(
        client, GetOrdersRequest=GetOrdersRequest, QueryOrderStatus=QueryOrderStatus,
        OrderSide=OrderSide)}


def cancel_pending_buys() -> dict:
    """Cancel every OPEN cockpit BUY order (``SEPA…`` tags only; never sells, stops or
    other tools' orders). This is the cockpit's control for a resting GTC limit whose setup
    broke: the pending-buy guard blocks re-submits but cannot cancel. Cancelling an
    unfilled OTO parent cancels its held stop leg too; nothing was bought, so nothing
    needs protecting.

    Returns ``{"cancelled": [{ticker, id}], "errors": [{ticker, id, error}]}``; one
    failed cancel never aborts the rest. A failed order query reads as no orders. Raises
    :class:`TradeUnavailable` only for missing package/credentials."""
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
    """Minervini's one-way GTC stop ratchet for a held position. The one implementation,
    shared by :func:`submit_buy_plan`'s held branch and :func:`rearm_stops`.

    ``existing`` is the symbol's open sell-stop orders, fetched by the caller. The in-force
    stop is the highest of their stop prices. It is NEVER lowered: old stops are cancelled
    and replaced only to raise the level, or to re-place it at the same level for the full
    held quantity when they cover fewer shares. GTC keeps the stop across sessions.

    Returns a PARTIAL result dict: ``{status, detail}``, plus ``stop_price`` when a level is
    in force. ``status`` is ``"stop_only"`` (placed, raised or re-placed), ``"stop_kept"``
    (a stop is in force and the new one is lower, equal or invalid), ``"skipped"`` (no
    valid stop and none in force) or ``"failed"`` (placement rejected).

    If placement fails after the old stops were cancelled, the old level is restored once
    for the full held quantity, as in :func:`submit_position_sell`. The position is never
    silently left unprotected: a failed restore is reported in ``detail``."""
    prices = [p for p in (_stop_price_of(od) for od in existing) if p is not None]
    cur = max(prices) if prices else None
    new_stop = round(float(desired_stop), 2) if desired_stop else None

    if not stop_is_valid(new_stop, price):
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

    # A lower-or-equal stop is kept unless the stops under-cover the position, e.g. after a
    # manual pyramid buy. Then re-place at the same level for the full quantity, or the added
    # shares sit unprotected while the UI shows a stop. Only 0 < covered < held counts, so an
    # unreadable qty never causes churn.
    if cur is not None and new_stop <= cur:
        covered = sum(_order_qty(od) for od in existing)
        if not (0 < covered < held_shares):
            return {"status": "stop_kept", "stop_price": cur,
                    "detail": f"kept existing stop @ {cur:.2f} — not lowering to {new_stop:.2f}"}
        cancelled = _cancel_orders(client, existing)
        resp = _try_place(cur)
        if resp is None:
            return _failed(cur, cancelled)
        return {"status": "stop_only", "stop_price": cur,
                "detail": f"GTC stop re-placed for full {held_shares} sh @ {cur:.2f} "
                          f"(was {covered} sh; id {getattr(resp, 'id', '?')})"}

    cancelled = _cancel_orders(client, existing)
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
    """Submit each plan row on the cockpit's Alpaca **paper** account (see
    :func:`_connect_paper`), with a protective stop when ``attach_stop`` is set.

    Per name:

    * **already held** — no buy. With ``attach_stop``, :func:`_rearm_gtc_stop` places a
      **GTC** sell-stop at ``stop_price`` for the WHOLE position, or raises the one in
      force; a lower-or-equal stop leaves it untouched (``"stop_kept"``). Without
      ``attach_stop`` the row is skipped. The 10% cap doesn't apply: a protective stop
      reduces risk.
    * **not held** — a BUY. With ``attach_stop`` it is a **GTC** OTO order with a stop-loss
      leg. The leg activates only on the fill, so it works for a buy queued to the next
      open, and then rests as a GTC stop past the close. A DAY leg would expire at the
      close and leave an intraday fill unprotected overnight. The held-name ratchet
      manages that stop from the next re-arm on. Without ``attach_stop``, a plain DAY
      market BUY.

      A positive ``limit_price`` sends a **limit** BUY instead. With ``attach_stop`` it is a
      GTC OTO limit with a stop leg, so a fill on any later day is protected at once; an
      unfilled limit rests until filled or cancelled, and the pending-buy guard blocks
      re-submits meanwhile. Without ``attach_stop`` it is a DAY limit. A present but
      non-positive ``limit_price`` is skipped. The 10% cap is recomputed from the limit.
      A row whose stop isn't below both the limit and the last price is skipped: a
      marketable limit fills near the price.

      With ``attach_stop``, a buy whose stop is more than ``MAX_LOSS_FROM_FILL`` below its
      highest possible fill (the limit, else the last price) is skipped. The stop and the
      limit are editable after Build, so this MUST re-check what the builder applied.

      **Build-time intent is binding.** A ``rearm_only`` row (held at Build; the preview
      offered no buy) or a ``stop_only`` row whose position has since closed is SKIPPED,
      never turned into a buy the user did not consent to. Rebuild the plan to buy it. A
      row under 1 share, a ``gate_blocked`` row and a symbol with a cockpit BUY already
      queued are skipped too.

    Every row goes through ``alpaca_trader``'s tradability check, and buys through its
    10%-of-equity order cap too. One failed symbol never aborts the rest. Returns ``{equity,
    cash, account_number, using_dedicated, results}``; each result is the plan entry plus
    a ``status`` ("submitted" / "stop_only" / "stop_kept" / "skipped" / "failed") and a
    ``detail`` string. Raises :class:`TradeUnavailable` if alpaca-py or credentials are
    missing.
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
    # Truncated to whole shares as alpaca_trader.get_account_state does.
    held = {p.symbol: int(float(p.qty)) for p in client.get_all_positions()}
    # A queued cockpit BUY isn't a position yet; skipping it stops a re-submit doubling one.
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

            # A rearm_only or stop_only row MUST NOT become a buy because the position
            # closed after Build; its stop firing is enough. The user consented to no buy.
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
            # None is a market plan. A present, non-positive limit is an edit error: skip it
            # rather than fall back to an uncapped market buy.
            if _lim is not None and not (float(_lim) > 0):
                results.append({**o, "status": "skipped",
                                "detail": "invalid limit price — set a limit > 0"})
                continue
            limit = float(_lim) if _lim else None
            # The cap binds on the worst-case fill. A limit row MUST recompute it from the
            # limit, which is editable after Build; est_value is not. A market row's price
            # isn't editable.
            _est = o["shares"] * limit if limit else o["est_value"]
            if _est > max_allowed:
                results.append({**o, "status": "skipped",
                                "detail": f"exceeds 10% of equity (${max_allowed:,.0f} cap)"})
                continue
            if attach_stop:
                # A limit BUY fills at or below the limit, near the price when marketable,
                # so the stop MUST be below both.
                _worst = min(limit, o["price"]) if limit else o["price"]
                if not stop_is_valid(stop, _worst):
                    results.append({**o, "status": "skipped",
                                    "detail": "stop not below entry — fix stop or turn off "
                                              "Attach stop"})
                    continue
                # ...and the max loss binds on the HIGHEST fill. The unattended entry
                # executor submits through here too.
                _paid = limit if limit else o["price"]
                if not stop_within_max_loss(stop, _paid):
                    results.append({**o, "status": "skipped",
                                    "detail": f"stop {float(stop):,.2f} is more than "
                                              f"{MAX_LOSS_FROM_FILL * 100:.0f}% below the "
                                              f"{'limit' if limit else 'price'} "
                                              f"{float(_paid):,.2f} — raise it to ≥ "
                                              f"{fill_floor(_paid):,.2f}"})
                    continue
                # The OTO MUST be GTC: the stop leg inherits it and outlives the close. A
                # DAY leg expires at the close and can leave an intraday fill unprotected.
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
    """``obj.attr`` as a float, or None when absent or not numeric, so the view renders
    without it."""
    v = getattr(obj, attr, None)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch_positions() -> dict:
    """The cockpit's **paper** holdings, each with P&L, in-force stop, 20- and 50-day SMAs,
    volume ratio, template read, earnings date, stop-ladder stage and exit advisories.
    Network: Alpaca, and price and fundamentals fetches when their caches are stale.

    Returns ``{"account": {account_number, equity, cash, using_dedicated, positions_count,
    total_unrealized_pl}, "positions": [per-position dict]}``. Raises
    :class:`TradeUnavailable` on missing package/credentials or an Alpaca read timeout.
    A per-position field reads None, never raises, when its Position attribute is absent
    or the price history is missing or too short."""
    client, using_dedicated = _connect_paper()
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import OrderSide, QueryOrderStatus, OrderType
    except ImportError as e:
        raise TradeUnavailable(str(e)) from e

    from requests.exceptions import Timeout
    try:
        acct = client.get_account()
        raw = list(client.get_all_positions())
    except Timeout as e:
        raise _alpaca_timeout(e) from e
    equity, cash = float(acct.equity), float(acct.cash)
    account_number = getattr(acct, "account_number", "?")
    stops_by_sym = _open_sell_stops_by_symbol(
        client, GetOrdersRequest=GetOrdersRequest, QueryOrderStatus=QueryOrderStatus,
        OrderSide=OrderSide, OrderType=OrderType)

    symbols = [s for s in (getattr(p, "symbol", None) for p in raw) if s]
    # One batched price pull. Lazy imports keep trade.py loadable without yfinance or the
    # vendored screening package.
    frames: dict = {}
    data_feed = None
    if symbols:
        try:
            from . import data_feed as _df_mod
            data_feed = _df_mod
            # The Positions page MUST NOT queue behind the bulk pipeline. _YF_LOCK
            # serializes every in-process download, so a mid-sweep scan or refresh would
            # stall this fetch and the page's SELL controls with it. Selling needs only
            # Alpaca data, and the SMA/volume advisories tolerate a cached bar.
            frames = data_feed.get_many_prices(
                symbols, allow_network=not data_feed.network_busy())
        except Exception:
            frames = {}
    try:
        from src.stock_screener.minervini_screener.screening import calculate_sma
    except Exception:
        calculate_sma = None
    import pandas as pd

    # Earnings dates, best-effort, from a weekly JSON cache per ticker. Serial: positions
    # are few.
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

        sma_50 = sma_20 = last_close = volume_ratio = None
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
            if calculate_sma is not None and len(df) >= 20:
                s = calculate_sma(df["Close"], 20)
                if len(s) and pd.notna(s.iloc[-1]):
                    sma_20 = float(s.iloc[-1])
            # The shared doctrine read, so the heavy-volume flag here and the trigger
            # job's confirmation can never diverge.
            from .indicators import volume_ratio as _vr
            volume_ratio = _vr(df, VOL_AVG_DAYS)

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
            "sma_50": sma_50, "sma_20": sma_20, "last_close": last_close,
            "volume_ratio": volume_ratio,
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
    """Place or raise GTC protective stops on held names through :func:`_rearm_gtc_stop`;
    never lowers one. Each target is ``{ticker, stop_price, price}``, where ``price`` is the
    reference a valid stop sits below. A ticker not held is skipped. Returns ``{equity,
    cash, account_number, using_dedicated, results}``, with the shape and statuses of
    :func:`submit_buy_plan`. Raises :class:`TradeUnavailable` on missing package/credentials
    or a timed-out stops read."""
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
    """Market SELL of part or all of a held position on the paper account, stop-aware.

    Alpaca RESERVES the shares an open sell-stop covers, so the order is: cancel the
    symbol's open sell-stops, submit the DAY market SELL (tagged ``SEPAsell-``), then
    re-place a GTC stop for the REMAINING shares at ``max(old level, remainder_stop)``.
    The ratchet never lowers. With no prior stop, a given ``remainder_stop`` places one;
    the free-roll's move to breakeven relies on this. If the sell fails after the cancel,
    the stop is restored at the OLD level for the full held quantity, so the position is
    never silently left unprotected.

    No $50 floor, 10% cap or tradability gate: a sell reduces risk. ``qty`` above the held
    count clamps to it; only a stale page gets there. A cancel that silently failed can
    still bounce the sell on reserved shares. That takes the failure path, and the
    duplicate restore's outcome is reported in ``detail``. Returns ``{status:
    submitted|skipped|failed, detail, symbol, sold_qty, remaining, stop_price,
    account_number, equity}``. Raises :class:`TradeUnavailable` if alpaca-py or
    credentials are missing."""
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
            # The OLD level: the sell never happened, so a raised remainder_stop MUST NOT
            # take effect.
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

    ``fills``: ``{symbol, side ("buy"/"sell"), qty, price, time, client_order_id}`` dicts,
    as :func:`fetch_order_fills` produces; any time pandas can parse works. A fill with no
    symbol, or a non-positive or unreadable qty or price, is ignored. Fills are sorted by
    time and grouped per symbol into POSITION EPISODES: flat → long → flat is one trade.
    Scale-ins and partial sells aggregate into one round trip (avg entry vs avg exit),
    which the "know your numbers" stats count as ONE trade.

    Returns ``{"closed": [...], "open": [...], "unmatched_sells": [...]}``:

    * ``closed`` — fully exited episodes: ``{symbol, entry_date, exit_date, hold_days,
      shares, avg_entry, avg_exit, cost, proceeds, pl, pl_pct, n_fills, tagged}``.
      ``pl_pct`` is a fraction of cost, like the positions page's ``gain_pct``;
    * ``open`` — episodes still holding shares: ``{symbol, entry_date, shares_open,
      avg_entry, realized_pl, n_fills, tagged}``. ``realized_pl`` books partial sells at
      the episode's average cost. Open episodes are EXCLUDED from the closed-trade stats;
    * ``unmatched_sells`` — a sell fill, or its excess, with no prior buy in the supplied
      history (pre-history or transferred shares). Recorded, never guessed at.

    ``tagged`` is True when ANY fill in the episode carries a cockpit client_order_id
    (``SEPA_TAG_PREFIXES``). The entry tag alone is enough: a triggered OTO stop leg exits
    under an Alpaca-generated id, not the parent's SEPA one.
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
    Percent fields are FRACTIONS (0.15 = +15%). ``batting_avg`` = wins / all closed; a $0
    scratch counts against the average but is neither win nor loss. ``expectancy_pct`` =
    mean ``pl_pct`` over ALL closed trades, i.e. batting × avg win + (1 − batting) × avg
    loss with scratches at 0. It is the per-trade edge that gates progressive exposure.
    A ratio with empty inputs is None, never raises, so a fresh account renders "—"."""
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

    Size down after losses; earn the right to size up. Batting ~.300 at the ~2:1 payoff a
    7-8% stop targets is about breakeven. So batting under .300, or expectancy <= 0, halves
    the unit to ``RISK_PCT_PILOT``. Batting >= .500 with positive expectancy steps up ONE
    notch to ``RISK_PCT_STRONG``; exposure grows stepwise, and the 10% single-order cap
    still clamps size. Under ``RISK_GUIDE_MIN_TRADES`` trades it stays at
    ``RISK_PCT_BASE``. Rows are re-sorted by ``exit_date``, because
    :func:`build_trade_journal` groups them by symbol.

    Returns ``{risk_pct, reason, n, wins, losses, batting_avg, expectancy_pct}``;
    ``reason`` carries the numbers, for display at the point of sizing."""
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


def derived_stop_pct(closed: List[dict]) -> dict:
    """The book's stop from your own record: half the average win, from the fill. Pure.

    Counts cockpit-tagged closed trades only; manual history is a different trader.
    Under ``DERIVED_STOP_MIN_WINS`` wins, ``stop_pct`` is None and the default stop
    applies. Otherwise it is ``avg_win × DERIVED_STOP_WIN_FRACTION``, clamped to
    ``[DERIVED_STOP_FLOOR, MAX_LOSS_FROM_FILL]``. Returns ``{stop_pct, reason, wins,
    avg_win_pct}``; ``reason`` is a caption."""
    tagged = [t for t in (closed or []) if t.get("tagged")]
    s = journal_stats(tagged)
    wins, avg_win = s["wins"], s["avg_win_pct"]
    base = {"wins": wins, "avg_win_pct": avg_win}
    if wins < DERIVED_STOP_MIN_WINS or avg_win is None:
        return {**base, "stop_pct": None,
                "reason": (f"default stop ({DEFAULT_STOP_FROM_PIVOT * 100:.1f}% below the "
                           f"pivot) — the derived stop (½ your average win) starts at "
                           f"{DERIVED_STOP_MIN_WINS} wins; you have {wins}")}
    raw = avg_win * DERIVED_STOP_WIN_FRACTION
    pct = min(max(raw, DERIVED_STOP_FLOOR), MAX_LOSS_FROM_FILL)
    bound = (f", floored at {DERIVED_STOP_FLOOR * 100:.0f}%" if raw < DERIVED_STOP_FLOOR
             else f", capped at {MAX_LOSS_FROM_FILL * 100:.0f}%" if raw > MAX_LOSS_FROM_FILL
             else "")
    return {**base, "stop_pct": pct,
            "reason": (f"derived stop {pct * 100:.1f}% below the fill — ½ × your "
                       f"{avg_win * 100:.1f}% average win over {wins} wins{bound}")}


# --------------------------------------------------------------------------- #
# Loss Adjustment Exercise (Think & Trade Like a Champion): tighter stops, replayed on
# closed trades.
# --------------------------------------------------------------------------- #
LOSS_ADJUST_GRID = tuple(2.0 + 0.5 * i for i in range(17))    # stop X% below cost, 2..10
LOSS_ADJUST_MIN_FLOOR = 3.0     # the engine's "too tight" minimum stop; the floor rule
                                # MUST NOT go below it


def _et_day(ts):
    """A fill timestamp's exchange-time date, as tz-naive midnight. Alpaca fills are UTC,
    and an after-hours fill is already the next UTC day."""
    import pandas as pd
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("America/New_York").tz_localize(None)
    return t.normalize()


def trade_path(trade: dict, df):
    """The daily bars a closed trade was exposed to: ``(entry day, exit day]``, exchange
    time. The entry day is excluded because its low may predate the fill.

    None when ``df`` is missing, lacks Open/Low, or ends before the exit day; a truncated
    path would under-count stop-outs. A same-day round trip returns an empty frame."""
    import pandas as pd
    if df is None or not len(df) or not {"Open", "Low"}.issubset(df.columns):
        return None
    try:
        e, x = _et_day(trade["entry_date"]), _et_day(trade["exit_date"])
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
        idx = idx.normalize()
    except Exception:
        return None
    if idx.max() < x:
        return None
    return df.loc[(idx > e) & (idx <= x)]


def _stopped_return(r: float, path, x: float, avg_entry: float) -> float:
    """A closed trade's return with a stop ``x`` (fraction) below its average cost.

    The first bar whose low reaches the stop exits at ``min(open, stop)``: a gap fills at
    the open, as in the backtest. The exit day counts only when the real exit went through
    the stop (``r < -x``). With no hit, the actual return, floored at ``-x``."""
    stop_r = -x
    last = len(path) - 1
    for i, (o, lo) in enumerate(zip(path["Open"], path["Low"])):
        if i == last and not r < stop_r:
            break
        if float(lo) / avg_entry - 1.0 <= stop_r:
            return min(float(o) / avg_entry - 1.0, stop_r)
    return max(r, stop_r)


def loss_adjustment_sweep(closed: List[dict], xs: Sequence[float] = LOSS_ADJUST_GRID,
                          frames: Optional[Dict[str, object]] = None) -> dict:
    """Minervini's Loss Adjustment Exercise over :func:`build_trade_journal` closed trades.
    Pure; ``frames`` is ``{symbol: daily OHLC}``.

    For each stop ``x`` (percent below the average cost):

    * **book**: every loss bigger than ``x`` is cut to ``-x``; wins are unchanged. It can
      only help, so it is an upper bound.
    * **price-aware**: the stop replayed on the trade's bars (:func:`trade_path`). It also
      stops out a winner that dipped ``x`` first. A trade without usable bars keeps its
      book value; ``n_price_aware`` counts the replayed ones.

    Totals compound returns in exit order, as if each trade used the whole account. They
    compare stops; they are not an account curve. ``winners_stopped`` counts winners the
    price-aware stop turned into losses. Returns ``{n, wins, wins_price_aware,
    n_price_aware, actual_total, actual_expectancy, rows: [{x, book_total, aware_total,
    book_expectancy, aware_expectancy, winners_stopped}]}``."""
    trades = sorted(closed or [], key=lambda t: t["exit_date"])
    frames = frames or {}
    paths = [trade_path(t, frames.get(t["symbol"])) for t in trades]
    actual = [float(t["pl_pct"]) for t in trades]

    def _compound(rs):
        tot = 1.0
        for r in rs:
            tot *= 1.0 + r
        return tot - 1.0

    def _mean(rs):
        return sum(rs) / len(rs) if rs else None

    rows = []
    for x in xs:
        xf = float(x) / 100.0
        book = [max(r, -xf) for r in actual]
        aware, stopped = [], 0
        for t, r, p in zip(trades, actual, paths):
            a = max(r, -xf) if p is None else _stopped_return(r, p, xf, float(t["avg_entry"]))
            if r > 0 and a < 0:
                stopped += 1
            aware.append(a)
        rows.append({"x": float(x), "book_total": _compound(book),
                     "aware_total": _compound(aware), "book_expectancy": _mean(book),
                     "aware_expectancy": _mean(aware), "winners_stopped": stopped})
    return {"n": len(trades), "wins": sum(1 for r in actual if r > 0),
            "wins_price_aware": sum(1 for r, p in zip(actual, paths) if r > 0 and p is not None),
            "n_price_aware": sum(1 for p in paths if p is not None),
            "actual_total": _compound(actual), "actual_expectancy": _mean(actual),
            "rows": rows}


def stop_floor_from_sweep(sweep: dict,
                          min_x: float = LOSS_ADJUST_MIN_FLOOR) -> Optional[float]:
    """The derived stop's floor, as a fraction, by the rule pre-registered in HANDOFF
    §6.73: the smallest grid ``x >= min_x`` at which no closed winner becomes a loss.

    None when no winner has price history. The floor MUST NOT be the return-maximising
    ``x``: on a handful of trades that fits noise (HANDOFF §2)."""
    if not sweep or not sweep.get("wins_price_aware"):
        return None
    for row in sorted(sweep.get("rows") or [], key=lambda r: r["x"]):
        if row["x"] >= min_x - 1e-9 and row["winners_stopped"] == 0:
            return row["x"] / 100.0
    return None


def fetch_order_fills() -> dict:
    """The cockpit account's closed-order history as the fill dicts
    :func:`build_trade_journal` consumes: ``{symbol, side, qty, price, time, order_id,
    client_order_id}``.

    Pages back through ``GetOrdersRequest(status=CLOSED)`` with ``until`` = the oldest
    ``submitted_at`` seen, for at most ``_MAX_ORDER_PAGES`` pages. Alpaca's ``until`` is
    exclusive, so pages don't overlap; orders are deduped by id anyway. Orders that never
    filled are dropped; a partial fill counts at ``filled_qty`` × ``filled_avg_price``.
    Returns ``{"account": {account_number, equity, cash, using_dedicated}, "fills":
    [...]}``, fills oldest first. Raises :class:`TradeUnavailable` on missing
    package/credentials."""
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
