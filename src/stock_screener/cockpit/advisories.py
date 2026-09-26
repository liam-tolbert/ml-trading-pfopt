"""Display-only SEPA reads, shown beside the numbers the cockpit acts on.

Nothing here places, sizes or blocks an order. Each function returns a small verdict
for a page caption or an evening-plan note. A read promoted to act MUST be gated by a
doctrine switch read at call time, not changed here.

pandas and the indicator helpers are imported inside functions: ``trade.py`` imports
this module and must not pull in the scan stack.
"""
from __future__ import annotations

from typing import Optional

from . import doctrine

TYPICAL_DAY_BARS = 42       # MUST equal vcp.DEAD_TAPE_BARS, so the Scan and Positions
                            # pages quote the same typical day


def typical_day_range(df, bars: int = TYPICAL_DAY_BARS) -> Optional[float]:
    """Median daily true range over the last ``bars`` sessions, as a fraction of price.
    The median, so one gap day can't make a quiet stock look wild. None without enough
    bars."""
    if df is None or len(df) < bars + 1:
        return None
    import numpy as np
    from .indicators import true_range_pct
    tr = true_range_pct(df.tail(bars + 1)).tail(bars).to_numpy(dtype=float)
    med = float(np.nanmedian(tr)) if np.isfinite(tr).any() else float("nan")
    return med if np.isfinite(med) and med > 0 else None


def stop_room(day_range, stop, fill) -> Optional[dict]:
    """Typical days of movement between ``fill`` and ``stop``.

    ``day_range`` is a fraction, as from :func:`typical_day_range`. Returns ``{loss_pct,
    room_days, warn}``; ``warn`` is set under ``STOP_ROOM_MIN_DAYS``. None when an input is
    missing or the stop isn't below the fill."""
    try:
        d, s, f = float(day_range), float(stop), float(fill)
    except (TypeError, ValueError):
        return None
    if not (d > 0 and 0 < s < f):
        return None
    loss = (f - s) / f
    room = loss / d
    return {"loss_pct": loss, "room_days": room,
            "warn": room < doctrine.STOP_ROOM_MIN_DAYS}


POST_BREAKOUT_DAYS = 20     # a close under the 20-day line counts only in this window
SMA_SHORT = 20
GIVEBACK_GAIN = 0.05        # a post-entry gain this big, fully given back, is a violation
LOWER_LOWS_RUN = 3          # consecutive lower lows with no buying support


def post_breakout_read(df, entry_date, *, avg_entry=None, below_sma50=False,
                       volume_ratio=None, today=None, now=None) -> Optional[dict]:
    """Post-breakout violations and follow-through, over settled bars from the entry day.

    Violations are reasons to doubt the breakout:

    * a close under the 20-day line within ``POST_BREAKOUT_DAYS``;
    * a light-volume breakout day, then a heavy down day;
    * ``LOWER_LOWS_RUN`` lower lows in a row with no above-average-volume up close;
    * more down closes than up (3+ sessions);
    * more lower-half closes than upper-half;
    * a close under the 50-day on heavy volume;
    * a ``GIVEBACK_GAIN`` gain fully given back.

    Follow-through is evidence it works: up closes on rising volume, 3 of the first 4 or 6
    of the first 8 sessions up, more upper-half closes, every dip recovered within 2
    sessions.

    A live read (``today`` None) drops today's bar while the session is open: its close is
    only the latest print. Returns ``{day_n, violations, follow_through, sma20,
    provisional_dropped}``, or None without a frame or an entry date."""
    if df is None or not len(df) or entry_date is None:
        return None
    import numpy as np
    import pandas as pd

    from .indicators import prior_volume_average

    try:
        e = pd.Timestamp(entry_date)
        if e.tzinfo is not None:
            e = e.tz_convert("America/New_York").tz_localize(None)
        e = e.normalize()
    except Exception:
        return None
    dropped = False
    if today is None:
        from .triggers import bar_is_provisional
        if bar_is_provisional(df.index[-1], now):
            df, dropped = df.iloc[:-1], True
    if not len(df):
        return None

    close = df["Close"].astype(float)
    sma20 = close.rolling(SMA_SHORT, min_periods=SMA_SHORT).mean()
    vol = df["Volume"].astype(float) if "Volume" in df.columns else None
    vavg = (prior_volume_average(vol, doctrine.VOL_AVG_DAYS)
            if vol is not None else None)
    idx = pd.DatetimeIndex(df.index).normalize()
    post_pos = np.flatnonzero(idx >= e)
    if not len(post_pos):
        return None
    first = int(post_pos[0])
    post = df.iloc[first:]
    n = len(post) - 1                                   # day 0 = the entry day
    c = post["Close"].to_numpy(dtype=float)
    hi = post["High"].to_numpy(dtype=float)
    lo = post["Low"].to_numpy(dtype=float)
    vr = ((vol / vavg).iloc[first:].to_numpy(dtype=float)
          if vol is not None else np.full(len(post), np.nan))
    s20 = sma20.iloc[first:].to_numpy(dtype=float)

    violations, follow = [], []

    # Strict "<", so a flat tape never trips it.
    under = [d for d in range(1, min(n, POST_BREAKOUT_DAYS) + 1)
             if np.isfinite(s20[d]) and c[d] < s20[d]]
    sma20_note = None
    if under:
        sma20_note = (f"closed below the 20-day line on day {under[-1]}"
                      + (f" ({len(under)}×)" if len(under) > 1 else ""))
        violations.append(sma20_note)

    ups = [d for d in range(1, n + 1) if c[d] > c[d - 1]]
    downs = [d for d in range(1, n + 1) if c[d] < c[d - 1]]

    # Light-volume breakout, then heavy selling into it.
    heavy = doctrine.VOL_CONFIRM_RATIO
    if np.isfinite(vr[0]) and vr[0] < heavy:
        hd = [d for d in downs if np.isfinite(vr[d]) and vr[d] >= heavy]
        if hd:
            violations.append(f"light-volume breakout ({vr[0]:.1f}×) then a heavy down "
                              f"day (day {hd[0]}, {vr[hd[0]]:.1f}×)")

    # An up close on above-average volume is support: it resets the run.
    run = 0
    for d in range(1, n + 1):
        support = d in ups and np.isfinite(vr[d]) and vr[d] >= 1.0
        run = run + 1 if (lo[d] < lo[d - 1] and not support) else 0
        if run >= LOWER_LOWS_RUN:
            violations.append(f"{run} lower lows in a row (to day {d})")
            break

    if n >= 3 and len(downs) > len(ups):
        violations.append(f"more down closes than up ({len(downs)} vs {len(ups)})")

    rng = hi[1:] - lo[1:]
    pos_in_range = np.where(rng > 0, (c[1:] - lo[1:]) / np.where(rng > 0, rng, 1.0), np.nan)
    lower_half = int(np.sum(pos_in_range < 0.5))
    upper_half = int(np.sum(pos_in_range > 0.5))
    if n >= 3 and lower_half > upper_half:
        violations.append(f"more closes in the lower half of the day's range "
                          f"({lower_half} vs {upper_half})")

    if below_sma50 and volume_ratio is not None and volume_ratio >= heavy:
        violations.append("closed below the 50-day on heavy volume")

    if avg_entry and n >= 1:
        best = float(np.max(hi[1:])) / float(avg_entry) - 1.0
        if best >= GIVEBACK_GAIN and c[-1] <= float(avg_entry):
            violations.append(f"gave back a +{best * 100:.0f}% gain")

    rising = [d for d in ups if vol is not None
              and float(post["Volume"].iloc[d]) > float(post["Volume"].iloc[d - 1])]
    if rising:
        follow.append(f"{len(rising)} up close(s) on rising volume")
    if n >= 4 and sum(1 for d in ups if d <= 4) >= 3:
        follow.append("3 of the first 4 sessions up")
    if n >= 8 and sum(1 for d in ups if d <= 8) >= 6:
        follow.append("6 of the first 8 sessions up")
    if n >= 3 and upper_half > lower_half:
        follow.append(f"more upper-half closes ({upper_half} vs {lower_half})")
    judged = [d for d in downs if d + 2 <= n]
    if judged and all(max(c[d + 1], c[d + 2]) > c[d - 1] for d in judged):
        follow.append("every dip recovered within 2 sessions")

    return {"day_n": n, "violations": violations, "follow_through": follow,
            "sma20": sma20_note, "provisional_dropped": dropped}


def regime_tier(label) -> str:
    """The scan regime label's tier: ``strong``, ``weak``, ``off`` or ``unknown``.
    Matched by prefix. A substring test is wrong: "TRANSITIONAL" contains "on"."""
    s = str(label or "").strip().upper()
    if s.startswith("RISK-ON (STRONG)") or s.startswith("RISK-ON (MODERATE)"):
        return "strong"
    if s.startswith("RISK-OFF"):
        return "off"
    if s.startswith("RISK-ON") or s.startswith("TRANSITIONAL"):
        return "weak"
    return "unknown"


def _tape_tier(regime=None, spy_note=None):
    """(tier, label) from the scan regime dict, else from the trigger report's SPY note."""
    if isinstance(regime, dict) and regime.get("regime"):
        return regime_tier(regime["regime"]), str(regime["regime"])
    if isinstance(spy_note, dict) and spy_note.get("trend"):
        t = str(spy_note["trend"])
        tier = ("strong" if t.lower().startswith("bull")
                else "off" if t.lower().startswith("bear") else "weak")
        return tier, f"SPY {t}"
    return "unknown", ""


def weak_market_advice(regime=None, spy_note=None, *, stop_pct=None, target_pct=None,
                       risk_pct=None) -> Optional[str]:
    """The book's weak-tape numbers beside the plan's own, as one caption. None in a
    strong or unknown tape. Advice only."""
    tier, label = _tape_tier(regime, spy_note)
    if tier not in ("weak", "off"):
        return None
    lo, hi = doctrine.WEAK_TAPE_STOP_PCT
    tlo, thi = doctrine.WEAK_TAPE_TARGET_PCT
    plan_stop = f" (plan: {stop_pct * 100:.1f}%)" if stop_pct else ""
    plan_tgt = f" (plan target +{target_pct * 100:.0f}%)" if target_pct else ""
    plan_risk = f" (plan: {risk_pct:.2f}% risk per trade)" if risk_pct else ""
    return (f"{label} — in a weak market the book tightens up: stops {lo * 100:.0f}–"
            f"{hi * 100:.0f}% below the buy{plan_stop}, profits taken at {tlo * 100:.0f}–"
            f"{thi * 100:.0f}%{plan_tgt}, and smaller size{plan_risk}.")


def in_weak_take_profit_band(gain_pct) -> bool:
    """A gain inside the book's weak-market take-profit band (``WEAK_TAPE_TARGET_PCT``)."""
    lo, hi = doctrine.WEAK_TAPE_TARGET_PCT
    # +0.5 pt: a gain that rounds to 12% is in the band.
    return gain_pct is not None and lo <= float(gain_pct) <= hi + 0.005


def spy_confirm_streak(spy_df, max_days: Optional[int] = None,
                       phase2_by_date: Optional[dict] = None) -> Optional[dict]:
    """Consecutive settled sessions, newest first, with SPY in Stage 1 or 2.

    This is the backtest's re-entry lag. With ``phase2_by_date`` (``{'YYYY-MM-DD':
    phase2_pct}``) a session also needs breadth of at least ``BREADTH_MIN_PHASE2``, as the
    backtest required; a session missing from the map counts on SPY alone and sets
    ``partial``. Without the map the count is SPY only. Counting stops at ``max_days``
    (default ``REGIME_CONFIRM_DAYS``): older bars can't change the answer. Returns
    ``{streak, satisfied, phase_now, breadth, partial}``; None under 200 bars."""
    if spy_df is None or len(spy_df) < 200:
        return None
    import pandas as pd
    from src.stock_screener.minervini_screener.screening import classify_phase
    cap = doctrine.REGIME_CONFIRM_DAYS if max_days is None else int(max_days)
    streak, phase_now, partial = 0, None, False
    for k in range(cap):
        sub = spy_df.iloc[:len(spy_df) - k]
        if len(sub) < 200:
            break
        ph = classify_phase(sub, float(sub["Close"].iloc[-1])).get("phase")
        if k == 0:
            phase_now = ph
        if ph not in (1, 2):
            break
        if phase2_by_date is not None:
            day = pd.Timestamp(sub.index[-1]).strftime("%Y-%m-%d")
            p2 = phase2_by_date.get(day)
            if p2 is None:
                partial = True
            elif float(p2) < doctrine.BREADTH_MIN_PHASE2:
                break
        streak += 1
    return {"streak": streak, "satisfied": streak >= cap, "phase_now": phase_now,
            "breadth": phase2_by_date is not None, "partial": partial}


def market_turn(spy_note, prior_market: Optional[dict] = None,
                has_prior_plan: bool = True) -> dict:
    """Whether SPY entered Stage 4 since the previous plan.

    ``turn`` fires on the entering evening only. Firing every evening in Stage 4 would
    halve the book night after night. ``prior_market`` is the previous plan's ``market``
    record. Without one the turn can't be dated, so Stage 4 reads ``unconfirmed``.
    Returns ``{spy_phase, stage4, turn, unconfirmed}``."""
    ph = (spy_note or {}).get("phase") if isinstance(spy_note, dict) else None
    stage4 = ph == 4
    prev = (prior_market or {}).get("spy_phase")
    turn = bool(stage4 and has_prior_plan and prior_market is not None and prev != 4)
    unconfirmed = bool(stage4 and not turn and (not has_prior_plan or prior_market is None))
    return {"spy_phase": ph, "stage4": stage4, "turn": turn, "unconfirmed": unconfirmed}


DEPTH_VS_MARKET_MAX = 3.0   # the books: avoid a name that fell more than ~2.5-3x the market


def _naive_index(frame):
    import pandas as pd
    idx = pd.DatetimeIndex(frame.index)
    return idx.tz_localize(None) if idx.tz is not None else idx


def _max_drawdown(frame) -> Optional[float]:
    """The deepest fall from a running high, as a fraction: High for the highs, Low for
    the lows, Close for both when a frame lacks them. None on an empty frame."""
    import numpy as np
    if frame is None or not len(frame):
        return None
    hi = frame["High"] if "High" in frame.columns else frame["Close"]
    lo = frame["Low"] if "Low" in frame.columns else frame["Close"]
    run_max = hi.cummax().to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = 1.0 - lo.to_numpy(dtype=float) / run_max
    dd = dd[np.isfinite(dd)]
    return float(dd.max()) if len(dd) else None


def depth_vs_market(df, spy, contractions) -> Optional[dict]:
    """The base's depth against the market's decline over the same dates.

    The window runs from the first selected contraction's ``peak_date`` to the last
    bar. Both depths are the deepest fall from a running high inside it. Returns
    ``{depth_pct, spy_depth_pct, ratio, flag}`` (percent, ratio to 1 dp; ``flag`` at
    ``DEPTH_VS_MARKET_MAX``). None without contractions, without two bars of either
    frame in the window, or when the market fell under 1%: a ratio to a flat market
    means nothing."""
    if df is None or spy is None or not contractions:
        return None
    import pandas as pd
    try:
        start = pd.Timestamp(contractions[0]["peak_date"])
        if start.tzinfo is not None:
            start = start.tz_localize(None)
        w = df[_naive_index(df) >= start]
        s = spy[_naive_index(spy) >= start]
    except Exception:
        return None
    if len(w) < 2 or len(s) < 2:
        return None
    depth, spy_depth = _max_drawdown(w), _max_drawdown(s)
    if depth is None or spy_depth is None or spy_depth < 0.01:
        return None
    ratio = depth / spy_depth
    return {"depth_pct": round(depth * 100.0, 1), "spy_depth_pct": round(spy_depth * 100.0, 1),
            "ratio": round(ratio, 1), "flag": bool(ratio >= DEPTH_VS_MARKET_MAX)}


def depth_vs_market_text(depth: Optional[dict]) -> str:
    """One caption for :func:`depth_vs_market`; empty when unknown."""
    if not depth:
        return ""
    body = (f"Base depth **{depth['depth_pct']:.0f}%** while the market fell "
            f"{depth['spy_depth_pct']:.0f}% (**{depth['ratio']:.1f}×**)")
    if depth["flag"]:
        return ("⚠️ " + body + f" — more than {DEPTH_VS_MARKET_MAX:.0f}× the market's "
                "decline leaves more trapped sellers overhead than the books accept.")
    return body + "."


CONCENTRATION_NAMES = 3     # this many held names in one industry is a concentration
CONCENTRATION_SHARE = 0.5   # ...or this share of the book, once there are two or more
GROUP_LEADERS = 3           # the industry's top names by RS, whose breakdown is a warning


def industry_concentration(positions) -> Optional[str]:
    """A warning when the book leans on one industry: ``CONCENTRATION_NAMES`` names in it,
    or ``CONCENTRATION_SHARE`` of two or more positions. Counts names, not dollars.
    Positions without an industry are left out of both counts. None otherwise."""
    from collections import Counter
    inds = [p.get("industry") for p in positions or [] if p.get("industry")]
    if not inds:
        return None
    ind, n = Counter(inds).most_common(1)[0]
    m = len(positions)
    if n >= CONCENTRATION_NAMES or (m >= 2 and n / m >= CONCENTRATION_SHARE and n >= 2):
        return (f"{n} of your {m} positions are in **{ind}**. The books: a group moves "
                "together, and a leader's breakdown often takes its neighbours with it.")
    return None


def group_leader_break(industry, candidates, payloads, exclude=None) -> Optional[str]:
    """A warning when one of the industry's top ``GROUP_LEADERS`` scan names by RS (other
    than ``exclude``) closed below its 50-day on heavy volume. ``candidates`` is the scan
    table (``ticker``, ``rs``, ``industry``); ``payloads`` holds each name's frame. None
    without an industry, a scan, or a break."""
    if not industry or candidates is None or not len(candidates) \
            or "industry" not in candidates.columns:
        return None
    from .indicators import volume_ratio
    peers = candidates[(candidates["industry"] == industry)
                       & (candidates["ticker"] != exclude)]
    peers = peers.sort_values("rs", ascending=False).head(GROUP_LEADERS)
    for t in peers["ticker"]:
        df = (payloads.get(t) or {}).get("df")
        if df is None or len(df) < 51:
            continue
        close = df["Close"].astype(float)
        sma50 = float(close.tail(50).mean())
        vr = volume_ratio(df, doctrine.VOL_AVG_DAYS)
        if float(close.iloc[-1]) < sma50 and vr is not None and vr >= doctrine.VOL_CONFIRM_RATIO:
            return (f"group leader {t} closed below its 50-day on {vr:.1f}× volume — "
                    f"watch the rest of {industry}")
    return None


def stop_room_text(room: Optional[dict], day_range) -> str:
    """A caption fragment for a :func:`stop_room` result, e.g. ``'stop 3.1 typical days
    away (2.4%/day)'``. Prefixed ⚠ on ``warn``; empty when unknown."""
    if not room or not day_range:
        return ""
    head = "⚠ " if room["warn"] else ""
    tail = " — inside ordinary daily noise" if room["warn"] else ""
    return (f"{head}stop {room['room_days']:.1f} typical days away "
            f"({float(day_range) * 100:.1f}%/day){tail}")


REPORT_MAX_AGE_DAYS = 100   # older than this, the release is not "the last report" any more


def earnings_reaction(df, report_date, report_time=None) -> Optional[dict]:
    """How the stock took its last earnings release: ``{date, gap_pct, day_pct, vol_ratio,
    since_pct, flag}``.

    The reaction session is the release date's bar, or the next bar when the release came
    at or after 16:00 New York time (``report_time`` 'HH:MM'). ``day_pct`` is its close
    against the prior close and ``gap_pct`` its open; ``vol_ratio`` is its volume over the
    ``VOL_AVG_DAYS`` bars before it; ``since_pct`` is the last close against the reaction
    close. ``flag`` is ``"hard_drop"`` for a fall of ``EARNINGS_REACTION_PCT`` or more on
    ``VOL_CONFIRM_RATIO`` volume, ``"strong"`` for the same rise, else None. None when the
    date is missing, outside ``df``, its reaction bar hasn't printed, or the release is more
    than ``REPORT_MAX_AGE_DAYS`` before the last bar."""
    if df is None or not report_date or len(df) < 2:
        return None
    import pandas as pd
    from .indicators import prior_volume_average
    try:
        day = pd.Timestamp(report_date).normalize()
    except Exception:
        return None
    idx = _naive_index(df).normalize()
    if day < idx[0] or (idx[-1] - day).days > REPORT_MAX_AGE_DAYS:
        return None
    after_close = bool(report_time) and str(report_time) >= "16:00"
    later = (idx > day) if after_close else (idx >= day)
    if not later.any():
        return None
    pos = int(later.argmax())
    if pos == 0:
        return None
    close = df["Close"].astype(float).to_numpy()
    prev = close[pos - 1]
    if prev <= 0:
        return None
    avg = prior_volume_average(df["Volume"].astype(float), doctrine.VOL_AVG_DAYS).iloc[pos]
    vol_ratio = (float(df["Volume"].iloc[pos]) / float(avg)
                 if pd.notna(avg) and avg > 0 else None)
    day_pct = (close[pos] / prev - 1.0) * 100.0
    heavy = vol_ratio is not None and vol_ratio >= doctrine.VOL_CONFIRM_RATIO
    flag = None
    if heavy and day_pct <= -doctrine.EARNINGS_REACTION_PCT:
        flag = "hard_drop"
    elif heavy and day_pct >= doctrine.EARNINGS_REACTION_PCT:
        flag = "strong"
    return {"date": idx[pos].strftime("%Y-%m-%d"),
            "gap_pct": round((float(df["Open"].iloc[pos]) / prev - 1.0) * 100.0, 1),
            "day_pct": round(day_pct, 1),
            "vol_ratio": round(vol_ratio, 1) if vol_ratio is not None else None,
            "since_pct": round((close[-1] / close[pos] - 1.0) * 100.0, 1),
            "flag": flag}


def earnings_reaction_text(reaction: Optional[dict], report_date=None,
                           report_time=None) -> str:
    """A Step-2 caption for an :func:`earnings_reaction` result, e.g. ``'Last report
    2026-08-06 (after the close): -9.2% on 3.4× volume (gap -7.0%) ⚠️ …; since then
    +4.1%'``. Empty when unknown."""
    if not reaction:
        return ""
    when = ""
    if report_time:
        when = (" (after the close)" if str(report_time) >= "16:00"
                else " (before the open)" if str(report_time) < "09:30"
                else " (during the session)")
    vol = ("" if reaction.get("vol_ratio") is None
           else f" on {reaction['vol_ratio']:.1f}× volume")
    note = {"hard_drop": " ⚠️ a hard drop on heavy volume; the books say big money knew "
                         "something",
            "strong": " ✅ a strong move on heavy volume"}.get(reaction.get("flag"), "")
    return (f"**Last report** {report_date or reaction['date']}{when}: "
            f"{reaction['day_pct']:+.1f}%{vol} (gap {reaction['gap_pct']:+.1f}%){note}; "
            f"since then {reaction['since_pct']:+.1f}%")
