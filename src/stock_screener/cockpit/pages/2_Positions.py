"""Positions page: stop management and manual sells for the Minervini Trader Alpaca paper
account (simulated money, no real trades).

A page of its own, so it loads instantly without the multi-minute scan. It is the page
used each day to manage stops. It shows each holding's P&L, whether a stop is in place,
its stop stage (initial, breakeven, trailing and so on), next earnings date, the sell
pillars (four keep-or-sell checks, P1-P4) and Minervini's exit advice, including the rules
for holding through an earnings report. One click places or raises every stop as a GTC
order; a stop only ever moves up. Each position has a manual market SELL with a two-step
confirm: cancel stops, sell, then place the stop again for any shares left, at the same
level. The page never sells on its own.

Run the app from the project root: ``streamlit run src/stock_screener/cockpit/app.py`` and pick
"Positions" from the page nav.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The cockpit imports need the repo root on sys.path. From pages/: pages=0, cockpit=1,
# stock_screener=2, src=3, root=4.
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.stock_screener.cockpit import advisories  # noqa: E402 (SEPA checks, shown only)
from src.stock_screener.cockpit import cache  # noqa: E402
from src.stock_screener.cockpit import journal_cache  # noqa: E402 (shared filled-orders cache)
from src.stock_screener.cockpit import scan_worker  # noqa: E402
from src.stock_screener.cockpit import sells  # noqa: E402 (evening sell plan + veto)
from src.stock_screener.cockpit import trade  # noqa: E402 (module import: tests can replace it)
from src.stock_screener.cockpit import triggers  # noqa: E402 (SPY status when there is no scan)
from src.stock_screener.cockpit.export import load_watchlist  # noqa: E402

# Start the stock scan in the background, so it is already running when the user opens
# the scan page. Does nothing under AppTest.
scan_worker.autostart()

st.set_page_config(page_title="Positions", page_icon="📊", layout="wide")
# Same reduced padding as app.py, so the first row isn't hidden under Streamlit's header.
st.markdown(
    "<style>.block-container{padding-top:4rem;padding-bottom:2rem;}"
    'div[data-testid="stVerticalBlock"]{gap:0.6rem;}</style>',
    unsafe_allow_html=True)

_BASIS_LABELS = {
    "auto": "Auto (the books' rules)",
    "initial": "Initial (8% below entry)",
    "breakeven": "Breakeven (entry price)",
    "sma50": "Trail 50-day SMA",
}
_ICONS = {"submitted": "✅", "stop_only": "🛑", "stop_kept": "🔒", "skipped": "—", "failed": "⚠️"}


POS_MAX_AGE_S = 60      # seconds a stored account read is reused before it is re-read


def _session_positions(nonce):
    """Read the account, keeping the result in ``pos_memo`` in session_state for this
    browser session. ``nonce`` is ``pos_nonce``, a counter that is bumped to force a
    re-read. Returns ``(data, read_at)``. Raises ``TradeUnavailable`` when the account
    can't be read; the error is not stored, so Refresh retries.

    The read MUST be stored in session_state, never ``st.cache_data``. That cache is shared
    by every visitor, and every session starts with the counter at 1, so every visitor
    would get the first visitor's result, never refreshed, and wait on its download. Widget
    reruns reuse the read for ``POS_MAX_AGE_S``. session_state survives page switches, so
    the age limit is what catches trades placed elsewhere. Refresh, sell and the Re-arm
    button bump the counter."""
    import time
    memo = st.session_state.get("pos_memo")
    if (memo is not None and memo["nonce"] == nonce
            and time.monotonic() - memo["mono"] < POS_MAX_AGE_S):
        return memo["data"], memo["read_at"]
    with st.spinner("Reading the paper account…"):
        data = trade.fetch_positions()          # looked up on the module, so tests can replace it
    import pandas as pd
    read_at = pd.Timestamp.now(tz="America/New_York")
    st.session_state["pos_memo"] = {"nonce": nonce, "mono": time.monotonic(),
                                    "data": data, "read_at": read_at}
    return data, read_at


def _do_rearm(positions, nonce, basis):
    """Callback for the Re-arm button: place or raise a GTC stop at the stop typed in each
    row (a stop only ever moves up), then bump ``pos_nonce`` so the next run reads the new
    stops."""
    targets = [{"ticker": p["symbol"],
                "stop_price": st.session_state.get(f"posstop_{p['symbol']}_{nonce}_{basis}"),
                "price": p["current_price"]}
               for p in positions]
    try:
        st.session_state["rearm_result"] = trade.rearm_stops(targets)
    except trade.TradeUnavailable as e:
        st.session_state["rearm_result"] = {"error": str(e)}
    st.session_state["pos_nonce"] = st.session_state.get("pos_nonce", 1) + 1


def _set_sell_qty(sym, nonce, value):
    """Callback for the ¼/½/All buttons. Streamlit lets a callback set another widget's
    value; the page body can't."""
    st.session_state[f"sellqty_{sym}_{nonce}"] = int(value)


def _cancel_sell():
    st.session_state.pop("sell_pending", None)


def _do_sell(symbol, qty, remainder_stop=None):
    """Callback for Confirm sell: sell the quantity saved when Sell was pressed, not the
    current value of the quantity box, then bump ``pos_nonce`` so the next run shows the
    smaller position and the re-placed stop. ``remainder_stop`` is a stop at the entry
    price for the shares left, so they can't lose. ``submit_position_sell`` only ever
    moves a stop up, so it can't lower the stop in force."""
    try:
        st.session_state["sell_result"] = trade.submit_position_sell(
            symbol, qty, remainder_stop=remainder_stop)
    except trade.TradeUnavailable as e:
        st.session_state["sell_result"] = {"error": str(e)}
    st.session_state.pop("sell_pending", None)
    st.session_state["pos_nonce"] = st.session_state.get("pos_nonce", 1) + 1


def _do_veto(symbol):
    """Callback for Veto: mark ``symbol``'s planned sell as vetoed (never sent) in the plan
    file. The plan is read again first, because the evening command-line run or another
    session may have rewritten it since this page was drawn. Never raises: an error is
    stored in ``veto_error`` for the page to show."""
    try:
        p = sells.load_latest_sell_plan()
        if p and sells.veto_order(p, symbol):
            sells.save_sell_plan(p)
    except Exception as e:
        st.session_state["veto_error"] = str(e)


def _earnings_cell(p) -> str:
    """Earnings cell for the table: 'YYYY-MM-DD (Nd)', marked ⚠ when the report is ~21
    days away or less; empty when unknown. It repeats the wording of app.py's trade panel
    because importing app.py would run the whole scan page."""
    ne, ei = p.get("next_earnings"), p.get("earnings_in")
    if not ne and ei is None:
        return ""
    warn = "⚠︎ " if (ei is not None and 0 <= ei <= trade.EARNINGS_SOON_DAYS) else ""
    days = f" ({int(ei)}d)" if ei is not None else ""
    return f"{warn}{ne or '?'}{days}"


# --------------------------------------------------------------------------- #
st.title("📊 Positions — stop management")
st.caption("Minervini Trader **paper** account · your daily stop check. Educational only — "
           "not financial advice; you place/confirm orders yourself.")

if "pos_nonce" not in st.session_state:
    st.session_state.pos_nonce = 1
if st.button("🔄 Refresh"):
    st.session_state.pos_nonce += 1

try:
    data, read_at = _session_positions(st.session_state.pos_nonce)
except trade.TradeUnavailable as e:
    st.warning(str(e))
    st.stop()

acct = data["account"]
positions = data["positions"]

# --- Account headline tiles ----------------------------------------------------------------- #
# A metric value MUST hold at most one '$': with two, the text between them renders as
# LaTeX math.
m = st.columns(4)
m[0].metric("Equity", f"${acct['equity']:,.0f}", border=True)
m[1].metric("Cash", f"${acct['cash']:,.0f}", border=True)
m[2].metric("Positions", str(acct["positions_count"]), border=True)
_pl = acct.get("total_unrealized_pl") or 0.0
m[3].metric("Unrealized P&L", f"${_pl:,.0f}", border=True)

_src = ("Minervini Trader keys" if acct.get("using_dedicated")
        else "shared ALPACA_* keys — set ALPACA_API_KEY_MINERVINI / "
             "ALPACA_API_KEY_SECRET_MINERVINI to use the Minervini account")
st.caption(f"Account **…{str(acct['account_number'])[-4:]}** ({_src}) · "
           f"as of {read_at:%H:%M:%S} ET")

if not positions:
    st.info("No open positions in this account.")
    st.stop()

# A waiting sell confirmation for a symbol no longer held (sold elsewhere) is out of date.
_sp = st.session_state.get("sell_pending")
if _sp and _sp.get("symbol") not in {p["symbol"] for p in positions}:
    st.session_state.pop("sell_pending", None)

# --- Sell pillars (P1-P4) ------------------------------------------------------------------- #
# Any input can be missing; a pillar with no data shows "—".
_open_by_sym = {}
_derived_pct = _avg_win = None
try:
    # Entry dates for P1 come from the filled-orders cache shared with the Journal page,
    # read under that page's ``jr_nonce`` (its re-read counter). If Alpaca can't be
    # reached, the pillars that need them show "—" and the page still loads.
    _fills = journal_cache.cached_fills(st.session_state.get("jr_nonce", 1))["fills"]
    _journal = trade.build_trade_journal(_fills)
    _open_by_sym = {r["symbol"]: r for r in _journal["open"]}
    # The stop percentage the trade plan uses, so the "initial" stop basis and R (gain as
    # a multiple of the initial risk) match what was ordered.
    _derived = trade.derived_stop_pct(_journal["closed"])
    _derived_pct = _derived["stop_pct"]
    # The stop moves based on the average win need as many wins as ``derived_stop_pct``.
    _avg_win = _derived["avg_win_pct"] if _derived_pct is not None else None
except Exception:
    _open_by_sym = {}
try:
    _wl_pivots = {e["ticker"]: e.get("judged_pivot")
                  for e in load_watchlist(cache.WATCHLIST_JSON)}
except Exception:
    _wl_pivots = {}
_regime = _spy = _latest = None
_rs_map = {}
try:
    # The newest scan result: None before any scan has finished, and under AppTest.
    _latest = scan_worker.get_worker().latest()
    _regime = getattr(_latest, "regime", None)
    _rs_map = getattr(_latest, "rs_ratings", None) or {}     # missing from older pickled scans
except Exception:
    _regime = None
if _regime is None:
    try:
        _spy = (triggers.load_latest_trigger_report() or {}).get("spy")
    except Exception:
        _spy = None
_pillars = {p["symbol"]: trade.sell_pillars(
                p, entry_date=(_open_by_sym.get(p["symbol"]) or {}).get("entry_date"),
                pivot=_wl_pivots.get(p["symbol"]), regime=_regime, spy_note=_spy,
                rs=_rs_map.get(p["symbol"]))
            for p in positions}

# --- The market: SPY in Stage 4 and weak-market advice (display only) ----------------------- #
try:
    _mk = (sells.load_latest_sell_plan() or {}).get("market") or {}
except Exception:
    _mk = {}
if _mk.get("turn"):
    st.error("**Market turn:** SPY closed in Stage 4 on the last evening plan. Hold less "
             "stock and buy nothing until SPY recovers (see the plan notes below).")
elif _mk.get("spy_phase") == 4:
    st.warning("**SPY is in Stage 4** — be careful; no new buys.")
_weak_tape = advisories.weak_market_advice(_regime, _spy, target_pct=0.25)
if _weak_tape:
    st.warning(_weak_tape)
_conc = advisories.industry_concentration(positions)
if _conc:
    st.warning(_conc)
_leader_breaks = {}
for p in positions:
    try:
        _lb = advisories.group_leader_break(
            p.get("industry"), getattr(_latest, "candidates", None),
            getattr(_latest, "payloads", None) or {}, exclude=p["symbol"])
    except Exception:
        _lb = None
    if _lb:
        _leader_breaks[p["symbol"]] = _lb
_PICON = {"ok": "✅", "warn": "⚠️", "fail": "❌", "unknown": "—"}
# '~' = the initial risk is an estimate (see trade.r_multiple).
_rmults = {p["symbol"]: trade.r_multiple(p["avg_entry"], p["current_price"],
                                         _wl_pivots.get(p["symbol"]),
                                         current_stop=p.get("current_stop"),
                                         stop_pct=_derived_pct)
           for p in positions}
_strength = {}
for p in positions:
    try:
        _strength[p["symbol"]] = advisories.strength_signs(
            p.get("df"), (_open_by_sym.get(p["symbol"]) or {}).get("entry_date"),
            avg_entry=p.get("avg_entry"))
    except Exception:
        _strength[p["symbol"]] = None
# Stop stage and advisories again, now with R, the average win and ``strength_signs``
# (signs of a fast run-up, when the books sell part).
for p in positions:
    _book = {"r_now": _rmults[p["symbol"]][0], "avg_win": _avg_win}
    p["stage"] = trade.position_stage(p["gain_pct"], avg_entry=p["avg_entry"],
                                      sma_50=p["sma_50"], **_book)
    p["advisories"] = trade.position_advisories(
        {**p, **_book, "strength_signs": (_strength.get(p["symbol"]) or {}).get("signs")})


def _r_cell(sym) -> str:
    _r, _approx = _rmults.get(sym, (None, True))
    return "" if _r is None else f"{'~' if _approx else ''}{_r:.1f}"

# --- Positions table ------------------------------------------------------------------------ #
import pandas as pd  # noqa: E402

rows = [{
    "symbol": p["symbol"], "qty": p["qty"], "avg_entry": p["avg_entry"],
    "current_price": p["current_price"],
    "gain_pct": (p["gain_pct"] * 100.0 if p["gain_pct"] is not None else None),
    "R": _r_cell(p["symbol"]),
    "stage": p.get("stage") or "",
    "P1": _PICON[_pillars[p["symbol"]]["P1"]["status"]],
    "P2": _PICON[_pillars[p["symbol"]]["P2"]["status"]],
    "P3": _PICON[_pillars[p["symbol"]]["P3"]["status"]],
    "P4": _PICON[_pillars[p["symbol"]]["P4"]["status"]],
    "market_value": p["market_value"], "unrealized_pl": p["unrealized_pl"],
    "current_stop": p["current_stop"], "sma_20": p.get("sma_20"), "sma_50": p["sma_50"],
    "earnings": _earnings_cell(p),
    "advisories": " · ".join(p["advisories"]) if p["advisories"] else "",
} for p in positions]
_num = st.column_config.NumberColumn
col_config = {
    "symbol": st.column_config.Column("Ticker"),
    "qty": _num("Shares", format="%d"),
    "avg_entry": _num("Avg entry", format="$%.2f"),
    "current_price": _num("Price", format="$%.2f"),
    "gain_pct": _num("Gain", format="%.1f%%",
                     help="Gain or loss on the position so far (shares not yet sold)."),
    "R": st.column_config.Column(
        "R", help="Gain as a multiple of the initial risk. Exact when the stop in force "
                  "is the one the trade plan set (from the pivot saved on the watchlist, "
                  "raised to 10% below the buy price); '~' = an estimate — the stop has "
                  "been raised since, or the stock isn't on the watchlist (8% below the "
                  "entry). At ≥2R: sell half and move the stop to breakeven, so the rest "
                  "can't lose."),
    "stage": st.column_config.Column(
        "Stop stage", help="Where the stop is, by the books' rules: underwater (at a loss) "
                           "· initial (no rule met yet) · breakeven (the rising 50-day "
                           "reached your cost, 3R, or 2× your average win) · back stop (at "
                           "your average win) · trailing 50-day (the 50-day is above your "
                           "cost). The average-win rules start at 5 wins."),
    "P1": st.column_config.Column(
        "P1", help="Is the breakout holding: a close below the pivot on breakout day, a "
                   "clear close below the pivot, a close below the breakout day's low once "
                   "back under the pivot (above it, a warning), and a slow start "
                   "(not ~3% up by ~day 10; flat or down by day 15). ⚠ also lists "
                   "warning signs since the breakout: a close under the 20-day line in the "
                   "first month, a heavy-volume down day after a light-volume breakout, "
                   "lower lows, more down days or closes in the lower half of the day's "
                   "range, a gain given back. Needs the entry date from the journal; the "
                   "pivot checks need the pivot saved on the watchlist."),
    "P2": st.column_config.Column(
        "P2", help="Minervini's Trend Template on this stock: the seven price checks "
                   "plus an RS rating ≥ 70 from the last scan. Strict: anything under "
                   "8/8 is a fail (an SMA check that is very close can flip for a day, and "
                   "RS is a rank that can dip under 70 for a day). Seven passing with no "
                   "scan shows —."),
    "P3": st.column_config.Column(
        "P3", help="The market: the scan's market reading when there is one, else the "
                   "trigger report's SPY check. When the market is risk-off (weak), every "
                   "⚠️ counts as ❌."),
    "P4": st.column_config.Column(
        "P4", help="Earnings: within ~21 days of the report, a loss or a gain under 8% "
                   "fails (the price can open far below a stop after a report); with a "
                   "bigger gain, still cut to a size you can hold through it."),
    "market_value": _num("Market value", format="$%.0f"),
    "unrealized_pl": _num("Unrealized P&L", format="$%.0f"),
    "current_stop": _num("Stop", format="$%.2f",
                         help="The highest stop order in place (blank = no stop)."),
    "sma_20": _num("20-day SMA", format="$%.2f",
                   help="A fresh breakout should hold its 20-day line for the first ~month; "
                        "a close below it is a warning sign — a reason to doubt the breakout."),
    "sma_50": _num("50-day SMA", format="$%.2f",
                   help="Once the rising 50-day passes your cost, the stop follows just "
                        "under it; a close below it on "
                        "heavy volume is a sign to sell."),
    "earnings": st.column_config.Column(
        "Earnings", help="Next earnings report. ⚠︎ within ~21 days: a stop can't protect "
                         "you, since the price can open far below it after the report — "
                         "hold through only with a gain of ~8%+, otherwise sell part or all "
                         "before the report."),
    "advisories": st.column_config.Column("Advice", width="large"),
}
col_order = ["symbol", "qty", "avg_entry", "current_price", "gain_pct", "R", "stage",
             "P1", "P2", "P3", "P4",
             "market_value", "unrealized_pl", "current_stop", "sma_20", "sma_50", "earnings",
             "advisories"]
st.dataframe(pd.DataFrame(rows), column_config=col_config, column_order=col_order,
             hide_index=True, width="stretch")
st.caption("Sell pillars (four keep-or-sell checks) — P1 breakout holding · P2 Trend "
           "Template · P3 market · P4 earnings. Any ❌ means the reason to own the stock is "
           "gone; the stop only guards against disaster between checks. — = not enough "
           "data (journal / saved pivot / scan).")

# --- Planned auto-sells: evening plan → overnight veto → morning submit --------------------- #
_plan = None
try:
    _plan = sells.load_latest_sell_plan()
except Exception:
    _plan = None
if _plan and _plan.get("orders"):
    st.markdown("#### Planned auto-sells")
    _armed = ("**ARMED** (on) — orders still planned are sent at ~9:25 ET"
              if sells.autosell_enabled()
              else "not armed (off: `AUTOSELL` unset) — plan only, nothing is sent")
    st.caption(f"Evening plan **{_plan.get('date')}** · morning send {_armed}. "
               "Veto any order you disagree with before the market opens; a vetoed order "
               "is never sent (its stop order stays in place).")
    _OICON = {"planned": "🕗", "vetoed": "🚫", "submitted": "✅",
              "failed": "⚠️", "skipped": "—"}
    for o in _plan["orders"]:
        c1, c2 = st.columns([5, 1])
        _part = " (partial)" if o.get("exit") == "partial" else ""
        c1.caption(f"{_OICON.get(o.get('status'), '•')} **{o['symbol']}** ×{o['qty']}{_part} "
                   f"— {o.get('status')} · " + "; ".join(o.get("reasons", []))
                   + (f" · {o['detail']}" if o.get("detail") else ""))
        if o.get("status") == sells.ORDER_PLANNED:
            c2.button("Veto", key=f"veto_{o['symbol']}_{_plan.get('date')}",
                      width="stretch", on_click=_do_veto, args=(o["symbol"],))
    if st.session_state.get("veto_error"):
        st.error(f"Veto failed: {st.session_state.pop('veto_error')}")
# Notes are the warnings that aren't orders. They MUST show with no orders too: most
# evenings have none.
if _plan and _plan.get("notes"):
    with st.expander(f"Evening plan notes — {_plan.get('date')} ({len(_plan['notes'])})",
                     expanded=not _plan.get("orders")):
        for _n in _plan["notes"]:
            st.caption(_n)

# --- Stop management: stop basis, an editable stop per row, the Re-arm button --------------- #
st.markdown("#### 🛡️ Stops & sells")
st.caption("A GTC stop only ever moves UP — a suggested stop below the one already in "
           "place is ignored. Edit any row before pressing Re-arm (place or raise the "
           "stops). Each row's Sell section sells part or all at market (two-step confirm); "
           "after a partial sell, the stop is placed again for the shares left, at the same "
           "level.")
basis = st.radio("Stop basis", trade.STOP_BASES, horizontal=True, key="pos_basis",
                 format_func=lambda b: _BASIS_LABELS.get(b, b),
                 help="How each row's suggested new stop is chosen. 'Auto' uses the books' "
                      "rules: breakeven once the rising 50-day reaches your cost, at 3R, "
                      "or at 2× your average win; a back stop at your average win; then a "
                      "stop that follows just under the 50-day. With no rule met it stays "
                      "at the initial stop.")
_nonce = st.session_state.pos_nonce
for p in positions:
    sym, price = p["symbol"], p["current_price"]
    suggested, eff = trade.suggest_stop(
        avg_entry=p["avg_entry"], current_price=price, sma_50=p["sma_50"],
        current_stop=p["current_stop"], gain_pct=p["gain_pct"], basis=basis,
        initial_pct=_derived_pct, r_now=_rmults[sym][0], avg_win=_avg_win)
    seed = suggested if suggested is not None else (p["current_stop"] or 0.0)
    cA, cB = st.columns([3, 2])
    _g = f"{p['gain_pct'] * 100:+.1f}%" if p["gain_pct"] is not None else "n/a"
    _eff = ((f" · {_BASIS_LABELS[eff].split(' (')[0].lower()}" if eff in _BASIS_LABELS
             else f" · {eff}") if basis == "auto" else "")
    cA.caption(f"• **{sym}** {p['qty']} sh · {_g}{_eff}"
               + (f" · {' · '.join(p['advisories'])}" if p["advisories"] else ""))
    _flagged = [(k, v) for k, v in _pillars.get(sym, {}).items()
                if v["status"] in ("fail", "warn")]
    if _flagged:
        cA.caption("  ↳ " + " · ".join(f"{k} {_PICON[v['status']]} {v['detail']}"
                                       for k, v in _flagged))
    # P1 already shows the bad signs after the breakout; this adds the good ones.
    _pb = advisories.post_breakout_read(
        p.get("df"), (_open_by_sym.get(sym) or {}).get("entry_date"),
        avg_entry=p.get("avg_entry"), below_sma50=bool(p.get("below_sma50")),
        volume_ratio=p.get("volume_ratio"))
    if _pb and _pb["follow_through"]:
        cA.caption(f"  ↳ ✅ good signs since the breakout (day {_pb['day_n']}): "
                   + " · ".join(_pb["follow_through"]))
    _signs = (_strength.get(sym) or {}).get("signs")
    if _signs:
        cA.caption(f"  ↳ 📈 selling-into-strength signs ({len(_signs)}): "
                   + " · ".join(_signs) + " — the books sell part while the price is rising")
    if sym in _leader_breaks:
        cA.caption(f"  ↳ ⚠️ {_leader_breaks[sym]}")
    if _weak_tape and advisories.in_weak_take_profit_band(p.get("gain_pct")):
        cA.caption(f"  ↳ weak market: the books take profits at 10–12% — this one is up "
                   f"{p['gain_pct'] * 100:.1f}%")
    cB.number_input(f"stop {sym}", min_value=0.0, value=float(seed), step=0.01, format="%.2f",
                    key=f"posstop_{sym}_{_nonce}_{basis}", label_visibility="collapsed")
    _ed = st.session_state.get(f"posstop_{sym}_{_nonce}_{basis}", seed)
    if not trade.stop_is_valid(_ed, price):
        cB.caption(":red[stop must be below the price — set it by hand]")
    elif price:
        _dr = advisories.typical_day_range(p.get("df"))
        _room = advisories.stop_room(_dr, _ed, price)
        cA.caption(f"  ↳ drop to the stop ≈{(price - _ed) / price * 100:.1f}%"
                   + (f" · {advisories.stop_room_text(_room, _dr)}" if _room else ""))

    # --- Manual sell (market, paper, two-step confirm) ------------------------------------ #
    _held = int(p["qty"] or 0)
    if _held >= 1:
        with st.expander(f"Sell {sym} (market order, paper account)"):
            sc = st.columns([2, 1, 1, 1])
            # The default MUST be set through session_state, not value=: the ¼/½/All
            # buttons also write this key, and a widget given both logs a warning.
            _qkey = f"sellqty_{sym}_{_nonce}"
            if _qkey not in st.session_state:
                st.session_state[_qkey] = _held
            sc[0].number_input(f"shares to sell {sym}", min_value=1, max_value=_held,
                               step=1, key=_qkey, label_visibility="collapsed")
            sc[1].button("¼", key=f"sellq4_{sym}_{_nonce}", width="stretch",
                         on_click=_set_sell_qty, args=(sym, _nonce, max(1, _held // 4)))
            sc[2].button("½", key=f"sellq2_{sym}_{_nonce}", width="stretch",
                         on_click=_set_sell_qty, args=(sym, _nonce, max(1, _held // 2)))
            sc[3].button("All", key=f"sellqa_{sym}_{_nonce}", width="stretch",
                         on_click=_set_sell_qty, args=(sym, _nonce, _held))
            # At 2R or more with the stop below the entry price: sell half and move the
            # stop on the rest to the entry price, so the rest can't lose.
            _rv, _rapprox = _rmults.get(sym, (None, True))
            if (_rv is not None and _rv >= trade.FREE_ROLL_R
                    and (p["current_stop"] is None
                         or p["current_stop"] < p["avg_entry"])):
                if _held >= 2:
                    if st.button(f"Sell ½ {sym} + move stop to breakeven",
                                 key=f"froll_{sym}_{_nonce}", width="stretch"):
                        st.session_state["sell_pending"] = {
                            "symbol": sym, "qty": _held // 2, "held": _held,
                            "stop": p["current_stop"],
                            "remainder_stop": p["avg_entry"]}
                else:
                    st.caption("≥2R on a single share — no half to sell; raise the "
                               "stop to breakeven instead (stop box above, then Re-arm).")
            if st.button(f"Sell at market — {sym}", key=f"sell_{sym}_{_nonce}",
                         width="stretch"):
                # Save the quantity now: Confirm MUST sell what was shown, not a quantity
                # edited while the warning is open.
                st.session_state["sell_pending"] = {
                    "symbol": sym, "qty": int(st.session_state.get(
                        f"sellqty_{sym}_{_nonce}", _held)),
                    "held": _held, "stop": p["current_stop"]}
            _sp = st.session_state.get("sell_pending")
            if _sp and _sp.get("symbol") == sym:
                _q, _stop = _sp["qty"], _sp.get("stop")
                _rem = _sp["held"] - _q
                _rstop = _sp.get("remainder_stop")
                if _rstop is not None and _rem > 0:
                    _sfx = (f"the remaining {_rem} sh get a stop at breakeven @ "
                            f"{_rstop:.2f} (stops only move up: the current stop is kept "
                            "if higher)")
                elif _stop is None:
                    _sfx = (f"no stop is in place; the remaining {_rem} sh stay unprotected"
                            if _rem > 0 else "no stop is in place")
                elif _rem > 0:
                    _sfx = (f"stop @ {_stop:.2f} is cancelled, then placed again for the "
                            f"remaining {_rem} sh at the same level")
                else:
                    _sfx = f"stop @ {_stop:.2f} is cancelled (no shares remain)"
                st.warning(f"Sell **{_q}/{_sp['held']} sh {sym}** at market (day order)? "
                           f"{_sfx}.")
                cc1, cc2 = st.columns(2)
                cc1.button("✅ Confirm sell", key=f"sellgo_{sym}_{_nonce}", type="primary",
                           width="stretch", on_click=_do_sell, args=(sym, _q, _rstop))
                cc2.button("Cancel", key=f"sellno_{sym}_{_nonce}", width="stretch",
                           on_click=_cancel_sell)

st.button("🛡️ Re-arm: place or raise all stops", type="primary", width="stretch",
          on_click=_do_rearm, args=(positions, _nonce, basis))

_sr = st.session_state.get("sell_result")
if _sr:
    if _sr.get("error"):
        st.error(f"Sell failed: {_sr['error']}")
    else:
        ic = _ICONS.get(_sr.get("status"), "•")
        st.success(f"Sell order on account …{str(_sr.get('account_number'))[-4:]} · "
                   f"equity ${_sr.get('equity', 0):,.0f}")
        st.caption(f"{ic} {_sr.get('symbol')}: {_sr.get('status')} — {_sr.get('detail', '')}")

_rr = st.session_state.get("rearm_result")
if _rr:
    if _rr.get("error"):
        st.error(f"Re-arm failed: {_rr['error']}")
    else:
        _act = [r for r in _rr["results"] if r["status"] in ("stop_only", "stop_kept")]
        st.success(f"Placed or kept {len(_act)}/{len(_rr['results'])} stop(s) on "
                   f"account …{str(_rr['account_number'])[-4:]} · equity ${_rr['equity']:,.0f}")
        for r in _rr["results"]:
            ic = _ICONS.get(r["status"], "•")
            st.caption(f"{ic} {r['ticker']}: {r['status']} — {r.get('detail', '')}")

st.caption("Educational tool — not financial advice. Stops are GTC (good till cancelled) "
           "orders on the paper account.")
