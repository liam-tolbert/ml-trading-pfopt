"""Positions page — stop management + manual sells for the Minervini Trader Alpaca paper account.

Separate from the scan page so it loads instantly (no multi-minute scan) — the daily
stop-management surface. Shows each holding's P&L, protective-stop status, stage on the stop
ladder, next earnings date, and Minervini exit advisories (incl. the earnings-cushion rules);
one-click "re-arm / raise all stops" via the GTC one-way ratchet (never lowers a stop); and a
per-position manual market SELL with a two-step confirm — cancel stops → sell → re-place the
stop for any remainder at the same level (the app never sells on its own).

Run the app from the project root: ``streamlit run src/stock_screener/cockpit/app.py`` and pick
"Positions" from the page nav.
"""
from __future__ import annotations

import sys
from pathlib import Path

# This page imports cockpit modules, so the repo ROOT must be on sys.path. From pages/:
# pages=0, cockpit=1, stock_screener=2, src=3, root=4.
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.stock_screener.cockpit import trade  # noqa: E402 (module import → patchable in tests)

st.set_page_config(page_title="Positions", page_icon="📊", layout="wide")
# Mirror app.py's padding trim so the first row isn't buried under Streamlit's header.
st.markdown(
    "<style>.block-container{padding-top:4rem;padding-bottom:2rem;}"
    'div[data-testid="stVerticalBlock"]{gap:0.6rem;}</style>',
    unsafe_allow_html=True)

_BASIS_LABELS = {
    "auto": "Auto (by gain/stage)",
    "initial": "Initial (8% below entry)",
    "breakeven": "Breakeven (entry)",
    "sma50": "Trail 50-day SMA",
}
_ICONS = {"submitted": "✅", "stop_only": "🛑", "stop_kept": "🔒", "skipped": "—", "failed": "⚠️"}


@st.cache_data(show_spinner="Reading the paper account…")
def _cached_positions(nonce):
    # Reference the MODULE attribute (trade.fetch_positions) so a test patch is honored; the
    # nonce lets Refresh / re-arm bust the cache. TradeUnavailable isn't cached, so a
    # credentials fix + Refresh recovers.
    return trade.fetch_positions()


def _do_rearm(positions, nonce, basis):
    """Re-arm button callback: read each row's edited stop from session_state, raise/place GTC
    stops via the ratchet, then bust the cache so the next run reflects the new stops."""
    targets = [{"ticker": p["symbol"],
                "stop_price": st.session_state.get(f"posstop_{p['symbol']}_{nonce}_{basis}"),
                "price": p["current_price"]}
               for p in positions]
    try:
        st.session_state["rearm_result"] = trade.rearm_stops(targets)
    except trade.TradeUnavailable as e:
        st.session_state["rearm_result"] = {"error": str(e)}
    st.session_state["pos_nonce"] = st.session_state.get("pos_nonce", 1) + 1
    _cached_positions.clear()


def _set_sell_qty(sym, nonce, value):
    """Preset (¼/½/All) callback — another widget's key may only be set inside a callback."""
    st.session_state[f"sellqty_{sym}_{nonce}"] = int(value)


def _cancel_sell():
    st.session_state.pop("sell_pending", None)


def _do_sell(symbol, qty):
    """Confirm-sell callback: submits the FROZEN pending quantity (not the live qty widget),
    then busts the cache so the next run shows the reduced position + re-placed stop."""
    try:
        st.session_state["sell_result"] = trade.submit_position_sell(symbol, qty)
    except trade.TradeUnavailable as e:
        st.session_state["sell_result"] = {"error": str(e)}
    st.session_state.pop("sell_pending", None)
    st.session_state["pos_nonce"] = st.session_state.get("pos_nonce", 1) + 1
    _cached_positions.clear()


def _earnings_cell(p) -> str:
    """Earnings display for the table: 'YYYY-MM-DD (Nd)', ⚠-marked inside the ~21-day
    no-cushion window. (Same wording convention as the app's trade panel — re-derived here;
    importing app.py would execute the whole scan page.)"""
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
    _cached_positions.clear()

try:
    data = _cached_positions(st.session_state.pos_nonce)
except trade.TradeUnavailable as e:
    st.warning(str(e))
    st.stop()

acct = data["account"]
positions = data["positions"]

# --- Account headline tiles (one '$' per metric value — two render as a LaTeX math span) ---- #
m = st.columns(4)
m[0].metric("Equity", f"${acct['equity']:,.0f}", border=True)
m[1].metric("Cash", f"${acct['cash']:,.0f}", border=True)
m[2].metric("Positions", str(acct["positions_count"]), border=True)
_pl = acct.get("total_unrealized_pl") or 0.0
m[3].metric("Unrealized P&L", f"${_pl:,.0f}", border=True)

_src = ("Minervini Trader keys" if acct.get("using_dedicated")
        else "shared ALPACA_* keys — set ALPACA_API_KEY_MINERVINI / "
             "ALPACA_API_KEY_SECRET_MINERVINI to target the Minervini account")
st.caption(f"Account **…{str(acct['account_number'])[-4:]}** ({_src})")

if not positions:
    st.info("No open positions in this account.")
    st.stop()

# A pending sell confirm for a symbol no longer held (sold elsewhere, refreshed away) is stale.
_sp = st.session_state.get("sell_pending")
if _sp and _sp.get("symbol") not in {p["symbol"] for p in positions}:
    st.session_state.pop("sell_pending", None)

# --- Positions table ------------------------------------------------------------------------ #
import pandas as pd  # noqa: E402

rows = [{
    "symbol": p["symbol"], "qty": p["qty"], "avg_entry": p["avg_entry"],
    "current_price": p["current_price"],
    "gain_pct": (p["gain_pct"] * 100.0 if p["gain_pct"] is not None else None),
    "stage": p.get("stage") or "",
    "market_value": p["market_value"], "unrealized_pl": p["unrealized_pl"],
    "current_stop": p["current_stop"], "sma_50": p["sma_50"],
    "earnings": _earnings_cell(p),
    "advisories": " · ".join(p["advisories"]) if p["advisories"] else "",
} for p in positions]
_num = st.column_config.NumberColumn
col_config = {
    "symbol": st.column_config.Column("Ticker"),
    "qty": _num("Shares", format="%d"),
    "avg_entry": _num("Avg entry", format="$%.2f"),
    "current_price": _num("Price", format="$%.2f"),
    "gain_pct": _num("Gain", format="%.1f%%", help="Unrealized gain/loss on the position."),
    "stage": st.column_config.Column(
        "Stage", help="The stop ladder by gain: underwater · fresh (<16%) · working "
                      "(16-20%, stop → breakeven) · well in profit (≥20%, trail 50-day)."),
    "market_value": _num("Mkt value", format="$%.0f"),
    "unrealized_pl": _num("Unreal. P&L", format="$%.0f"),
    "current_stop": _num("Stop", format="$%.2f",
                         help="Highest open protective sell-stop in force (blank = no stop)."),
    "sma_50": _num("50-day SMA", format="$%.2f",
                   help="Minervini trails the 50-day once well in profit; a close below it on "
                        "heavy volume is an exit signal."),
    "earnings": st.column_config.Column(
        "Earnings", help="Next scheduled report. ⚠︎ inside ~21 days: a stop can't protect "
                         "against an earnings gap — hold through only with a profit cushion "
                         "(~8%+), otherwise trim or exit before the report."),
    "advisories": st.column_config.Column("Advisories", width="large"),
}
col_order = ["symbol", "qty", "avg_entry", "current_price", "gain_pct", "stage",
             "market_value", "unrealized_pl", "current_stop", "sma_50", "earnings",
             "advisories"]
st.dataframe(pd.DataFrame(rows), column_config=col_config, column_order=col_order,
             hide_index=True, width="stretch")

# --- Stop management: basis + per-row editable stops + re-arm -------------------------------- #
st.markdown("#### 🛡️ Stops & sells")
st.caption("The GTC ratchet only ever RAISES a stop — a suggestion below the stop already in "
           "force is ignored. Edit any row before re-arming. Each row's expander sells "
           "part/all at market (two-step confirm); a partial sell re-places the stop for the "
           "remaining shares at the same level.")
basis = st.radio("Stop basis", trade.STOP_BASES, horizontal=True, key="pos_basis",
                 format_func=lambda b: _BASIS_LABELS.get(b, b),
                 help="How each row's suggested new stop is chosen. 'Auto' picks per position by "
                      "its gain: fresh → initial 8% below entry, working → breakeven, well in "
                      "profit → trail the 50-day SMA.")
_nonce = st.session_state.pos_nonce
for p in positions:
    sym, price = p["symbol"], p["current_price"]
    suggested, eff = trade.suggest_stop(
        avg_entry=p["avg_entry"], current_price=price, sma_50=p["sma_50"],
        current_stop=p["current_stop"], gain_pct=p["gain_pct"], basis=basis)
    seed = suggested if suggested is not None else (p["current_stop"] or 0.0)
    cA, cB = st.columns([3, 2])
    _g = f"{p['gain_pct'] * 100:+.1f}%" if p["gain_pct"] is not None else "n/a"
    _eff = f" · {_BASIS_LABELS.get(eff, eff).split(' (')[0].lower()}" if basis == "auto" else ""
    cA.caption(f"• **{sym}** {p['qty']} sh · {_g}{_eff}"
               + (f" · {' · '.join(p['advisories'])}" if p["advisories"] else ""))
    cB.number_input(f"stop {sym}", min_value=0.0, value=float(seed), step=0.01, format="%.2f",
                    key=f"posstop_{sym}_{_nonce}_{basis}", label_visibility="collapsed")
    _ed = st.session_state.get(f"posstop_{sym}_{_nonce}_{basis}", seed)
    if not trade.stop_is_valid(_ed, price):
        cB.caption(":red[stop must be < price — set manually]")
    elif price:
        cA.caption(f"  ↳ risk to stop ≈ {(price - _ed) / price * 100:.1f}%")

    # --- Manual sell (market, paper) — two-step confirm; the app never sells on its own --- #
    _held = int(p["qty"] or 0)
    if _held >= 1:
        with st.expander(f"Sell {sym} (market, paper)"):
            sc = st.columns([2, 1, 1, 1])
            # Seed the default via session_state (not value=) — the presets also write this
            # key, and a widget with BOTH a default and a state-set value logs a warning.
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
            if st.button(f"Sell (market) — {sym}", key=f"sell_{sym}_{_nonce}",
                         width="stretch"):
                # FREEZE the quantity now — the confirm must submit what was shown, not a
                # qty edited while the banner is open.
                st.session_state["sell_pending"] = {
                    "symbol": sym, "qty": int(st.session_state.get(
                        f"sellqty_{sym}_{_nonce}", _held)),
                    "held": _held, "stop": p["current_stop"]}
            _sp = st.session_state.get("sell_pending")
            if _sp and _sp.get("symbol") == sym:
                _q, _stop = _sp["qty"], _sp.get("stop")
                _rem = _sp["held"] - _q
                if _stop is None:
                    _sfx = (f"no stop is armed; the remaining {_rem} sh stay unprotected"
                            if _rem > 0 else "no stop is armed")
                elif _rem > 0:
                    _sfx = (f"stop @ {_stop:.2f} is cancelled, then re-placed for the "
                            f"remaining {_rem} sh at the same level")
                else:
                    _sfx = f"stop @ {_stop:.2f} is cancelled (no shares remain)"
                st.warning(f"Sell **{_q}/{_sp['held']} sh {sym}** at market (DAY)? {_sfx}.")
                cc1, cc2 = st.columns(2)
                cc1.button("✅ Confirm sell", key=f"sellgo_{sym}_{_nonce}", type="primary",
                           width="stretch", on_click=_do_sell, args=(sym, _q))
                cc2.button("Cancel", key=f"sellno_{sym}_{_nonce}", width="stretch",
                           on_click=_cancel_sell)

st.button("🛡️ Re-arm / raise all stops", type="primary", width="stretch",
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
        st.success(f"Actioned {len(_act)}/{len(_rr['results'])} stop(s) on "
                   f"account …{str(_rr['account_number'])[-4:]} · equity ${_rr['equity']:,.0f}")
        for r in _rr["results"]:
            ic = _ICONS.get(r["status"], "•")
            st.caption(f"{ic} {r['ticker']}: {r['status']} — {r.get('detail', '')}")

st.caption("Educational tool — not financial advice. Stops are GTC on the paper account.")
