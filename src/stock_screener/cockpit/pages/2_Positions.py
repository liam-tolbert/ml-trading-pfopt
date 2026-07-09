"""Positions page — stop management for the Minervini Trader Alpaca paper account.

A dedicated page (opened from the sidebar / page nav), deliberately SEPARATE from the scan page
so it loads instantly without running the multi-minute scan — this is the daily stop-management
surface. Shows each holding's P&L and protective-stop status, Minervini exit advisories, and a
one-click "re-arm / raise all stops" that reuses the GTC one-way ratchet (never lowers a stop).

Run the app from the project root: ``streamlit run src/stock_screener/cockpit/app.py`` and pick
"Positions" from the page nav (or the sidebar link).
"""
from __future__ import annotations

import sys
from pathlib import Path

# This page imports cockpit modules, so it needs the repo ROOT on sys.path (the guide page
# doesn't import anything and so skips this). From pages/: pages=0, cockpit=1, stock_screener=2,
# src=3, root=4 — one level deeper than app.py's parents[3].
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
    # Reference the MODULE attribute (trade.fetch_positions) so a test patch is honored. The
    # nonce lets Refresh / a re-arm bust the cache. TradeUnavailable is NOT cached (st.cache_data
    # doesn't cache exceptions), so a credentials fix + Refresh recovers.
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

# --- Positions table ------------------------------------------------------------------------ #
import pandas as pd  # noqa: E402

rows = [{
    "symbol": p["symbol"], "qty": p["qty"], "avg_entry": p["avg_entry"],
    "current_price": p["current_price"],
    "gain_pct": (p["gain_pct"] * 100.0 if p["gain_pct"] is not None else None),
    "market_value": p["market_value"], "unrealized_pl": p["unrealized_pl"],
    "current_stop": p["current_stop"], "sma_50": p["sma_50"],
    "advisories": " · ".join(p["advisories"]) if p["advisories"] else "",
} for p in positions]
_num = st.column_config.NumberColumn
col_config = {
    "symbol": st.column_config.Column("Ticker"),
    "qty": _num("Shares", format="%d"),
    "avg_entry": _num("Avg entry", format="$%.2f"),
    "current_price": _num("Price", format="$%.2f"),
    "gain_pct": _num("Gain", format="%.1f%%", help="Unrealized gain/loss on the position."),
    "market_value": _num("Mkt value", format="$%.0f"),
    "unrealized_pl": _num("Unreal. P&L", format="$%.0f"),
    "current_stop": _num("Stop", format="$%.2f",
                         help="Highest open protective sell-stop in force (blank = no stop)."),
    "sma_50": _num("50-day SMA", format="$%.2f",
                   help="Minervini trails the 50-day once well in profit; a close below it on "
                        "heavy volume is an exit signal."),
    "advisories": st.column_config.Column("Advisories", width="large"),
}
col_order = ["symbol", "qty", "avg_entry", "current_price", "gain_pct", "market_value",
             "unrealized_pl", "current_stop", "sma_50", "advisories"]
st.dataframe(pd.DataFrame(rows), column_config=col_config, column_order=col_order,
             hide_index=True, width="stretch")

# --- Stop management: basis + per-row editable stops + re-arm -------------------------------- #
st.markdown("#### 🛡️ Raise / arm protective stops")
st.caption("The GTC ratchet only ever RAISES a stop — a suggestion below the stop already in "
           "force is ignored. Edit any row before re-arming.")
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

st.button("🛡️ Re-arm / raise all stops", type="primary", width="stretch",
          on_click=_do_rearm, args=(positions, _nonce, basis))

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
