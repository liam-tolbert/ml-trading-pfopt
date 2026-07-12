"""Trade journal page — "know your numbers" for the Minervini Trader Alpaca paper account.

A dedicated page (sidebar nav, right below Positions), SEPARATE from the scan page so it
loads instantly. Reconstructs closed round trips from the account's order history — every
cockpit order is tagged via client_order_id (SEPAoto-/SEPAstop-/SEPAcockpit-), so no
separate bookkeeping exists or is needed — and shows the numbers Minervini says drive
progressive exposure (*Think & Trade Like a Champion*): batting average, average win vs
average loss, and per-trade expectancy.

Run the app from the project root: ``streamlit run src/stock_screener/cockpit/app.py`` and
pick "Journal" from the page nav.
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

st.set_page_config(page_title="Journal", page_icon="🧾", layout="wide")
# Mirror app.py's padding trim so the first row isn't buried under Streamlit's header.
st.markdown(
    "<style>.block-container{padding-top:4rem;padding-bottom:2rem;}"
    'div[data-testid="stVerticalBlock"]{gap:0.6rem;}</style>',
    unsafe_allow_html=True)


@st.cache_data(show_spinner="Reading the order history…")
def _cached_fills(nonce):
    # Reference the MODULE attribute (trade.fetch_order_fills) so a test patch is honored. The
    # nonce lets Refresh bust the cache. TradeUnavailable is NOT cached (st.cache_data doesn't
    # cache exceptions), so a credentials fix + Refresh recovers.
    return trade.fetch_order_fills()


def _pct(v, signed=False, digits=1):
    """Format a fraction as a percent string, or an em-dash when the stat has no data yet."""
    if v is None:
        return "—"
    return f"{v * 100:+.{digits}f}%" if signed else f"{v * 100:.{digits}f}%"


# --------------------------------------------------------------------------- #
st.title("🧾 Journal — know your numbers")
st.caption("Closed round trips reconstructed from the Minervini Trader **paper** account's "
           "order history (cockpit orders are tagged by client_order_id — no separate "
           "bookkeeping). Educational only — not financial advice.")

if "jr_nonce" not in st.session_state:
    st.session_state.jr_nonce = 1
if st.button("🔄 Refresh"):
    st.session_state.jr_nonce += 1
    _cached_fills.clear()

try:
    data = _cached_fills(st.session_state.jr_nonce)
except trade.TradeUnavailable as e:
    st.warning(str(e))
    st.stop()

acct = data["account"]
fills = data["fills"]

_src = ("Minervini Trader keys" if acct.get("using_dedicated")
        else "shared ALPACA_* keys — set ALPACA_API_KEY_MINERVINI / "
             "ALPACA_API_KEY_SECRET_MINERVINI to target the Minervini account")
st.caption(f"Account **…{str(acct['account_number'])[-4:]}** ({_src}) · "
           f"equity ${acct['equity']:,.0f}")

if not fills:
    st.info("No filled orders in this account yet — the journal starts with your first trade.")
    st.stop()

journal = trade.build_trade_journal(fills)
only_tagged = st.checkbox(
    "Cockpit trades only (SEPA-tagged orders)", value=True,
    help="On: only round trips containing a cockpit order (client_order_id SEPAoto-/SEPAstop-/"
         "SEPAcockpit-). Off: every trade in the account's history, including manual ones — "
         "and the All-Weather mirror's if this page is running on the shared keys.")
closed = [t for t in journal["closed"] if t["tagged"] or not only_tagged]
open_eps = [t for t in journal["open"] if t["tagged"] or not only_tagged]
stats = trade.journal_stats(closed)

# --- Headline stats (one '$' per metric value — two render as a LaTeX math span) ------------ #
m = st.columns(5)
m[0].metric("Closed trades", str(stats["n"]), border=True)
m[1].metric("Batting avg", _pct(stats["batting_avg"], digits=0), border=True,
            help="Winners as a share of all closed trades. Minervini: ~50% is plenty when "
                 "the win/loss ratio is ≥ 2:1.")
m[2].metric("Avg win / loss", f"{_pct(stats['avg_win_pct'], signed=True)} / "
                              f"{_pct(stats['avg_loss_pct'], signed=True)}", border=True,
            help="Average gain of winners vs average loss of losers (% of cost).")
m[3].metric("Expectancy", _pct(stats["expectancy_pct"], signed=True, digits=2), border=True,
            help="Mean P&L% per closed trade = batting × avg win + (1 − batting) × avg loss. "
                 "This is the edge that earns progressive exposure.")
m[4].metric("Realized P&L", f"${stats['total_pl']:,.0f}", border=True)

_ratio = stats["win_loss_ratio"]
_hw, _hl = stats["avg_hold_days_win"], stats["avg_hold_days_loss"]
st.caption(f"{stats['wins']}W · {stats['losses']}L · {stats['scratches']} scratch — "
           f"win/loss ratio **{f'{_ratio:.2f}' if _ratio is not None else '—'}** · avg hold "
           f"**{f'{_hw:.0f}d' if _hw is not None else '—'}** wins / "
           f"**{f'{_hl:.0f}d' if _hl is not None else '—'}** losses. Positive expectancy → "
           "press exposure progressively; negative → cut size until the numbers turn "
           "(*Think & Trade Like a Champion*).")

# --- Closed trades table --------------------------------------------------------------------- #
import pandas as pd  # noqa: E402

if not closed:
    st.info("No closed round trips yet"
            + (f" — {len(open_eps)} trade(s) still open." if open_eps else "."))
else:
    rows = [{
        "symbol": t["symbol"],
        "entered": t["entry_date"].date(), "exited": t["exit_date"].date(),
        "days": t["hold_days"], "shares": t["shares"],
        "avg_entry": t["avg_entry"], "avg_exit": t["avg_exit"],
        "pl": t["pl"], "pl_pct": t["pl_pct"] * 100.0,
        "source": "cockpit" if t["tagged"] else "other",
    } for t in sorted(closed, key=lambda t: t["exit_date"], reverse=True)]
    _num = st.column_config.NumberColumn
    col_config = {
        "symbol": st.column_config.Column("Ticker"),
        "entered": st.column_config.Column("Entered"),
        "exited": st.column_config.Column("Exited"),
        "days": _num("Days", format="%d"),
        "shares": _num("Shares", format="%g"),
        "avg_entry": _num("Avg entry", format="$%.2f"),
        "avg_exit": _num("Avg exit", format="$%.2f"),
        "pl": _num("P&L", format="$%.0f"),
        "pl_pct": _num("P&L %", format="%.1f%%"),
        "source": st.column_config.Column(
            "Source", help="cockpit = the round trip contains a SEPA-tagged order."),
    }
    df = pd.DataFrame(rows)
    st.dataframe(df, column_config=col_config, hide_index=True, width="stretch",
                 column_order=list(col_config))
    st.download_button("⬇️ Journal CSV", df.to_csv(index=False).encode(),
                       file_name="trade_journal.csv", mime="text/csv")

# --- Open trades (not in the stats) ----------------------------------------------------------- #
if open_eps:
    st.markdown("#### Open trades — excluded from the stats until closed")
    orows = [{
        "symbol": t["symbol"], "entered": t["entry_date"].date(),
        "shares_open": t["shares_open"], "avg_entry": t["avg_entry"],
        "realized_pl": t["realized_pl"],
        "source": "cockpit" if t["tagged"] else "other",
    } for t in sorted(open_eps, key=lambda t: t["entry_date"], reverse=True)]
    _num = st.column_config.NumberColumn
    st.dataframe(pd.DataFrame(orows), hide_index=True, width="stretch", column_config={
        "symbol": st.column_config.Column("Ticker"),
        "entered": st.column_config.Column("Entered"),
        "shares_open": _num("Shares open", format="%g"),
        "avg_entry": _num("Avg entry", format="$%.2f"),
        "realized_pl": _num("Realized so far", format="$%.0f",
                            help="Partial sells booked at the episode's average cost."),
        "source": st.column_config.Column("Source"),
    })

if journal["unmatched_sells"]:
    st.caption(f"⚠ {len(journal['unmatched_sells'])} sell fill(s) had no matching prior buy "
               "in the pulled history (pre-history or transferred shares) — excluded from "
               "every number above.")

st.caption("Educational tool — not financial advice. Stats count position episodes "
           "(flat → long → flat), so scale-ins and partial exits aggregate into one trade.")
