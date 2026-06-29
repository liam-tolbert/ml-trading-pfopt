"""SEPA Cockpit — Streamlit UI.

Run from the project root:

    streamlit run src/stock_screener/cockpit/app.py

Sidebar drives the scan; the main pane is the candidate table and, for the selected
name, the chart (you judge the VCP) plus Step-2 fundamentals and Step-4 advisory
entry levels. You are the judge — the tool only does the mechanical filtering.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:                       # so `from src.X import ...` resolves
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.stock_screener.cockpit.charts import build_chart  # noqa: E402
from src.stock_screener.cockpit.scan import ScanConfig, run_scan  # noqa: E402

st.set_page_config(page_title="SEPA Cockpit", layout="wide")


@st.cache_data(show_spinner=False)
def _cached_scan(universe, min_criteria, min_rs, require_vcp, min_fund, nonce):
    cfg = ScanConfig(min_criteria=min_criteria, min_rs=float(min_rs),
                     require_vcp=require_vcp, min_fundamental_score=min_fund)
    return run_scan(universe=universe, cfg=cfg, force=bool(nonce and nonce % 2 == 0))


# --------------------------------------------------------------------------- #
# Sidebar — scan settings
# --------------------------------------------------------------------------- #
st.sidebar.title("SEPA Cockpit")
st.sidebar.caption("Mechanical Steps 1-2; you judge Steps 3-4.")
universe = st.sidebar.selectbox("Universe", ["sp500", "tickers"], index=0,
                                help="sp500 = S&P 500 constituents; tickers = data/tickers.txt")
min_criteria = st.sidebar.radio("Trend template gate", [7, 8], index=0,
                                format_func=lambda x: f"{x}/8 criteria")
min_rs = st.sidebar.slider("Min RS rating", 0, 99, 70,
                           help="IBD-style percentile of trailing 6-mo return vs the scanned universe")
require_vcp = st.sidebar.checkbox("VCP only (hint filter)", value=False)
min_fund = st.sidebar.slider("Min fundamental checks (0-4)", 0, 4, 0)

if "nonce" not in st.session_state:
    st.session_state.nonce = 1
if st.sidebar.button("🔄 Re-scan (refresh prices)"):
    st.session_state.nonce += 1
    _cached_scan.clear()

with st.spinner("Scanning… (first run pulls prices; later runs use the cache)"):
    res = _cached_scan(universe, min_criteria, min_rs, require_vcp, min_fund,
                       st.session_state.nonce)

# --------------------------------------------------------------------------- #
# Regime banner
# --------------------------------------------------------------------------- #
reg = res.regime
buy_ok = reg.get("should_generate_buys")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Market regime", str(reg.get("regime")))
c2.metric("SPY trend", str(reg.get("spy_trend")))
c3.metric("Phase-2 breadth", f"{reg.get('phase2_pct', 0):.0f}%",
          help=f"breadth quality: {reg.get('breadth_quality')}")
c4.metric("Environment", "BUY OK" if buy_ok else "CAUTION",
          delta=None, delta_color="off")
if not buy_ok:
    st.warning("SEPA discipline: the tape is weak — most breakouts fail here. "
               + "; ".join(reg.get("reasons", [])))

st.caption(f"Scanned {res.n_scanned} names · {res.n_passed} pass the "
           f"{min_criteria}/8 trend template"
           + (f" · {len(res.errors)} errors" if res.errors else ""))

# --------------------------------------------------------------------------- #
# Candidate table + per-name detail
# --------------------------------------------------------------------------- #
cand = res.candidates
if cand is None or len(cand) == 0:
    st.info("No candidates passed the current filters. "
            "Loosen the RS / fundamental filters, or wait for a better tape.")
    st.stop()

st.subheader(f"Candidates ({len(cand)})")
query = st.text_input("🔎 Filter by ticker", "", placeholder="e.g. NVDA").strip().upper()
if query:
    view = cand[cand["ticker"].str.upper().str.contains(query, na=False, regex=False)]
else:
    view = cand

if len(view) == 0:
    st.info(f"No candidates match '{query}'.")
    st.stop()

st.caption("Click a row to chart it.")
event = st.dataframe(view, width="stretch", hide_index=True,
                     on_select="rerun", selection_mode="single-row", key="cand_table")

# selection.rows are positional indices into the displayed (filtered) frame
_sel = getattr(event, "selection", None)
_rows = (_sel.get("rows", []) if isinstance(_sel, dict)
         else getattr(_sel, "rows", [])) if _sel else []
row_pos = _rows[0] if _rows and _rows[0] < len(view) else 0
pick = view.iloc[row_pos]["ticker"]

colA, colB = st.columns([3, 1])
with colB:
    st.markdown(f"### {pick}")
    weekly = st.checkbox("Weekly view", value=False)
    show_overlays = st.checkbox("VCP + entry overlays", value=True)

payload = res.payloads[pick]
with colA:
    fig = build_chart(pick, payload["df"], vcp=payload.get("vcp"),
                      levels=payload.get("levels"), show_overlays=show_overlays,
                      weekly=weekly)
    st.plotly_chart(fig, width="stretch")

# Step 2 + Step 4 panels
p2, p4 = st.columns(2)
with p2:
    st.markdown("**Step 2 — Fundamentals**")
    f = payload.get("fundamentals")
    s2 = payload.get("step2", {})
    if not f:
        st.caption("No fundamental data available (yfinance).")
    else:
        for label, key in [("Revenue YoY", "revenue_yoy"), ("Revenue QoQ", "revenue_qoq"),
                           ("EPS YoY", "eps_yoy"), ("EPS QoQ", "eps_qoq"),
                           ("Operating margin", "operating_margin"),
                           ("Margin trend (pp)", "margin_trend"),
                           ("Inventory QoQ", "inventory_qoq")]:
            v = f.get(key)
            st.write(f"- {label}: " + ("n/a" if v is None else f"{v:+.1f}%"))
        st.write("Checks passed: "
                 + ", ".join(k for k, ok in s2.get("checks", {}).items() if ok) or "—")

with p4:
    lv = payload.get("levels", {})
    st.markdown("**Step 4 — Entry (advisory)**")
    bz = lv.get("buy_zone", (None, None))
    st.write(f"- Pivot: ${lv.get('pivot', 0):.2f}")
    st.write(f"- Buy zone (no chase): ${bz[0]:.2f} – ${bz[1]:.2f}")
    st.write(f"- Stop (~7-8%): ${lv.get('stop', 0):.2f}")
    st.write(f"- Target (~20-25%): ${lv.get('target', 0):.2f}")
    pct = lv.get("pct_to_pivot")
    st.write(f"- Distance to pivot: " + ("n/a" if pct is None else f"{pct:+.1f}%"))
    st.write(f"- Breakout today: {'✅' if lv.get('breakout_today') else '—'} "
             f"(vol {lv.get('volume_ratio', 1):.1f}× avg)")

    st.markdown("---")
    acct = st.number_input("Account $", min_value=0.0, value=100_000.0, step=1000.0)
    risk_pct = st.number_input("Risk % per trade", min_value=0.0, value=1.0, step=0.25)
    entry, stop = bz[0], lv.get("stop")
    if entry and stop and entry > stop:
        shares = (acct * risk_pct / 100.0) / (entry - stop)
        st.write(f"**Size:** {shares:,.0f} sh (~${shares * entry:,.0f}) "
                 f"at the pivot, risking {risk_pct:.2f}% to stop.")

st.caption("Educational tool — not financial advice. You place orders yourself.")
