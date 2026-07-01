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

# --------------------------------------------------------------------------- #
# Contextual help — shown via ℹ️ popovers next to each step. The full method
# reference lives on the "SEPA Guide" page (pages/1_SEPA_Guide.py).
# --------------------------------------------------------------------------- #
INFO_REGIME = """
**Market environment — check this first.** SEPA is market-aware: most breakouts
fail in a weak tape. **Phase-2 breadth** = % of scanned names in confirmed uptrends.
- **BUY OK / Risk-On** → trade actively.
- **CAUTION / weak breadth / Risk-Off** → preserve capital, wait.

Don't force trades when few names qualify — the market is telling you something.
"""

INFO_STEP1 = """
**Step 1 — Trend Template (automated gate).** Every row passes **all 8** of Minervini's
trend-template criteria: price above stacked **50 > 150 > 200-day SMAs**, 200-day rising
≥1 month, **≥30% above the 52-wk low**, **within 25% of the 52-wk high**, confirmed Stage 2.
This is *eligibility, not a buy signal.*

- **RS** = relative-strength rating (percentile of 6-mo return vs the scanned set);
  Minervini wants **70+**.
- Tighten further in the sidebar: raise **min RS** or require a VCP.
- Sort by `fund_score` / `rs`, then **click a row** to study the chart.
"""

INFO_STEP2 = """
**Step 2 — Fundamentals (the fuel).** A great chart with weak earnings is a trap.
Look for:
- **EPS & revenue YoY ≥ ~20%** and **accelerating** (this quarter ≥ last),
- **stable or expanding margins** (positive *margin trend*).

`fund_score` (0–4) counts how many checks pass. yfinance often exposes only ~4
quarters, so **YoY may read n/a** — QoQ is the fallback. Use this to rank the
Step-1 list, not as a hard cutoff unless you set "min fundamental checks".
"""

INFO_STEP3 = """
**Step 3 — Read the VCP yourself (this is the discretionary part).** On the chart,
look for a **Volatility Contraction Pattern**:
- 2–6 pullbacks, each **tighter** than the last (e.g. 18% → 12% → 6%),
- **higher lows**, **volume drying up** into the tightest part,
- price holding above the **50-day SMA**, total base depth ~10–35%.

Shaded bands mark *detected* contractions (a hint — you decide). Use the **Time range**
buttons to zoom into the base (a tight VCP is invisible over 2 years), **Weekly** for
base structure, **Daily** for the exact pivot. **No clean VCP → no trade.**
"""

INFO_STEP4 = """
**Step 4 — Entry (you pull the trigger).** Buy only when price **closes above the
pivot** on volume **≥ 40–50% above average** (the "breakout today" flag).
- **Don't chase** more than ~5% above the pivot (the buy zone).
- Set the **7–8% stop immediately** and never lower it.
- First target ~**20–25%**; trail with the 50-day SMA once well in profit.
- **Size** so a stop-out costs ~1% of the account (set Account $ / Risk %).

These levels are advisory — place the order in your broker.
"""


def info_btn(body: str, label: str = "ℹ️ How to use") -> None:
    """A small clickable info popover (falls back to an expander on older Streamlit)."""
    try:
        with st.popover(label):
            st.markdown(body)
    except Exception:
        with st.expander(label):
            st.markdown(body)


def filter_table(df, key_prefix: str = "flt"):
    """Interactive value filter — pick one or more columns and narrow by their values.

    Renders a control matched to each column's type: a range slider for numbers
    (n/a rows drop out only once you narrow the range), a True/False/All picker for
    booleans, and a multi-select of distinct values for text. Filters combine (AND).
    """
    import pandas as pd

    out = df
    with st.expander("🔎 Filter by column values", expanded=False):
        columns_to_filter = df.loc[:, ~df.columns.str.contains('ticker')].columns
        cols = st.multiselect(
            "Columns to filter on", list(columns_to_filter), key=f"{key_prefix}_cols",
            help="Pick one or more columns; a matching control appears for each.")
        for col in cols:
            s = df[col]
            lc, rc = st.columns([0.28, 0.72])
            lc.markdown(f"**{col}**")
            with rc:
                if pd.api.types.is_bool_dtype(s):
                    choice = st.selectbox(
                        "value", ["All", "True", "False"], key=f"{key_prefix}_{col}",
                        label_visibility="collapsed")
                    if choice != "All":
                        out = out[out[col] == (choice == "True")]
                elif pd.api.types.is_numeric_dtype(s):
                    nn = s.dropna()
                    if nn.empty:
                        st.caption("no numeric values")
                        continue
                    cmin, cmax = float(nn.min()), float(nn.max())
                    if cmin == cmax:
                        st.caption(f"all rows = {cmin:g}")
                        continue
                    is_int = pd.api.types.is_integer_dtype(s) or bool((nn % 1 == 0).all())
                    step = 1.0 if is_int else max((cmax - cmin) / 100.0, 0.01)
                    lo, hi = st.slider(
                        "range", cmin, cmax, (cmin, cmax), step=step,
                        key=f"{key_prefix}_{col}", label_visibility="collapsed")
                    if (lo, hi) != (cmin, cmax):        # only filter once narrowed
                        out = out[out[col].between(lo, hi)]
                else:                                    # text / categorical
                    opts = sorted(s.dropna().astype(str).unique().tolist())
                    chosen = st.multiselect(
                        "values", opts, key=f"{key_prefix}_{col}",
                        label_visibility="collapsed")
                    if chosen:
                        out = out[out[col].astype(str).isin(chosen)]
    return out


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
try:                                                  # full method reference page
    st.sidebar.page_link("pages/1_SEPA_Guide.py", label="📖 Full SEPA method guide")
except Exception:
    st.sidebar.caption("📖 Open the **SEPA Guide** page from the menu (top-left).")
with st.sidebar.popover("ℹ️ How to use this tool"):
    st.markdown(
        "1. **Check the market environment** banner — only push in a healthy tape.\n"
        "2. **Step 1:** the table lists trend-template passers (eligibility).\n"
        "3. **Step 2:** rank by fundamental quality (the 'fuel').\n"
        "4. **Step 3:** click a row and *judge the VCP yourself* on the chart.\n"
        "5. **Step 4:** if it breaks out on volume, use the advisory entry/stop/size.\n\n"
        "Each section has its own **ℹ️** button. Full details on the **SEPA Guide** page.")
universe = st.sidebar.selectbox("Universe", ["sp500", "tickers"], index=0,
                                help="sp500 = S&P 500 constituents; tickers = data/tickers.txt")
min_criteria = 8  # full Minervini trend template — all 8 criteria required (no 7/8)
st.sidebar.caption("Gate: full **8/8** trend template")
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
hcol, icol = st.columns([0.8, 0.2])
hcol.markdown("#### Market environment")
with icol:
    info_btn(INFO_REGIME)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Market regime", str(reg.get("regime")),
          help="Risk-On/Off from SPY phase + breadth. Trade actively only when Risk-On.")
c2.metric("SPY trend", str(reg.get("spy_trend")),
          help="SPY's own Stage 1-4 phase. Stage 2 = bullish backdrop for breakouts.")
c3.metric("Phase-2 breadth", f"{reg.get('phase2_pct', 0):.0f}%",
          help=f"% of scanned names in confirmed uptrends (quality: "
               f"{reg.get('breadth_quality')}). Higher = healthier tape.")
c4.metric("Environment", "BUY OK" if buy_ok else "CAUTION", delta=None, delta_color="off",
          help="Whether SEPA says it's worth taking new breakouts right now.")
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

h1, i1 = st.columns([0.8, 0.2])
h1.subheader(f"Step 1 — Candidates ({len(cand)})")
with i1:
    info_btn(INFO_STEP1)
query = st.text_input("🔎 Search by ticker", "", placeholder="e.g. NVDA").strip().upper()
if query:
    view = cand[cand["ticker"].str.upper().str.contains(query, na=False, regex=False)]
else:
    view = cand

view = filter_table(view)

if len(view) == 0:
    st.info("No candidates match the current filters. Clear a filter to see more.")
    st.stop()

st.caption(f"Showing {len(view)} of {len(cand)}. Click a row to chart it.")
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
    st.caption("Step 3 — judge the VCP")
    info_btn(INFO_STEP3, label="ℹ️ How to read the chart")
    weekly = st.checkbox("Weekly view", value=False)
    show_overlays = st.checkbox("VCP + entry overlays", value=True)

payload = res.payloads[pick]
with colA:
    _ranges = {"3M": 90, "6M": 180, "9M": 270, "1Y": 365, "2Y / All": None}
    rsel = st.radio("Time range", list(_ranges), index=1, horizontal=True,
                    help="Zoom in to see the VCP base; a tight base is hard to read over 2 years.")
    fig = build_chart(pick, payload["df"], vcp=payload.get("vcp"),
                      levels=payload.get("levels"), show_overlays=show_overlays,
                      weekly=weekly, lookback_days=_ranges[rsel])
    st.plotly_chart(fig, width="stretch")

# Step 2 + Step 4 panels
p2, p4 = st.columns(2)
with p2:
    st.markdown("**Step 2 — Fundamentals**")
    info_btn(INFO_STEP2)
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
    info_btn(INFO_STEP4)
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
