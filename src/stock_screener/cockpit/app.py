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

# Reclaim vertical space so the candidate table is visible on load: trim Streamlit's
# large default top padding and tighten the gap between stacked elements. Conservative,
# stable selectors — if a future version ignores one, the page still renders fine.
st.markdown(
    "<style>"
    # padding-top must stay >= Streamlit's fixed header height (~3.75rem) or the top row
    # slides under it; 4rem clears the header while still reclaiming ~2rem vs the default.
    ".block-container{padding-top:4rem;padding-bottom:2rem;}"
    'div[data-testid="stVerticalBlock"]{gap:0.6rem;}'
    "</style>",
    unsafe_allow_html=True,
)

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

Shaded bands mark *detected* contractions (a hint — you decide). The bottom **RMV** pane
(Relative Measured Volatility, 0–100) tracks how tight the base is versus the stock's own
recent range — falling **into the green < 25 band = a genuine volatility contraction**, the
VCP sweet spot; a high RMV means the base is still loose. Use the **Time range** buttons to
zoom into the base (a tight VCP is invisible over 2 years), **Weekly** for base structure,
**Daily** for the exact pivot. **No clean VCP → no trade.**
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


# Raw candidate-frame column name -> human-readable label. Used both to relabel the
# rendered table headers (via st.dataframe column_config) and the filter picker/row
# headers. Keys stay the raw column names so selection & filtering logic is unchanged.
READABLE_COLS = {
    "ticker": "Ticker",
    "price": "Price ($)",
    "rs": "RS rating",
    "criteria": "Trend criteria (/8)",
    "fund_score": "Fundamental score (0-4)",
    "rev_yoy": "Revenue YoY (%)",
    "eps_yoy": "EPS YoY (%)",
    "op_margin": "Operating margin (%)",
    "vcp": "VCP detected",
    "num_contractions": "# Contractions",
    "vcp_quality": "VCP quality (0-100)",
    "breakout_today": "Breakout today",
    "vol_confirmed": "Vol confirmed",
    "pct_to_pivot": "Distance to pivot (%)",
    "pivot": "Pivot ($)",
    "stop": "Stop ($)",
    "target": "Target ($)",
}

# One-line meaning per column — shown as a header hover tooltip and in the guide.
COL_HELP = {
    "ticker": "Stock symbol. Click a row to chart it.",
    "price": "Latest close price.",
    "rs": "Relative-strength rating 1–99: percentile of the trailing ~6-mo return vs the "
          "scanned universe. Minervini wants 70+.",
    "fund_score": "Step-2 fundamental checks passed (0–4): revenue ≥20%, EPS ≥20%, EPS "
                  "accelerating, margins expanding.",
    "rev_yoy": "Revenue growth vs the year-ago quarter. 'n/a' = too few quarters in yfinance "
               "(unknown, not zero).",
    "eps_yoy": "EPS growth vs the year-ago quarter. Want ≥20% and accelerating.",
    "op_margin": "Current operating margin. Look for stable or expanding.",
    "vcp": "A valid Volatility Contraction Pattern was detected (2–6 tightening pullbacks, "
           "quality ≥50).",
    "num_contractions": "Number of peak→trough pullbacks in the current base. Minervini's "
                        "range is 2–6.",
    "vcp_quality": "Base quality 0–100 (tightening 30 + volume-drying 20 + #contractions 20 "
                   "+ near-high 20 + base length 10). Shown even when VCP is False.",
    "breakout_today": "Price is clearing the pivot right now (price only — see 'Vol "
                      "confirmed' for the volume side).",
    "vol_confirmed": "True when the latest-day volume is ≥ 1.5× its 20-day average — the "
                     "breakout confirmation threshold. A price breakout without this is suspect.",
    "pct_to_pivot": "Distance from price to the pivot. Positive = below pivot (needs to rise); "
                    "negative = already above/extended.",
    "pivot": "Buy-trigger line — the breakout/base level (or 52-wk high). Buy a close above it.",
    "stop": "Advisory stop-loss, ~7–8% below the pivot.",
    "target": "First objective, ~+22.5% above the pivot (the 20–25% zone).",
}

# The table's four decision groups, in display order. This drives BOTH the column
# order/visibility in the table and the "Column guide" popover. `criteria` is left out
# on purpose — it's a constant 8 on this table (the 8/8 gate), so it adds no signal
# (it still lives in the underlying scan frame for tests / analysis).
COL_GROUPS = [
    ("Identify", ["ticker", "price"]),
    ("Fuel — catalyst & strength", ["rs", "fund_score", "rev_yoy", "eps_yoy", "op_margin"]),
    ("Base — the VCP setup", ["vcp", "num_contractions", "vcp_quality"]),
    ("Entry — timing & risk", ["breakout_today", "vol_confirmed", "pct_to_pivot",
                               "pivot", "stop", "target"]),
]
DISPLAY_ORDER = [c for _, cols in COL_GROUPS for c in cols]

INFO_COLUMNS = ("**What each table column means** (hover any header for the same tip).\n\n"
                + "\n\n".join(
                    f"**{group}**\n"
                    + "\n".join(f"- **{READABLE_COLS.get(c, c)}** — {COL_HELP[c]}" for c in cols)
                    for group, cols in COL_GROUPS))


def info_btn(body: str, label: str = "ℹ️ How to use") -> None:
    """A small clickable info popover (falls back to an expander on older Streamlit)."""
    try:
        with st.popover(label):
            st.markdown(body)
    except Exception:
        with st.expander(label):
            st.markdown(body)


def _tag(text, color: str = "blue") -> str:
    """A colored-background inline chip via Streamlit markdown; None -> 'n/a'."""
    return f":{color}-background[{'n/a' if text is None else text}]"


def _regime_color(regime) -> str:
    """Risk-On -> green, Risk-Off -> red, anything else/unknown -> orange."""
    r = str(regime or "").lower()
    return "green" if "on" in r else "red" if "off" in r else "orange"


def step_badge(step: str, title: str) -> str:
    """A consistent blue step chip + title, e.g. ':blue-background[Step 3]  Judge the VCP'."""
    return f":blue-background[{step}]  {title}"


def filter_table(df, key_prefix: str = "flt"):
    """Interactive value filter — pick one or more columns and narrow by their values.

    Renders a control matched to each column's type: a range slider for numbers
    (n/a rows drop out only once you narrow the range), a True/False/All picker for
    booleans, and a multi-select of distinct values for text. Filters combine (AND).
    """
    import pandas as pd

    readable = READABLE_COLS  # display labels; keys are the raw column names

    out = df
    with st.expander("🔎 Filter by column values", expanded=False):
        # Same columns (minus ticker) and order as the displayed table.
        columns_to_filter = [c for c in DISPLAY_ORDER if c in df.columns and c != "ticker"]
        cols = st.multiselect(
            "Columns to filter on", columns_to_filter, key=f"{key_prefix}_cols",
            format_func=lambda c: readable.get(c, c),
            help="Pick one or more columns; a matching control appears for each.")
        for col in cols:
            s = df[col]
            lc, rc = st.columns([0.28, 0.72])
            lc.markdown(f"**{readable.get(col, col)}**")
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
# Full method reference page, opened in a NEW TAB. st.page_link navigates in-tab only
# (no target option), so use a plain anchor to the page's URL slug (from the filename
# `1_SEPA_Guide.py` -> `SEPA_Guide`); relative href stays correct behind a proxy.
st.sidebar.markdown(
    '<a href="SEPA_Guide" target="_blank">📖 Full SEPA method guide ↗</a>',
    unsafe_allow_html=True)
with st.sidebar.popover("ℹ️ How to use this tool"):
    st.markdown(
        "1. **Check the market environment** banner — only push in a healthy tape.\n"
        "2. **Step 1:** the table lists trend-template passers (eligibility).\n"
        "3. **Step 2:** rank by fundamental quality (the 'fuel').\n"
        "4. **Step 3:** click a row and *judge the VCP yourself* on the chart.\n"
        "5. **Step 4:** if it breaks out on volume, use the advisory entry/stop/size.\n\n"
        "Each section has its own **ℹ️** button. Full details on the **SEPA Guide** page.")
universe = st.sidebar.selectbox(
    "Universe", ["sp500", "full_us", "tickers"], index=0,
    format_func=lambda k: {
        "sp500": "S&P 500 (fast)",
        "full_us": "All US common stocks (~3–4k · slow first scan)",
        "tickers": "My tickers.txt",
    }[k],
    help="sp500 = S&P 500 constituents (fast). full_us = broad US common-stock universe "
         "from Nasdaq/NYSE listings (~3–4k names; the FIRST cold scan pulls thousands of "
         "price histories and can take several minutes — later scans reuse the cache and "
         "only fetch new days). tickers = data/tickers.txt")
if universe == "full_us":
    st.sidebar.caption("⏳ First full_us scan pulls ~3–4k price histories (several minutes). "
                       "Later scans use the cache and fetch only new days.")
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
# Compact one-line status strip (replaces four tall metric cards) — same underlying
# values, a fraction of the vertical space, pinned at the top of the page.
p2 = reg.get("phase2_pct", 0)
p2 = p2 if isinstance(p2, (int, float)) else 0
env = ":green-background[BUY OK]" if buy_ok else ":orange-background[CAUTION]"
strip = " · ".join([
    "**Market** " + _tag(reg.get("regime"), _regime_color(reg.get("regime"))),
    f"SPY {reg.get('spy_trend') or 'n/a'}",
    f"Breadth {p2:.0f}% ({reg.get('breadth_quality') or '?'})",
    env,
])
scol, icol = st.columns([0.92, 0.08], vertical_alignment="center")
scol.markdown(strip)
with icol:
    info_btn(INFO_REGIME, label="ℹ️")
if not buy_ok:
    st.caption("⚠︎ Weak tape — most breakouts fail here. "
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

# Compact header row: step badge · inline ticker search · info popovers — one line.
hcol, scol, icol = st.columns([0.5, 0.32, 0.18], vertical_alignment="center")
hcol.markdown(step_badge("Step 1", f"Candidates ({len(cand)})"))
query = scol.text_input("ticker", "", label_visibility="collapsed",
                        placeholder="🔎 ticker e.g. NVDA").strip().upper()
with icol:
    info_btn(INFO_STEP1)
    info_btn(INFO_COLUMNS, label="ℹ️ Columns")
if query:
    view = cand[cand["ticker"].str.upper().str.contains(query, na=False, regex=False)]
else:
    view = cand

# Reserve the table's slot ABOVE the filter: a container renders where it is created,
# not where it is written to. So we draw the (filtered) table into `table_box`, while the
# filter UI below runs first — filtered results still apply on the SAME run, no lag.
table_box = st.container()
view = filter_table(view)

if len(view) == 0:
    st.info("No candidates match the current filters. Clear a filter to see more.")
    st.stop()

table_box.caption(f"Showing {len(view)} of {len(cand)} — click a row to chart it.")
# Relabel headers + attach a hover tooltip per column (display only — underlying column
# names stay raw, so the selection + filter logic that references e.g. "ticker" keeps
# working). column_order groups the columns and hides any not listed (e.g. `criteria`).
with table_box:
    col_config = {c: st.column_config.Column(READABLE_COLS.get(c, c), help=COL_HELP.get(c))
                  for c in view.columns}
    event = st.dataframe(view, width="stretch", hide_index=True, height=380,
                         column_config=col_config, column_order=DISPLAY_ORDER,
                         on_select="rerun", selection_mode="single-row", key="cand_table")

# selection.rows are positional indices into the displayed (filtered) frame
_sel = getattr(event, "selection", None)
_rows = (_sel.get("rows", []) if isinstance(_sel, dict)
         else getattr(_sel, "rows", [])) if _sel else []
row_pos = _rows[0] if _rows and _rows[0] < len(view) else 0
pick = view.iloc[row_pos]["ticker"]

payload = res.payloads[pick]

# Step 2 — Fundamentals: sits directly ABOVE the Step 3 chart (read the fuel, then the base).
with st.container(border=True):
    st.markdown(step_badge("Step 2", "Fundamentals — the fuel"))
    info_btn(INFO_STEP2)
    f = payload.get("fundamentals")
    s2 = payload.get("step2", {})
    if not f:
        st.caption("No fundamental data available (yfinance).")
    else:
        fields = [("Revenue YoY", "revenue_yoy"), ("Revenue QoQ", "revenue_qoq"),
                  ("EPS YoY", "eps_yoy"), ("EPS QoQ", "eps_qoq"),
                  ("Operating margin", "operating_margin"),
                  ("Margin trend (pp)", "margin_trend"),
                  ("Inventory QoQ", "inventory_qoq")]
        lines = [f"**{label}:** "
                 + ("n/a" if f.get(key) is None else f"{f.get(key):+.1f}%")
                 for label, key in fields]
        g1, g2 = st.columns(2)
        half = (len(lines) + 1) // 2
        g1.markdown("\n\n".join(lines[:half]))
        g2.markdown("\n\n".join(lines[half:]))
        checks = s2.get("checks", {})
        st.markdown(" ".join(
            f"{'✅' if checks.get(k) else '—'} {lbl}"
            for k, lbl in [("revenue_growth", "Rev ≥20%"), ("eps_growth", "EPS ≥20%"),
                           ("eps_accelerating", "EPS accel"),
                           ("margin_expanding", "Margin ↑")]))
        st.caption(f"Score {s2.get('score', 0)}/4")

# Step 3 — the chart (judge the VCP) + its controls.
colA, colB = st.columns([3, 1])
with colB:
    with st.container(border=True):
        st.markdown(f"### {pick}")
        st.markdown(step_badge("Step 3", "Judge the VCP"))
        info_btn(INFO_STEP3, label="ℹ️ How to read the chart")
        weekly = st.checkbox("Weekly view", value=False)
        show_overlays = st.checkbox("VCP + entry overlays", value=True)
with colA:
    with st.container(border=True):
        _ranges = {"3M": 90, "6M": 180, "9M": 270, "1Y": 365, "2Y / All": None}
        rsel = st.radio("Time range", list(_ranges), index=1, horizontal=True,
                        help="Zoom in to see the VCP base; a tight base is hard to read over 2 years.")
        fig = build_chart(pick, payload["df"], vcp=payload.get("vcp"),
                          levels=payload.get("levels"), show_overlays=show_overlays,
                          weekly=weekly, lookback_days=_ranges[rsel])
        st.plotly_chart(fig, width="stretch")

# Step 4 — Entry (advisory) + position sizer.
with st.container(border=True):
    lv = payload.get("levels", {})
    st.markdown(step_badge("Step 4", "Entry — advisory"))
    info_btn(INFO_STEP4)
    bz = lv.get("buy_zone", (None, None))

    def _usd(x):
        return "n/a" if x is None else f"${x:,.2f}"

    bz_lo, bz_hi = (bz[0], bz[1]) if bz else (None, None)
    # Only ONE '$' here: two '$' in a single st.metric value get parsed as a LaTeX math
    # span ($...$), which renders that portion in a different (serif) font.
    buy_zone = ("n/a" if (bz_lo is None or bz_hi is None)
                else f"${bz_lo:,.2f} – {bz_hi:,.2f}")
    pct = lv.get("pct_to_pivot")
    pct_s = "n/a" if pct is None else f"{pct:+.1f}%"
    vol = lv.get("volume_ratio", 1)
    vol_metric = f"{vol:.1f}×" if isinstance(vol, (int, float)) else "n/a"
    price_ok = bool(lv.get("breakout_today"))       # price cleared the pivot
    vol_ok = bool(lv.get("volume_confirmed"))        # latest volume >= 1.5x the 20-day avg

    r1 = st.columns(3)
    r2 = st.columns(3)
    r1[0].metric("Pivot", _usd(lv.get("pivot")), border=True)
    r1[1].metric("Buy zone", buy_zone, border=True)
    r1[2].metric("Stop", _usd(lv.get("stop")), border=True)
    r2[0].metric("Target", _usd(lv.get("target")), border=True)
    r2[1].metric("To pivot", pct_s, border=True)
    r2[2].metric("Volume", vol_metric, border=True,
                 help="Latest-day volume ÷ its own 20-day average. A confirmed breakout "
                      "needs ≥ 1.5× (50%+ above that average).")

    # A tradeable breakout needs BOTH: price above the pivot AND volume confirmation.
    st.markdown(
        f"**Breakout:** price {'✅ above pivot' if price_ok else '— below pivot'} · "
        f"volume {'✅ confirmed' if vol_ok else '— not confirmed'}"
        + (f" ({vol:.1f}× vs 1.5×+ needed)" if isinstance(vol, (int, float)) else ""))

    # RMV (Relative Measured Volatility): advisory base-tightness read. Low = a quiet,
    # coiled base (the VCP sweet spot). Purely a hint — it does NOT move the levels above.
    rmv = lv.get("rmv")
    if rmv is None:
        rmv_disp, rmv_flag, rmv_label, rmv_note = "n/a", "", "n/a", "not enough history."
    elif rmv < 25:
        rmv_disp, rmv_flag, rmv_label = f"{rmv:.0f}", "✅", "tight"
        rmv_note = "low volatility, a classic VCP contraction — tight stop, high-quality base."
    elif rmv < 50:
        rmv_disp, rmv_flag, rmv_label = f"{rmv:.0f}", "", "normal"
        rmv_note = "middling volatility — the base isn't fully coiled yet."
    else:
        rmv_disp, rmv_flag, rmv_label = f"{rmv:.0f}", "⚠️", "loose"
        rmv_note = "still volatile — lower-quality base; consider waiting for it to tighten."
    rc = st.columns([1, 2])
    rc[0].metric("RMV", rmv_disp, border=True,
                 help="Relative Measured Volatility (0–100): today's price volatility vs "
                      "the stock's own recent range. Low = a tight, low-volatility base "
                      "(the VCP contraction). Advisory only — it does not move the levels.")
    rc[1].markdown(f"**Base tightness:** {rmv_flag} **{rmv_label}** — {rmv_note}")
    st.markdown("---")
    # Explicit keys pin these widgets' identity so their values persist across reruns.
    # Without a key, a keyless input's identity depends on the (variable) element count
    # above it, so a rerun could reset it to the default and the size would go stale
    # until a full page reload.
    acct = st.number_input("Account $", min_value=0.0, value=100_000.0, step=1000.0,
                           key="size_acct")
    risk_pct = st.number_input("Risk % per trade", min_value=0.0, value=1.0, step=0.25,
                               key="size_risk_pct")
    entry, stop = bz[0], lv.get("stop")
    if entry and stop and entry > stop:
        shares = (acct * risk_pct / 100.0) / (entry - stop)
        st.write(f"**Size:** {shares:,.0f} sh (~${shares * entry:,.0f}) "
                 f"at the pivot, risking {risk_pct:.2f}% to stop.")

st.caption("Educational tool — not financial advice. You place orders yourself.")
