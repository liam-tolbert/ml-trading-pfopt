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
from src.stock_screener.cockpit.export import (  # noqa: E402
    parse_ticker_list, watchlist_list_csv, watchlist_ohlcv_csv)
from src.stock_screener.cockpit.scan import ScanConfig, run_scan  # noqa: E402
from src.stock_screener.cockpit.trade import (  # noqa: E402
    TradeUnavailable, build_buy_plan, fetch_account_summary, stop_is_valid, submit_buy_plan)

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

The **next earnings date** shows here too — it's an *entry-timing* input
(see Step 4): don't open a fresh position within ~2–3 weeks of a report.
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
**Step 4 — Entry (you pull the trigger).** Two opposite states, in sequence — the same
two axes flip from quiet to loud:
- **The base** (you wait on it): volume **dries up** and volatility **contracts** (tight
  RMV / BBWP, squeeze on).
- **The breakout** (the trigger): price **closes above the pivot** on volume **≥ 40–50%
  above average** (the "breakout today" flag), and volatility **expands** (the squeeze fires).
- **Don't chase** more than ~5% above the pivot (the buy zone).
- **Check the calendar:** don't open a fresh position within ~2–3 weeks of a
  scheduled **earnings report** — with no profit cushion, an earnings gap can blow
  straight through the stop. (Minervini holds through earnings only with a cushion.)
- Set the **7–8% stop immediately** and never lower it.
- First target ~**20–25%**; trail with the 50-day SMA once well in profit.
- **Size** so a stop-out costs ~1% of the account (set Account $ / Risk %).

These levels are advisory — place the order in your broker.
"""

EARNINGS_SOON_DAYS = 21   # flag entries within ~3 weeks of a scheduled report


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
    "earnings_in": "Earnings in (days)",
    "tier": "Tier",
    "vcp": "VCP detected",
    "num_contractions": "# Contractions",
    "vcp_quality": "VCP quality (0-100)",
    "tier_reason": "Tier reason",
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
    "earnings_in": "Calendar days until the next scheduled earnings report (yfinance). "
                   "Minervini: don't open a fresh position within ~2–3 weeks of a report — "
                   "with no profit cushion, an earnings gap can blow straight through the "
                   "stop. Negative = just reported (the safest window); n/a = no date found.",
    "tier": "Review tier (recall-first): A = valid tightening base in/near the buy zone — "
            "review these. B = watch: base still forming, or valid but extended past the "
            "buy zone; never hidden. C = safely skipped (dead tape / no pullbacks / stale "
            "base) — the reason is in 'Tier reason'. Benchmarked on 200 hand-labeled "
            "charts: zero real setups landed in C.",
    "vcp": "A valid Volatility Contraction Pattern was detected: 2–6 progressively tighter "
           "pullbacks, the last one tight (≤12%), with price near its 52-week high.",
    "num_contractions": "Number of peak→trough pullbacks in the current base. Minervini's "
                        "range is 2–6.",
    "vcp_quality": "Base quality 0–100 (tightening 30 + volume-drying 20 + #contractions 20 "
                   "+ near-high 20 + base length 10). Shown even when VCP is False.",
    "tier_reason": "Why the detector placed the name in its tier — every exclusion or "
                   "demotion is auditable, nothing is silently hidden.",
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
    ("Base — the VCP setup", ["tier", "vcp", "num_contractions", "vcp_quality",
                              "tier_reason"]),
    ("Entry — timing & risk", ["earnings_in", "breakout_today", "vol_confirmed",
                               "pct_to_pivot", "pivot", "stop", "target"]),
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


def _earnings_flag(days) -> str:
    """'⚠︎ earnings in Nd' when a report is 0–21 days out, else '' (unknown/far/past)."""
    return (f"⚠︎ earnings in {int(days)}d"
            if days is not None and 0 <= days <= EARNINGS_SOON_DAYS else "")


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


# --------------------------------------------------------------------------- #
# Watchlist — a per-session shortlist you build by clicking, then export. The
# canonical store is a plain ordered list in session_state (NOT a widget key), so
# button callbacks and the multiselect can both mutate it without fighting over
# widget ownership. It's the export that persists it — download the CSV to keep it.
# --------------------------------------------------------------------------- #
def _wl() -> list:
    return st.session_state.setdefault("watchlist", [])


def _wl_add(ticker: str) -> None:
    wl = _wl()
    if ticker and ticker not in wl:
        wl.append(ticker)


def _wl_remove(ticker: str) -> None:
    wl = _wl()
    if ticker in wl:
        wl.remove(ticker)


def _wl_clear() -> None:
    st.session_state["watchlist"] = []


def _wl_add_from_picker() -> None:
    """Merge the bulk-add multiselect's picks into the watchlist, then clear the picker
    (resetting a widget's value is allowed inside its own on_change callback)."""
    for t in st.session_state.get("wl_picker", []):
        _wl_add(t)
    st.session_state["wl_picker"] = []


def _wl_add_from_upload() -> None:
    """Merge tickers from an uploaded .txt into the watchlist. Names may be separated by
    commas and/or any whitespace/newlines; each is upper-cased and de-duplicated. Fires on
    the uploader's on_change, so it processes a given file exactly once (not every rerun)."""
    up = st.session_state.get("wl_upload")
    if up is None:                                       # file was cleared/removed
        return
    try:
        text = up.getvalue().decode("utf-8", errors="ignore")
    except Exception:
        st.session_state["_wl_upload_msg"] = "Could not read that file."
        return
    before = len(_wl())
    for sym in parse_ticker_list(text):                  # commas OR whitespace/newlines
        _wl_add(sym)
    st.session_state["_wl_upload_msg"] = (
        f"Added {len(_wl()) - before} new ticker(s) from {getattr(up, 'name', 'the file')}.")


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
# TESTING: sp500 / tickers universes disabled — full_us only (restore the list to re-enable).
universe = st.sidebar.selectbox(
    "Universe", ["full_us"], index=0,
    # ["sp500", "full_us", "tickers"], index=0,
    format_func=lambda k: {
        "sp500": "S&P 500 (fast)",
        "full_us": "All US common stocks (~3–4k · slow first scan)",
        "tickers": "My tickers.txt",
    }[k],
    help="full_us = broad US common-stock universe from Nasdaq/NYSE listings (~3–4k names; "
         "the FIRST cold scan pulls thousands of price histories and can take several "
         "minutes — later scans reuse the cache and only fetch new days).")
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
# Sidebar — Watchlist (build by clicking ⭐ on charts / the picker; export to keep).
# Rendered here (right after the scan) so it always shows, even when later filters
# leave zero rows and the page st.stop()s below.
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("---")
    _watch = _wl()
    st.markdown(f"### ⭐ Watchlist ({len(_watch)})")
    _all_tickers = (res.candidates["ticker"].tolist()
                    if res.candidates is not None and len(res.candidates) else [])
    st.multiselect(
        "Add tickers", options=sorted(set(_all_tickers) | set(_watch)),
        key="wl_picker", on_change=_wl_add_from_picker,
        placeholder="Pick tickers to add…",
        help="Add several at once, or click the ⭐ button next to any chart. "
             "The list lives for this session — download it to keep it.")
    st.file_uploader(
        "Upload tickers (.txt)", type=["txt"], key="wl_upload",
        on_change=_wl_add_from_upload,
        help="A .txt file of ticker symbols separated by commas (and/or new lines) — "
             "e.g. `AAPL, MSFT, NVDA`. They're merged into the watchlist, upper-cased "
             "and de-duplicated. Pairs with the '⬇ Names (.txt)' download below.")
    _up_msg = st.session_state.pop("_wl_upload_msg", None)
    if _up_msg:
        st.caption(_up_msg)
    if _watch:
        st.caption(", ".join(_watch))
        _d1, _d2 = st.columns(2)
        _d1.download_button(
            "⬇ List (CSV)",
            watchlist_list_csv(res.candidates, _watch, DISPLAY_ORDER),
            file_name="watchlist.csv", mime="text/csv", width="stretch",
            help="Your shortlist with its decision columns (tier, pivot, stop, target, …), "
                 "in the order you added them.")
        _d2.download_button(
            "⬇ OHLCV (CSV)", watchlist_ohlcv_csv(_watch, res.payloads),
            file_name="watchlist_ohlcv.csv", mime="text/csv", width="stretch",
            help="Daily Open/High/Low/Close/Volume for every watchlisted name, stacked "
                 "long-format with a Ticker column.")
        st.download_button(
            "⬇ Names (.txt)", ",".join(_watch), file_name="watchlist.txt",
            mime="text/plain", width="stretch",
            help="Just the tickers, comma-separated — the format the uploader above reads "
                 "back in.")
        st.button("🗑 Clear watchlist", on_click=_wl_clear, width="stretch")
        _missing = [t for t in _watch if t not in res.payloads]
        if _missing:
            st.caption(f"⚠︎ {', '.join(_missing)} not in the current scan — the list CSV "
                       "keeps the ticker only and the OHLCV omits it. Re-scan the universe "
                       "that has them to include their data.")

        # --- Paper-trade the watchlist via Alpaca (paper account only) --------------- #
        st.markdown("---")
        st.markdown("**⚡ Paper trade (Alpaca)**")
        # How to size EACH name's market BUY — % of portfolio, raw $, raw share count, or
        # risk-to-stop (Minervini's sizer).
        _mode_label = st.selectbox(
            "Size each buy by", ["% of portfolio", "$ per name", "# shares", "Risk % to stop"],
            key="trade_mode",
            help="Applied to EACH watchlisted name. '% of portfolio' = that % of your Alpaca "
                 "equity per name; '$ per name' = that many dollars each; '# shares' = exactly "
                 "that many shares each; 'Risk % to stop' = size so a stop-out costs that % of "
                 "equity — shares = (equity × risk%) / (price − stop), Minervini's position "
                 "sizer. Needs a stop on the name.")
        if _mode_label == "% of portfolio":
            _mode = "pct"
            _amount = st.number_input("% of equity per name", min_value=0.0, value=5.0,
                                      step=0.5, key="trade_amt_pct")
            _size_note = f"{_amount:.1f}% of equity per name"
        elif _mode_label == "$ per name":
            _mode = "dollars"
            _amount = st.number_input("$ per name", min_value=0.0, value=5000.0,
                                      step=500.0, key="trade_amt_dol")
            _size_note = f"~${_amount:,.0f} per name"
        elif _mode_label == "# shares":
            _mode = "shares"
            _amount = float(st.number_input("Shares per name", min_value=0, value=100,
                                            step=10, key="trade_amt_sh"))
            _size_note = f"{int(_amount)} shares per name"
        else:                                    # Risk % to stop (Minervini position sizer)
            _mode = "risk"
            _amount = st.number_input("Risk % of equity per trade", min_value=0.0, value=1.0,
                                      step=0.25, key="trade_amt_risk",
                                      help="A stop-out costs about this % of equity. Note a "
                                           "risk-sized position is roughly risk% ÷ stop-distance% "
                                           "of equity — e.g. 1% risk with an 8% stop wants a "
                                           "12.5% position, which the 10% single-order cap "
                                           "clamps (realized risk then falls below target).")
            _size_note = f"{_amount:.2f}% of equity risked to each stop"
        st.caption(f"Market BUYs: {_size_note}. Paper account only; whole shares, "
                   "each order still capped at 10% of equity.")
        if st.button("Build trade plan", key="trade_build", width="stretch"):
            # Fetch the target account ONCE here (not every rerun) so the user can confirm
            # which paper account will be traded — and, for '% of portfolio', to size on
            # its live equity.
            try:
                _account = fetch_account_summary()
            except TradeUnavailable as _e:
                _account = {"error": str(_e)}
            _plan, _skip = build_buy_plan(_watch, res.payloads, mode=_mode,
                                          amount=_amount, equity=_account.get("equity"))
            # Bump a build counter used as a nonce in the per-ticker stop widget keys, so a
            # fresh Build re-seeds each stop to its computed default instead of Streamlit
            # retaining a stale edited value from the previous plan.
            _bn = st.session_state.get("trade_build_n", 0) + 1
            st.session_state["trade_build_n"] = _bn
            st.session_state["trade_plan"] = {"plan": _plan, "skipped": _skip,
                                              "account": _account, "build_ts": _bn}
            st.session_state.pop("trade_result", None)

        _tp = st.session_state.get("trade_plan")
        if _tp:
            _plan, _skip = _tp["plan"], _tp["skipped"]
            if _plan:
                _tot = sum(o["est_value"] for o in _plan)
                st.caption(f"**{len(_plan)} order(s)** · ~${_tot:,.0f} est.")
                # Master switch: attach a protective sell-stop to each order. When off, buys go
                # in naked and already-held names are skipped (no buy, no stop).
                _attach = st.toggle(
                    "Attach protective stop (sell-all, GTC)", value=True,
                    key="trade_attach_stop",
                    help="Places a stop-loss under each buy via an OTO order (its stop leg rides "
                         "the market buy as a DAY order, then becomes a persistent GTC stop on "
                         "the next re-arm). If a name is already held, no buy is sent — a GTC "
                         "stop protects the whole position and, per Minervini, only ever "
                         "RATCHETS UP: a re-arm that would lower the stop is ignored and the "
                         "existing higher stop kept (shown as 'stop_kept' 🔒). Edit each stop "
                         "below (defaults to the app-computed stop).")
                _nonce = _tp.get("build_ts")
                _eq = (_tp.get("account") or {}).get("equity")
                for _o in _plan:
                    _t = _o["ticker"]
                    _fl = " ⚠︎ extended" if _o["extended"] else ""
                    if _o.get("capped"):
                        _fl += " ⚠︎ capped"
                    _ew = _earnings_flag(_o.get("earnings_in"))
                    _cA, _cB = st.columns([3, 2])
                    _cA.caption(f"• **{_t}** {_o['shares']} sh @ "
                                f"~${_o['price']:.2f} (~${_o['est_value']:,.0f}){_fl}"
                                + (f" · {_ew}" if _ew else ""))
                    _cB.number_input(
                        f"stop {_t}", min_value=0.0,
                        value=float(_o["stop_price"]) if _o["stop_price"] else 0.0,
                        step=0.01, format="%.2f", key=f"stop_{_t}_{_nonce}",
                        label_visibility="collapsed", disabled=not _attach)
                    _edstop = st.session_state.get(f"stop_{_t}_{_nonce}", _o["stop_price"])
                    if _attach and not stop_is_valid(_edstop, _o["price"]):
                        _cB.caption(":red[stop must be < price]")
                    elif _attach and _eq and _edstop and _o["price"] > _edstop:
                        # Live risk-to-stop for the CURRENT shares + (possibly edited) stop, so a
                        # risk-sized position stays honest even after the stop is nudged (build
                        # doesn't re-scale shares on an edit).
                        _rusd = _o["shares"] * (_o["price"] - _edstop)
                        _cA.caption(f"  ↳ risk to stop ≈ {_rusd / _eq * 100:.2f}% (${_rusd:,.0f})")
                if any(_o["extended"] for _o in _plan):
                    st.caption("⚠︎ *extended* = >5% above the pivot; sized at pivot risk, so "
                               "the real risk to your stop is larger.")
                if any(_o.get("capped") for _o in _plan):
                    st.caption("⚠︎ *capped* = the risk-sized quantity hit the 10%-of-equity "
                               "order cap and was clamped down, so the realized risk sits below "
                               "your target. Lower the risk % or tighten the stop to fit.")
                if any(_earnings_flag(_o.get("earnings_in")) for _o in _plan):
                    st.caption(f"⚠︎ *earnings in Nd* = a report is scheduled within "
                               f"~{EARNINGS_SOON_DAYS} days. A fresh buy has no profit "
                               "cushion to absorb an earnings gap — Minervini would wait "
                               "for the report or already have a cushion.")
                # Confirm WHICH account before submitting (each paper account has its own keys).
                _account = _tp.get("account", {})
                if _account.get("error"):
                    st.warning(_account["error"])
                else:
                    _src = ("Minervini Trader keys" if _account.get("using_dedicated")
                            else "shared ALPACA_* keys — set ALPACA_API_KEY_MINERVINI / "
                                 "ALPACA_API_KEY_SECRET_MINERVINI to target the Minervini account")
                    st.caption(f"Target account **…{str(_account['account_number'])[-4:]}** "
                               f"({_src}) · equity ${_account['equity']:,.0f}")
                _c1, _c2 = st.columns(2)
                if _c1.button("✅ Submit (paper)", key="trade_submit",
                              type="primary", width="stretch"):
                    # Merge each ticker's edited stop (session_state) into the plan entries.
                    _final = [{**_o, "stop_price": st.session_state.get(
                        f"stop_{_o['ticker']}_{_nonce}", _o["stop_price"])} for _o in _plan]
                    with st.spinner("Submitting to Alpaca paper…"):
                        try:
                            st.session_state["trade_result"] = submit_buy_plan(
                                _final, attach_stop=_attach)
                        except TradeUnavailable as _e:
                            st.session_state["trade_result"] = {"error": str(_e)}
                    st.session_state.pop("trade_plan", None)
                    st.rerun()
                if _c2.button("Cancel", key="trade_cancel", width="stretch"):
                    st.session_state.pop("trade_plan", None)
                    st.rerun()
            else:
                st.caption("No tradable orders from the current watchlist.")
            if _skip:
                st.caption("Skipped: "
                           + " · ".join(f"{s['ticker']} ({s['reason']})" for s in _skip))

        _tr = st.session_state.get("trade_result")
        if _tr:
            if _tr.get("error"):
                st.error(f"Trade failed: {_tr['error']}")
            else:
                _act = [r for r in _tr["results"]
                        if r["status"] in ("submitted", "stop_only", "stop_kept")]
                st.success(f"Actioned {len(_act)}/{len(_tr['results'])} order(s) on "
                           f"account …{str(_tr['account_number'])[-4:]} · "
                           f"equity ${_tr['equity']:,.0f}")
                for _r in _tr["results"]:
                    _ic = {"submitted": "✅", "stop_only": "🛑", "stop_kept": "🔒",
                           "skipped": "—", "failed": "⚠️"}.get(_r["status"], "•")
                    st.caption(f"{_ic} {_r['ticker']}: {_r['status']} — {_r.get('detail', '')}")
    else:
        st.caption("Empty — click ⭐ on a chart, or use the picker above.")

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

# Steps 2 + 3 share one row: the large chart on the LEFT, with the Step-2 (fundamentals)
# and Step-3 (chart controls) boxes stacked in a same-width column on the RIGHT — read the
# fuel, then judge the base, without scrolling. The side column is written FIRST in code so
# its control values (weekly / overlays / bollinger) exist before the chart builds; it still
# renders on the right because it's the second column returned.
colChart, colSide = st.columns([3, 1])

with colSide:
    # Step 2 — Fundamentals, condensed to sit beside the chart (~same height as Step 3).
    with st.container(border=True):
        st.markdown(step_badge("Step 2", "Fundamentals — the fuel"))
        info_btn(INFO_STEP2)
        f = payload.get("fundamentals")
        s2 = payload.get("step2", {})
        if not f:
            st.caption("No fundamental data available (yfinance).")
        else:
            def _p(v):                       # signed % (growth), or n/a
                return "n/a" if v is None else f"{v:+.1f}%"

            mt = f.get("margin_trend")
            opm = f.get("operating_margin")
            st.markdown(f"**Rev:** {_p(f.get('revenue_yoy'))} YoY · "
                        f"{_p(f.get('revenue_qoq'))} QoQ")
            st.markdown(f"**EPS:** {_p(f.get('eps_yoy'))} YoY · "
                        f"{_p(f.get('eps_qoq'))} QoQ")
            st.markdown(f"**Op margin:** {'n/a' if opm is None else f'{opm:.1f}%'} "
                        f"(Δ {'n/a' if mt is None else f'{mt:+.1f}pp'})")
            ne, ei = f.get("next_earnings"), payload.get("earnings_in")
            if ne:
                when = ("" if ei is None
                        else f" ({-ei}d ago)" if ei < 0 else f" (in {ei}d)")
                warn = " ⚠️" if _earnings_flag(ei) else ""
                st.markdown(f"**Earnings:** {ne}{when}{warn}")
            checks = s2.get("checks", {})
            st.markdown(" ".join(
                f"{'✅' if checks.get(k) else '—'} {lbl}"
                for k, lbl in [("revenue_growth", "Rev ≥20%"), ("eps_growth", "EPS ≥20%"),
                               ("eps_accelerating", "EPS accel"),
                               ("margin_expanding", "Margin ↑")]))
            st.caption(f"Score {s2.get('score', 0)}/4")

    # Step 3 — chart controls (the chart itself renders in colChart on the left).
    with st.container(border=True):
        st.markdown(f"### {pick}")
        # Add/remove the charted name to the export watchlist (the "judge it → keep it" flow).
        if pick in _wl():
            st.button(f"✓ In watchlist — remove {pick}", key="wl_toggle",
                      on_click=_wl_remove, args=(pick,), width="stretch")
        else:
            st.button(f"⭐ Add {pick} to watchlist", key="wl_toggle", type="primary",
                      on_click=_wl_add, args=(pick,), width="stretch")
        st.markdown(step_badge("Step 3", "Judge the VCP"))
        info_btn(INFO_STEP3, label="ℹ️ How to read the chart")
        weekly = st.checkbox("Weekly view", value=False)
        show_overlays = st.checkbox("VCP + entry overlays", value=True)
        show_bollinger = st.checkbox(
            "Bollinger bands", value=False,
            help="Overlay the 20-period / 2σ Bollinger envelope on the price row — the same "
                 "bands the Step-4 squeeze (BBWP) read is built from. Narrowing bands = the "
                 "compression a VCP base coils into.")

with colChart:
    with st.container(border=True):
        _ranges = {"3M": 90, "6M": 180, "9M": 270, "1Y": 365, "2Y / All": None}
        rsel = st.radio("Time range", list(_ranges), index=1, horizontal=True,
                        help="Zoom in to see the VCP base; a tight base is hard to read over 2 years.")
        fig = build_chart(pick, payload["df"], vcp=payload.get("vcp"),
                          levels=payload.get("levels"), show_overlays=show_overlays,
                          weekly=weekly, lookback_days=_ranges[rsel],
                          show_bollinger=show_bollinger)
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
    price_ok = bool(lv.get("breakout_today"))       # price cleared the pivot
    vol_ok = bool(lv.get("volume_confirmed"))        # latest volume >= 1.5x the 20-day avg

    # Price levels — neutral (WHERE you'd act, not WHETHER to).
    r1 = st.columns(3)
    r1[0].metric("Pivot", _usd(lv.get("pivot")), border=True)
    r1[1].metric("Buy zone", buy_zone, border=True)
    r1[2].metric("Stop", _usd(lv.get("stop")), border=True)
    r2 = st.columns(2)
    r2[0].metric("Target", _usd(lv.get("target")), border=True)
    r2[1].metric("To pivot", pct_s, border=True)
    # Risk from the pivot (buy point) to the stop — Minervini's 7-8% ideal / 10% hard max.
    spp = lv.get("stop_pct_from_pivot")
    if spp is not None:
        _clamp = (" — capped at the 10% max (logical support sat lower; a base needing a wider "
                  "stop is too loose to risk more than 10%)" if lv.get("stop_clamped") else "")
        _mark = "✅" if spp <= 8.0 + 1e-9 else "⚠️"
        st.caption(f"{_mark} Risk pivot → stop: **{spp:.1f}%** "
                   f"(Minervini: 7–8% ideal, 10% hard max){_clamp}")

    # Step 4 has two OPPOSITE states, one after the other in time. The SAME two axes —
    # volume and volatility — flip from quiet to loud:
    #   THE BASE      (you wait on it):  volume dries up, volatility contracts  → QUIET
    #   THE BREAKOUT  (the trigger):     volume surges,   volatility expands     → LOUD
    vcp_data = payload.get("vcp", {})
    st.markdown("**The base — what you're waiting on (should be _quiet_):**")

    # Base volatility (tight): RMV, then BBWP / squeeze as a cross-check. Point-in-time, so as
    # a breakout fires these naturally rise (the base tightness is "spent") and the breakout
    # reads below light up — the same volatility axis handing off from quiet to loud.
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
    rc[1].markdown(f"**Base volatility:** {rmv_flag} **{rmv_label}** — {rmv_note}")

    bbwp = lv.get("bbwp")
    squeeze_on = bool(lv.get("squeeze"))
    if bbwp is None:
        bbwp_disp, bbwp_note = "n/a", "not enough history."
    elif bbwp < 25:
        bbwp_disp, bbwp_note = f"{bbwp:.0f}", "band-width in its bottom quartile — a Bollinger squeeze (tight)."
    elif bbwp < 50:
        bbwp_disp, bbwp_note = f"{bbwp:.0f}", "band-width mid-range — not yet a squeeze."
    else:
        bbwp_disp, bbwp_note = f"{bbwp:.0f}", "band-width wide — volatility is expanded, not contracted."
    sq_flag = "✅ squeeze on" if squeeze_on else "— no squeeze"
    bc = st.columns([1, 2])
    bc[0].metric("BBWP", bbwp_disp, border=True,
                 help="Bollinger Band-Width Percentile (0–100): today's Bollinger band width "
                      "vs its own trailing range. Low = a squeeze (bands tight). Cross-checks "
                      "RMV from the Bollinger (close-based) side. Advisory only.\n\n"
                      "**Rule of thumb:** prioritize RMV — it's the gate. BBWP is close-to-"
                      "close, so a smooth uptrend inflates it: if RMV reads tight but BBWP "
                      "isn't a squeeze (a low-range drift, e.g. IFF), that's usually the trend "
                      "fooling BBWP — go with RMV. Only defer to BBWP when it's *more* cautious "
                      "than RMV — RMV can read falsely tight after a recent volatility spike "
                      "(its min-max scaling), and there BBWP's skepticism wins.")
    bc[1].markdown(f"**Squeeze:** {sq_flag} — {bbwp_note}")

    # Base volume (should be DRYING UP): % of contractions whose volume ran lighter than the
    # advance into them (vcp volume_quality). This is the quiet-base counterpart to the loud
    # breakout-volume surge below — a different yardstick (vs. the run-up, not vs. 1.5× avg).
    vq = vcp_data.get("volume_quality")
    if vq is None:
        st.markdown("**Base volume:** n/a — no contractions detected.")
    else:
        vq_flag = "✅" if vq >= 60 else ("" if vq >= 30 else "⚠️")
        vq_note = ("volume dried up through the base — supply withdrawing." if vq >= 60
                   else "only a partial dry-up — mixed." if vq >= 30
                   else "volume did NOT dry up — a weaker base.")
        st.markdown(f"**Base volume:** {vq_flag} drying up in **{vq:.0f}%** of contractions — {vq_note}")

    st.markdown("**The breakout — the entry trigger (should be _loud_):**")
    # Price cleared the pivot?
    st.markdown(f"- **Price:** {'✅ above pivot' if price_ok else '— below the pivot (no trigger yet)'}")
    # Volume SURGE — the loud counterpart to base dry-up (latest bar vs its 20-day average).
    vol_txt = f"{vol:.1f}× the 20-day average" if isinstance(vol, (int, float)) else "n/a"
    st.markdown(f"- **Volume:** {'✅' if vol_ok else '—'} {vol_txt} (a breakout needs ≥ 1.5×)")
    # Volatility EXPANDING — the squeeze firing (the loud counterpart to the tight base).
    if bool(lv.get("squeeze_released")):
        volat_txt = "✅ squeeze released — volatility expanding out of the base"
    elif squeeze_on:
        volat_txt = "— still coiled (squeeze on) — no expansion yet"
    else:
        volat_txt = "— no active squeeze to release"
    st.markdown(f"- **Volatility:** {volat_txt}")
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
