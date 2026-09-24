"""SEPA Cockpit — Streamlit UI.

Run from the project root:

    streamlit run src/stock_screener/cockpit/app.py

The sidebar holds the scan filters, the watchlist and paper trading. The main pane shows
the candidate table and, for the selected name, the chart, Step-2 fundamentals and
Step-4 advisory entry levels. The tool filters mechanically; the user judges the VCP.
"""
from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:                       # so `from src.X import ...` resolves
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
from streamlit.errors import StreamlitAPIException  # noqa: E402

from src.stock_screener.cockpit import advisories  # noqa: E402 (display-only SEPA reads)
from src.stock_screener.cockpit import cache  # noqa: E402 (path read at call time → patchable)
from src.stock_screener.cockpit import journal_cache  # noqa: E402 (shared fills cache)
from src.stock_screener.cockpit import scan_worker  # noqa: E402
from src.stock_screener.cockpit import entries  # noqa: E402 (armed next-open entries)
from src.stock_screener.cockpit import trade  # noqa: E402 (module import → patchable in tests)
from src.stock_screener.cockpit.charts import build_chart  # noqa: E402
from src.stock_screener.cockpit.export import (  # noqa: E402
    load_watchlist, make_entry, merge_frozen_pivots, parse_ticker_list, save_watchlist,
    watchlist_list_csv, watchlist_ohlcv_csv, watchlist_tickers)
from src.stock_screener.cockpit.scan import filter_candidates  # noqa: E402
from src.stock_screener.cockpit.trade import (  # noqa: E402
    STALE_PLAN_BARS, TradeUnavailable, build_buy_plan, cancel_pending_buys,
    fetch_account_summary, fetch_gate_inputs, fetch_held_shares, fill_floor, freshen_prices,
    gate_status, stop_is_valid, stop_within_max_loss, submit_buy_plan)
from src.stock_screener.cockpit.triggers import (load_latest_trigger_report,  # noqa: E402
                                                 save_trigger_report)

st.set_page_config(page_title="SEPA Cockpit", layout="wide")

# Less top padding and a tighter element gap, so the candidate table is visible on load.
st.markdown(
    "<style>"
    # padding-top MUST stay >= the fixed header height (~3.75rem) or the top row slides
    # under it. 4rem clears it and still saves ~2rem on the default.
    ".block-container{padding-top:4rem;padding-bottom:2rem;}"
    'div[data-testid="stVerticalBlock"]{gap:0.6rem;}'
    "</style>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Contextual help for the ℹ️ popovers (full reference: pages/1_SEPA_Guide.py)
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

- **RS** = relative-strength rating (IBD-style weighted blend of 3/6/9/12-mo returns,
  recent 3-mo counted double, percentiled vs the scanned set); Minervini wants **70+**.
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
- Set the **stop immediately** — 7–8% below the price you pay, **never more than 10%** —
  and never lower it. The loss is measured from your fill, not the pivot: buying higher in
  the zone means a tighter stop (the trade plan raises it for you).
- First target **+25%** above the pivot; trail with the 50-day SMA once well in profit.
- **Size** so a stop-out costs ~1% of the account (set Account $ / Risk %).

These levels are advisory — place the order in your broker.
"""

from src.stock_screener.cockpit.doctrine import (DEFAULT_STOP_FROM_PIVOT, EARNINGS_SOON_DAYS,
                                                 MAX_LOSS_FROM_FILL, REGIME_CONFIRM_DAYS)



# Raw candidate-frame column -> display label, for the table headers and the filter
# picker. Only the display is relabelled: selection and filtering use the raw names.
READABLE_COLS = {
    "ticker": "Ticker",
    "price": "Price ($)",
    "rs": "RS rating",
    "rs_nh": "RS line NH",
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
    "breakout_today": "Breakout today",
    "vol_confirmed": "Vol confirmed",
    "pct_to_pivot": "Distance to pivot (%)",
    "day_range": "Typical day (%)",
    "pivot": "Pivot ($)",
    "stop": "Stop ($)",
    "target": "Target ($)",
}

# Per-column meaning, for the header tooltips and the ℹ️ Columns popover.
COL_HELP = {
    "ticker": "Stock symbol. Click a row to chart it.",
    "price": "Latest close price.",
    "rs": "Relative-strength rating 1–99: IBD-style weighted return blend (2×3-mo + 6-mo "
          "+ 9-mo + 12-mo — recent strength counts double), percentiled vs the scanned "
          "universe. Minervini wants 70+.",
    "rs_nh": "RS line (price ÷ SPY) at its 52-week high while the PRICE is still below its "
             "own 52-week high — IBD's 'RS new high before price' blue dot: the stock is "
             "outperforming the market while still basing, the classic institutional-"
             "accumulation tell. One of the strongest breakout-confirmation signals for a "
             "coiled name. n/a = under ~6 months of overlapping history.",
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
            "base). Benchmarked on 200 hand-labeled charts: zero real setups landed in C.",
    "vcp": "A valid Volatility Contraction Pattern was detected: 2–6 progressively tighter "
           "pullbacks, the last one tight (≤12%), with price near its 52-week high.",
    "num_contractions": "Number of peak→trough pullbacks in the current base. Minervini's "
                        "range is 2–6.",
    "vcp_quality": "Base quality 0–100 (tightening 30 + volume-drying 20 + #contractions 20 "
                   "+ near-high 20 + base length 10). Shown even when VCP is False.",
    "breakout_today": "Price is clearing the pivot right now (price only — see 'Vol "
                      "confirmed' for the volume side).",
    "vol_confirmed": "The scan's 20-day context read (latest volume ≥ 1.5× it). The trigger "
                     "job confirms on the prior 50-day average instead — this column is a "
                     "hint, the trigger report is the decision.",
    "pct_to_pivot": "Distance from price to the pivot. Positive = below pivot (needs to rise); "
                    "negative = already above/extended.",
    "day_range": "How far this stock ordinarily moves in a day: the median daily true range "
                 "over the last ~2 months, as % of price. A stop only one or two of these "
                 "below your buy gets hit by normal noise before the trade can work — a "
                 "wild mover needs a wider stop (and so a smaller position), or a pass.",
    "pivot": "Buy-trigger line — the breakout/base level (or 52-wk high). Buy a close above it.",
    "stop": "Advisory stop-loss, ~7–8% below the pivot. The 10% maximum loss is measured "
            "from the price you PAY, so a trade plan raises this stop for a fill higher in "
            "the buy zone.",
    "target": "First objective, +25% above the pivot.",
}

# The table's four decision groups, in display order. They set column order and
# visibility, and the ℹ️ Columns popover. `criteria` is left out: the 8/8 gate makes it a
# constant 8. It stays in the scan frame, where tests read it.
COL_GROUPS = [
    ("Identify", ["ticker", "price"]),
    ("Fuel — catalyst & strength", ["rs", "rs_nh", "fund_score", "rev_yoy", "eps_yoy",
                                    "op_margin"]),
    ("Base — the VCP setup", ["tier", "vcp", "num_contractions", "vcp_quality"]),
    ("Entry — timing & risk", ["earnings_in", "breakout_today", "vol_confirmed",
                               "pct_to_pivot", "day_range", "pivot", "stop", "target"]),
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
    """'⚠︎ earnings in Nd' when a report is 0 to ``EARNINGS_SOON_DAYS`` days out, else ''."""
    return (f"⚠︎ earnings in {int(days)}d"
            if days is not None and 0 <= days <= EARNINGS_SOON_DAYS else "")


def _regime_color(regime) -> str:
    """Strong/moderate Risk-On -> green, Risk-Off -> red, anything else -> orange."""
    return {"strong": "green", "off": "red"}.get(advisories.regime_tier(regime), "orange")


def step_badge(step: str, title: str) -> str:
    """A consistent blue step chip + title, e.g. ':blue-background[Step 3]  Judge the VCP'."""
    return f":blue-background[{step}]  {title}"


def filter_table(df, key_prefix: str = "flt"):
    """Render value filters for the columns the user picks; return the rows of ``df``
    that pass all of them. ``key_prefix`` namespaces the widget keys.

    Numbers get a range slider; n/a rows drop out only once the range is narrowed.
    Booleans get an All/True/False picker, text a multi-select of distinct values.
    """
    import pandas as pd

    readable = READABLE_COLS

    out = df
    with st.expander("🔎 Filter by column values", expanded=False):
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
                    if (lo, hi) != (cmin, cmax):        # full range keeps n/a rows
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
# Watchlist
# --------------------------------------------------------------------------- #
# The store is an ordered list of entry dicts {ticker, judged_pivot, date_added,
# pivot_source, note} in session_state. It MUST NOT be a widget key: button callbacks and
# the multiselect both mutate it. `judged_pivot` is the frozen trigger level. A ⭐ add
# freezes the charted pivot ("judged"). Picker and .txt adds stay unfrozen until 📌 or the
# trigger check ("auto") freezes one. `_wl()` loads cache.WATCHLIST_JSON once per session.
# Every mutation merges with the file and saves it: the refresh job writes it too.
def _wl() -> list:
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = load_watchlist(cache.WATCHLIST_JSON)
    return st.session_state["watchlist"]


def _wl_tickers() -> list:
    return watchlist_tickers(_wl())


def _wl_entry(ticker: str):
    return next((e for e in _wl() if isinstance(e, dict) and e.get("ticker") == ticker), None)


def _wl_persist() -> None:
    # The file MUST be merged in before it is rewritten. cockpit-refresh auto-freezes pivots
    # into it while this session holds an older copy; a blind rewrite would clobber them.
    # Disk pivots win for entries left unfrozen here. The session wins membership, order
    # and its own freezes. The merge becomes the session copy, so the UI shows the
    # adopted pivots.
    merged = merge_frozen_pivots(_wl(), load_watchlist(cache.WATCHLIST_JSON))
    st.session_state["watchlist"] = merged
    save_watchlist(cache.WATCHLIST_JSON, merged)


def _invalidate_trade_plan() -> None:
    """Drop the built trade plan and its submit result.

    A plan is a snapshot of prices, sizing and watchlist membership. Every change to those
    MUST call this, so a stale plan can't stay rendered and submittable. ``trade_build_n``
    is left alone: the next Build bumps it and re-seeds the row widget keys."""
    st.session_state.pop("trade_plan", None)
    st.session_state.pop("trade_result", None)


def _do_disarm(ticker) -> None:
    """The Disarm callback: mark ``ticker``'s row disarmed in the latest entry plan and
    save it. Errors are swallowed.

    The plan MUST be re-read from disk here: the executor or another session may have
    rewritten it since this render."""
    try:
        p = entries.load_latest_entry_plan()
        if p and entries.disarm_row(p, ticker):
            entries.save_entry_plan(p)
    except Exception:
        pass


def _risk_guidance():
    """Recent-form sizing guidance for the risk mode: :func:`trade.suggest_risk_pct` over
    the cockpit's own tagged closed trades. None when the journal read fails.

    Memoized in session state per ``jr_nonce`` for ``FILLS_MAX_AGE_S``, the fills' own max
    age, so a newly closed trade reaches the sizing. A failure is memoized too: an Alpaca
    outage MUST cost one fetch per window, not a block on every rerun."""
    n = st.session_state.get("jr_nonce", 1)
    memo = st.session_state.get("risk_guide")
    if (memo is not None and memo.get("nonce") == n
            and time.monotonic() - memo.get("mono", 0.0) < journal_cache.FILLS_MAX_AGE_S):
        return memo["data"]
    try:
        fills = journal_cache.cached_fills(n)["fills"]
        closed = [t for t in trade.build_trade_journal(fills)["closed"] if t["tagged"]]
        data = trade.suggest_risk_pct(closed)
    except Exception:
        data = None
    st.session_state["risk_guide"] = {"nonce": n, "mono": time.monotonic(), "data": data}
    return data


def _apply_risk_suggestion(v: float) -> None:
    # on_click runs before widgets instantiate, so writing the widget key is safe here.
    # The same write mid-script raises.
    st.session_state["trade_amt_risk"] = float(v)
    _invalidate_trade_plan()


def _wl_add(ticker: str, judged_pivot=None, note: str = "", persist: bool = True) -> None:
    entry = make_entry(ticker, judged_pivot, date_added=datetime.date.today().isoformat(),
                       pivot_source="judged" if judged_pivot else None, note=note)
    if entry and entry["ticker"] not in _wl_tickers():   # present ticker -> no-op (📌 re-freezes)
        _wl().append(entry)
        _invalidate_trade_plan()                         # covers the picker/upload bulk adders too
        if persist:                                      # bulk adders persist ONCE at the end
            _wl_persist()


def _wl_freeze(ticker: str, pivot) -> None:
    """The 📌 callback: set an existing entry's judged pivot to ``pivot`` and persist.

    The source becomes "judged", which overrides an auto-frozen pivot. ``date_added``
    moves to today, the date of this decision; the note is kept. A no-op for an invalid
    pivot or a ticker not on the watchlist."""
    probe = make_entry(ticker, pivot)                    # normalizes + validates the pivot
    ent = _wl_entry(str(ticker or "").strip().upper())
    if probe is None or probe["judged_pivot"] is None or ent is None:
        return
    ent["judged_pivot"] = probe["judged_pivot"]
    ent["date_added"] = datetime.date.today().isoformat()
    ent["pivot_source"] = "judged"
    _invalidate_trade_plan()                             # frozen pivot feeds the plan's stops
    _wl_persist()


def _wl_remove(ticker: str) -> None:
    wl = _wl()
    kept = [e for e in wl
            if (e.get("ticker") if isinstance(e, dict) else e) != ticker]
    if len(kept) != len(wl):
        st.session_state["watchlist"] = kept
        _invalidate_trade_plan()
        _wl_persist()

def _wl_sync_from_picker() -> None:
    """The watchlist multiselect's on_change: sync its pills back into the watchlist.

    The page re-seeds the widget from the list; this is the other direction. A new pick
    becomes an unfrozen entry. A pill dismissed with × drops the entry and its frozen
    pivot, so a later re-add freezes at the current pivot. Persists once, on any change."""
    picked = list(st.session_state.get("wl_picker", []))
    have = set(_wl_tickers())
    changed = False
    for t in picked:
        if t not in have:
            _wl_add(t, persist=False)                    # persist once below, not per pick
            changed = True
    keep = set(picked)
    kept = [e for e in _wl()
            if (e.get("ticker") if isinstance(e, dict) else e) in keep]
    if len(kept) != len(_wl()):
        st.session_state["watchlist"] = kept
        _invalidate_trade_plan()
        changed = True
    if changed:
        _wl_persist()


def _wl_add_from_upload() -> None:
    """The uploader's on_change: merge an uploaded .txt's tickers into the watchlist.

    Names split on commas and whitespace, and are upper-cased and de-duplicated. As an
    on_change it runs once per file, not on every rerun. Leaves a message for the page in
    ``_wl_upload_msg``."""
    up = st.session_state.get("wl_upload")
    if up is None:                                       # file was cleared/removed
        return
    try:
        text = up.getvalue().decode("utf-8", errors="ignore")
    except Exception:
        st.session_state["_wl_upload_msg"] = "Could not read that file."
        return
    before = len(_wl())
    for sym in parse_ticker_list(text):
        _wl_add(sym, persist=False)                      # persist once below, not per name
    if len(_wl()) != before:
        _wl_persist()
    st.session_state["_wl_upload_msg"] = (
        f"Added {len(_wl()) - before} new ticker(s) from {getattr(up, 'name', 'the file')}.")


# --------------------------------------------------------------------------- #
# Sidebar — scan settings
# --------------------------------------------------------------------------- #
st.sidebar.title("SEPA Cockpit")
st.sidebar.caption("Mechanical Steps 1-2; you judge Steps 3-4.")

with st.sidebar.popover("ℹ️ How to use this tool"):
    st.markdown(
        "1. **Check the market environment** banner — only push in a healthy tape.\n"
        "2. **Step 1:** the table lists trend-template passers (eligibility).\n"
        "3. **Step 2:** rank by fundamental quality (the 'fuel').\n"
        "4. **Step 3:** click a row and *judge the VCP yourself* on the chart.\n"
        "5. **Step 4:** if it breaks out on volume, use the advisory entry/stop/size.\n\n"
        "Each section has its own **ℹ️** button. Full details on the **SEPA Guide** page.")
# The universe is always the full US common-stock list. data_feed falls back to the cached
# sp500 list only when no copy of that listing can be read. The universe and gate live in
# scan_worker, so the other pages' warm-up starts the same scan this page reads.
min_criteria = scan_worker.DEFAULT_MIN_CRITERIA  # all 8 trend-template criteria
st.sidebar.caption("Universe: **all US common stocks** (~3–4k names from Nasdaq/NYSE "
                   "listings). ⏳ The first cold scan pulls every price history (several "
                   "minutes) — it runs in the background, so you can browse the other "
                   "pages while it finishes; later scans use the cache and fetch only "
                   "new days.")
st.sidebar.caption("Gate: full **8/8** trend template")
min_rs = st.sidebar.slider("Min RS rating", 0, 99, 70,
                           help="IBD-style weighted multi-horizon return percentile "
                                "(2×3-mo + 6-mo + 9-mo + 12-mo) vs the scanned universe")
require_vcp = st.sidebar.checkbox("VCP only (hint filter)", value=False)
min_fund = st.sidebar.slider("Min fundamental checks (0-4)", 0, 4, 0)

# The scan runs in scan_worker's daemon thread. It starts when any cockpit page loads and
# survives page switches: a switch cancels the script run, never the worker. The worker
# memoizes per (universe, gate, generation). Re-scan bumps the generation, forcing a fresh
# run that also tops up the latest bars. The min_rs / require_vcp / min_fund sliders MUST
# stay out of that key: the scan runs once at the loosest gates and the sliders are
# instant post-filters (`filter_candidates`).
_worker = scan_worker.get_worker()
if st.sidebar.button("🔄 Re-scan (refresh prices)", key="rescan"):
    _worker.request_rescan()
    _invalidate_trade_plan()                             # plan prices predate the re-scan
# Tucked away so a misclick can't cost a multi-minute refetch. New tickers don't need it:
# a name with no cache is fully fetched on any scan.
with st.sidebar.expander("⚙ Advanced"):
    if st.button("⟳ Full re-download (2y, slow)", key="full_refetch",
                 help="Ignores every price cache and re-downloads the full 2-year history "
                      "for ALL names in the universe (several minutes; yfinance rate-limit "
                      "risk). Only for re-baselining suspect caches — new tickers are "
                      "fetched in full automatically, and normal scans already top up "
                      "the latest bars."):
        _worker.request_rescan(force=True)
        _invalidate_trade_plan()

_worker.ensure_started()
# Stale while refreshing: latest() returns the newest result in the process, even
# mid-refresh. That is the current run's, or the store's from another session or an
# earlier scan. The full-page wait below is reached only on a true cold start. Under
# AppTest latest() stays None mid-run, so tests block in wait() deterministically.
res = _worker.latest()
if res is None:
    # A short grace lets a quick warm run (fresh cache, test fake) render in one pass.
    # wait() anchors the grace to the run's start, so reruns during a long cold scan fall
    # straight through to the wait below.
    res = _worker.wait(grace=3.0)
if res is None:
    _snap = _worker.snapshot()
    if _snap["status"] == "error":
        st.error("Scan failed — the market-data fetch or screen raised:")
        st.code(_snap["error"] or "unknown error", language=None)
        if st.button("🔁 Retry scan", key="scan_retry"):
            _worker.request_rescan()
            st.rerun()
        st.stop()
    # True cold start only: the first scan on this machine. Later restarts load the
    # persisted last scan at once and show progress in the status line.
    st.info("First scan in progress — the table appears when it completes (a few "
            "minutes cold). Switching pages won't cancel it. After this one-time "
            "scan, restarts load the last result instantly.")
    time.sleep(1.0)
    st.rerun()

# The page renders from this one res for the whole run, so a refresh landing mid-render
# can't tear it; the next run adopts it. A changed as_of means the data under a built
# trade plan changed: to the plan, a background refresh is a re-scan.
_snap0 = _worker.snapshot()
_res_as_of = _snap0.get("as_of")
if st.session_state.get("_res_as_of") not in (None, _res_as_of):
    _invalidate_trade_plan()
st.session_state["_res_as_of"] = _res_as_of

# The status line is a fragment that repaints itself: 2s while a run is in flight, 30s
# idle. The table stays usable while fresh data loads.
_frag_iv = "2s" if _snap0["status"] == "running" else "30s"


def _clock(ts) -> str:
    """``ts`` on a 12-hour clock, e.g. '2:22 PM', with the date prepended unless it is
    today. A bare time on yesterday's table would read as fresh."""
    now = datetime.datetime.now()
    return (ts.strftime("%I:%M %p").lstrip("0") if ts.date() == now.date()
            else ts.strftime("%b %d %I:%M %p").replace(" 0", " "))


def _price_asof():
    """When prices were last topped up, as a :func:`_clock` string, or None.

    Read from the newest trigger report's ``generated_at``. cockpit-refresh stamps it every
    run, so it dates the price cache. The scan's ``as_of`` dates the last screen instead.
    None when there is no report, the stamp won't parse, or the read fails."""
    try:
        rep = load_latest_trigger_report(cache.TRIGGERS_DIR)
        raw = str((rep or {}).get("generated_at") or "")
        if not raw:
            return None
        return _clock(datetime.datetime.fromisoformat(raw).replace(tzinfo=None))
    except Exception:
        return None


@st.fragment(run_every=_frag_iv)
def _scan_status_line() -> None:
    s = _worker.snapshot()
    _ts = (_clock(datetime.datetime.fromtimestamp(s["as_of"]))
           if s.get("as_of") else None)
    _px = _price_asof()
    # "scan", not "data": the table is the last screen, which only Re-scan advances. Prices
    # refresh on their own schedule and are usually far newer, so they get their own stamp.
    _tail = f" · prices {_px}" if _px else ""
    if s["status"] == "running":
        st.caption((f"scan {_ts}{_tail} · " if _ts else "")
                   + f"⏳ {s['phase_label']} {s['done']}/{s['total']} — refreshing in "
                     "the background")
    elif s["status"] == "error" and _ts:
        st.caption(f":orange[⚠ background refresh failed — showing scan {_ts}]{_tail}")
        if st.button("🔁 Retry refresh", key="bg_retry"):
            _worker.request_rescan()
            try:                                  # fragment → escalate to app scope
                st.rerun(scope="app")
            except StreamlitAPIException:
                st.rerun()
    elif _ts and s.get("as_of") != _res_as_of:
        # A refresh finished since this page rendered. It MUST NOT swap in by itself, so
        # data never changes mid-read. The button, or any interaction via ensure_started,
        # adopts it.
        if st.button(f"⬆ Updated scan ready ({_ts}) — load", key="adopt_new"):
            try:
                st.rerun(scope="app")
            except StreamlitAPIException:
                st.rerun()
    elif _ts:
        st.caption(f"scan {_ts}{_tail}")


_scan_status_line()

# The sliders mask the memoized result; they never re-screen. The watchlist CSV export
# MUST use the unfiltered frame, so a watchlisted name keeps its columns at any setting.
cand_view = filter_candidates(res.candidates, min_rs, require_vcp, min_fund)

# --------------------------------------------------------------------------- #
# Sidebar — Watchlist
# --------------------------------------------------------------------------- #
# Rendered before the table: the page st.stop()s when the filters leave zero rows, and
# the watchlist MUST still show.
with st.sidebar:
    st.markdown("---")
    _watch = _wl()
    _watch_t = _wl_tickers()
    st.markdown(f"### ⭐ Watchlist ({len(_watch)})")
    _all_tickers = (cand_view["ticker"].tolist()
                    if cand_view is not None and len(cand_view) else [])
    # A controlled widget: the selected pills are the watchlist. It MUST be re-seeded from
    # the list every run, so changes made elsewhere (⭐, 📌, upload, the refresh job's
    # merge) show. The on_change syncs picks and dismissals back.
    st.session_state["wl_picker"] = _watch_t
    st.multiselect(
        "Watchlist tickers", options=sorted(set(_all_tickers) | set(_watch_t)),
        key="wl_picker", on_change=_wl_sync_from_picker,
        placeholder="Pick tickers to add…",
        help="Pick to add; click a pill's × to remove — removing forgets the frozen 📌 "
             "pivot (a re-add auto-freezes at the CURRENT scan pivot, not the old level). "
             "The ⭐ button next to any chart adds too. Saved automatically; persists "
             "between sessions; the downloads below give you a portable copy.")
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
        # TICKER 34.12 = frozen judged pivot · (a) = machine-frozen · \* = not frozen yet
        st.caption(" · ".join(
            (f"{e['ticker']} {e['judged_pivot']:.2f}"
             + (" (a)" if e.get("pivot_source") == "auto" else ""))
            if e.get("judged_pivot") else f"{e['ticker']}\\*"
            for e in _watch))
        if any(not e.get("judged_pivot") or e.get("pivot_source") == "auto" for e in _watch):
            st.caption("\\* no frozen pivot yet — the nightly EOD check freezes one on "
                       "first sight · (a) = auto-frozen; chart it and 📌 to judge your own.")
        _d1, _d2 = st.columns(2)
        _d1.download_button(
            "⬇ List (CSV)",
            watchlist_list_csv(res.candidates, _watch, DISPLAY_ORDER),
            file_name="watchlist.csv", mime="text/csv", width="stretch",
            help="Your shortlist with its decision columns (tier, pivot, stop, target, …) "
                 "plus the frozen judged_pivot/date/source, in the order you added them.")
        _d2.download_button(
            "⬇ OHLCV (CSV)", watchlist_ohlcv_csv(_watch_t, res.payloads),
            file_name="watchlist_ohlcv.csv", mime="text/csv", width="stretch",
            help="Daily Open/High/Low/Close/Volume for every watchlisted name, stacked "
                 "long-format with a Ticker column.")
        st.download_button(
            "⬇ Names (.txt)", ",".join(_watch_t), file_name="watchlist.txt",
            mime="text/plain", width="stretch",
            help="Just the tickers, comma-separated — the format the uploader above reads "
                 "back in.")
        _missing = [t for t in _watch_t if t not in res.payloads]
        if _missing:
            st.caption(f"⚠︎ {', '.join(_missing)} not in the current scan — the list CSV "
                       "keeps the ticker only and the OHLCV omits it. Re-scan the universe "
                       "that has them to include their data.")

        # --- Paper-trade the watchlist via Alpaca (paper account only) --------------- #
        st.markdown("---")
        st.markdown("**⚡ Paper trade (Alpaca)**")
        # The regime warning is repeated at the point of action; the banner is at the top.
        if not res.regime.get("should_generate_buys"):
            st.caption(":orange[**⚠︎ CAUTION tape** — the market regime advises against "
                       "NEW buys (most breakouts fail in a weak tape). Managing stops is "
                       "fine; think twice before submitting fresh entries.]")
        _weak = advisories.weak_market_advice(res.regime, stop_pct=DEFAULT_STOP_FROM_PIVOT,
                                              target_pct=0.25)
        if _weak:
            st.caption(f":orange[{_weak}]")
        _stk = res.regime.get("spy_ok_streak")
        if _stk is not None and not res.regime.get("spy_ok_satisfied"):
            st.caption(f":orange[SPY has been in Stage 1–2 for only **{_stk}/"
                       f"{REGIME_CONFIRM_DAYS}** sessions — the backtest waited "
                       f"{REGIME_CONFIRM_DAYS} before adding again after a break (the one "
                       "market-timing rule that held out of sample). Don't add yet.]")
        _mode_label = st.selectbox(
            "Size each buy by", ["% of portfolio", "$ per name", "# shares", "Risk % to stop"],
            key="trade_mode", on_change=_invalidate_trade_plan,
            help="Applied to EACH watchlisted name. '% of portfolio' = that % of your Alpaca "
                 "equity per name; '$ per name' = that many dollars each; '# shares' = exactly "
                 "that many shares each; 'Risk % to stop' = size so a stop-out costs that % of "
                 "equity — shares = (equity × risk%) / (price − stop), Minervini's position "
                 "sizer. Needs a stop on the name.")
        if _mode_label == "% of portfolio":
            _mode = "pct"
            _amount = st.number_input("% of equity per name", min_value=0.0, value=5.0,
                                      step=0.5, key="trade_amt_pct",
                                      on_change=_invalidate_trade_plan)
            _size_note = f"{_amount:.1f}% of equity per name"
        elif _mode_label == "$ per name":
            _mode = "dollars"
            _amount = st.number_input("$ per name", min_value=0.0, value=5000.0,
                                      step=500.0, key="trade_amt_dol",
                                      on_change=_invalidate_trade_plan)
            _size_note = f"~${_amount:,.0f} per name"
        elif _mode_label == "# shares":
            _mode = "shares"
            _amount = float(st.number_input("Shares per name", min_value=0, value=100,
                                            step=10, key="trade_amt_sh",
                                            on_change=_invalidate_trade_plan))
            _size_note = f"{int(_amount)} shares per name"
        else:                                    # Risk % to stop (Minervini position sizer)
            _mode = "risk"
            # Seeded if absent, not via value=: the "Use suggested" callback writes this key,
            # and a widget with both a default and session state warns.
            if "trade_amt_risk" not in st.session_state:
                st.session_state["trade_amt_risk"] = 1.0
            _amount = st.number_input("Risk % of equity per trade", min_value=0.0,
                                      step=0.25, key="trade_amt_risk",
                                      on_change=_invalidate_trade_plan,
                                      help="A stop-out costs about this % of equity. Note a "
                                           "risk-sized position is roughly risk% ÷ stop-distance% "
                                           "of equity — e.g. 1% risk with an 8% stop wants a "
                                           "12.5% position, which the 10% single-order cap "
                                           "clamps (realized risk then falls below target).")
            _size_note = f"{_amount:.2f}% of equity risked to each stop"
            # Recent-form guidance, in risk mode only: the journal read is paid only here.
            _guide = _risk_guidance()
            if _guide is None:
                st.caption("journal unavailable — no sizing guidance")
            else:
                st.caption(f"📒 {_guide['reason']}")
                if abs(_guide["risk_pct"] - float(_amount)) > 1e-9:
                    st.button(f"Use suggested {_guide['risk_pct']:.2f}%",
                              key="risk_apply", on_click=_apply_risk_suggestion,
                              args=(_guide["risk_pct"],))
        _ot_label = st.radio(
            "Order type", ["Market", "Limit (no-chase cap)"], horizontal=True,
            key="trade_order_type", on_change=_invalidate_trade_plan,
            help="**Market** fills at the next print, whatever it is — a gap can fill you "
                 "past the 5% buy zone. **Limit** caps each buy at a max price, defaulting "
                 "to its buy-zone top (pivot × 1.05, frozen 📌 pivot preferred): it never "
                 "fills ABOVE the limit, so a gap past the zone can't fill you high. It "
                 "CAN still fill below the zone — a below-pivot name fills immediately at "
                 "~the market (watch the per-row ⚠ warnings) — and a RESTING GTC limit "
                 "fills on any later pullback to it, even a failed breakout weeks on: "
                 "cancel it (🗑 button below) if the setup breaks. Sizing, risk, and the "
                 "10% cap use the worst-case fill. With a stop attached the order is GTC "
                 "end-to-end and its stop arms whenever the fill happens; without "
                 "a stop it's a DAY order that expires at the close.")
        _order_type = "limit" if _ot_label.startswith("Limit") else "market"
        st.caption(f"{'Limit' if _order_type == 'limit' else 'Market'} BUYs: {_size_note}. "
                   "Paper account only; whole shares, each order still capped at 10% of "
                   "equity.")
        if st.button("Build trade plan", key="trade_build", width="stretch"):
            # Fetched once per Build, not per rerun: it names the account for confirmation
            # and gives the equity for sizing.
            try:
                _account = fetch_account_summary()
            except TradeUnavailable as _e:
                _account = {"error": str(_e)}
            # Best-effort. The plan and its preview treat held names as stop re-arms. No
            # credentials means no held names.
            try:
                _held = fetch_held_shares()
            except TradeUnavailable:
                _held = {}
            # The exposure gate is computed at Build, which already calls the broker, never
            # at render. Unknown leaves this manual path open for the user to judge; the
            # unattended morning executor fails closed.
            try:
                _gi = fetch_gate_inputs()
                _gate = gate_status(_gi["positions"], _gi["open_episodes"],
                                    _gi["closed_episodes"])
                # Same journal read as the gate: no extra Alpaca call.
                _derived = trade.derived_stop_pct(_gi["closed_episodes"])
            except Exception:
                _gate = {"open": None,
                         "reason": "gate unknown (account/journal unreachable) — "
                                   "your judgment",
                         "probe_size_factor": 1.0, "consecutive_losses": 0}
                _derived = {"stop_pct": None,
                            "reason": "journal unreachable — default stop"}
            # Sizing and stops use freshly pulled bars, not the scan memo's older closes. The
            # staleness guard skips a name the refresh couldn't freshen. A frozen
            # judged_pivot overrides the drifting scan pivot for the buy zone, stop, extended
            # flag and sizing.
            _pivots = {e["ticker"]: e["judged_pivot"] for e in _watch
                       if isinstance(e, dict) and e.get("ticker") and e.get("judged_pivot")}
            with st.spinner("Refreshing prices & building plan…"):
                _fresh_payloads = freshen_prices(_watch_t, res.payloads)
                _plan, _skip = build_buy_plan(
                    _watch_t, _fresh_payloads, mode=_mode, amount=_amount,
                    equity=_account.get("equity"), max_bar_age_days=STALE_PLAN_BARS,
                    pivots=_pivots, held=_held, order_type=_order_type,
                    stop_pct=_derived.get("stop_pct"))
            # The build counter is the nonce in the per-row widget keys, so each Build
            # re-seeds the checkboxes, limits and stops to their computed defaults.
            _bn = st.session_state.get("trade_build_n", 0) + 1
            st.session_state["trade_build_n"] = _bn
            st.session_state["trade_plan"] = {"plan": _plan, "skipped": _skip,
                                              "account": _account, "held": _held,
                                              "build_ts": _bn, "order_type": _order_type,
                                              "gate": _gate, "derived": _derived}
            st.session_state.pop("trade_result", None)

        _tp = st.session_state.get("trade_plan")
        if _tp:
            _plan, _skip = _tp["plan"], _tp["skipped"]
            # The account error MUST render outside `if _plan:`. A build with no credentials
            # yields an empty plan, which alone shows only "No tradable orders".
            _account = _tp.get("account") or {}
            if _account.get("error"):
                st.warning(_account["error"])
            # The gate verdict from Build. A plan without the key shows nothing and blocks
            # nothing. Quiet when open at full size, red when closed, a warning at half size
            # or when unknown.
            _gate = _tp.get("gate") or {}
            _gate_closed = _gate.get("open") is False
            if _gate_closed:
                st.caption(f":red[**Exposure gate closed** — {_gate.get('reason')}. "
                           "New buys are blocked; stop re-arms and sells still work.]")
            elif _gate.get("open") is None and _gate.get("reason"):
                st.caption(f":orange[⚠︎ {_gate.get('reason')}]")
            elif _gate.get("probe_size_factor", 1.0) < 1.0:
                st.caption(f":orange[⚠︎ Exposure gate open — {_gate.get('reason')}]")
            _dv = _tp.get("derived") or {}
            if _dv.get("reason"):
                st.caption(f"🛑 Stops: {_dv['reason']}. Every buy's stop is at most "
                           f"{MAX_LOSS_FROM_FILL * 100:.0f}% below its fill.")
            if _plan:
                # Submit sends no buy for a held name, only a stop re-arm, so the est. total
                # counts buyable names only.
                _held = _tp.get("held") or {}
                _nonce = _tp.get("build_ts")

                # A checkbox per buy row picks what submits. Earnings-flagged names start
                # unchecked. Keys carry the build nonce, so a fresh Build re-seeds the
                # defaults.
                def _buy_key(t):
                    return f"buy_{t}_{_nonce}"

                def _buy_default(o):
                    return not _earnings_flag(o.get("earnings_in"))

                _buyable = [o for o in _plan if _held.get(o["ticker"], 0) <= 0]
                _buys = [o for o in _buyable
                         if st.session_state.get(_buy_key(o["ticker"]), _buy_default(o))]
                _tot = sum(o["est_value"] for o in _buys)
                _cap = f"**{len(_buys)}/{len(_buyable)} buy(s) selected** · ~${_tot:,.0f} est."
                if len(_buyable) != len(_plan):
                    _cap += f" · {len(_plan) - len(_buyable)} already held (no buy)"
                st.caption(_cap)
                # With the toggle off, buys go in with no stop and held names are skipped.
                _attach = st.toggle(
                    "Attach protective stop (sell-all, GTC)", value=True,
                    key="trade_attach_stop",
                    help="Places a stop-loss under each buy via a GTC OTO order — the stop leg "
                         "waits for the fill, then rests as a persistent GTC stop that survives "
                         "the close. If a name is already held, no buy is sent — a GTC "
                         "stop protects the whole position and, per Minervini, only ever "
                         "RATCHETS UP: a re-arm that would lower the stop is ignored and the "
                         "existing higher stop kept (shown as 'stop_kept' 🔒). Edit each stop "
                         "below (defaults to the app-computed stop).")
                _eq = (_tp.get("account") or {}).get("equity")
                _is_lim = _tp.get("order_type") == "limit"
                if _is_lim and any(_held.get(_o["ticker"], 0) <= 0 for _o in _plan):
                    st.caption("each buy row: ☑ name · **limit** · stop")
                for _o in _plan:
                    _t = _o["ticker"]
                    _held_sh = _held.get(_t, 0)
                    if _is_lim:
                        _cA, _cL, _cB = st.columns([2.6, 1.2, 1.2])
                    else:
                        _cA, _cB = st.columns([3, 2])
                        _cL = None
                    _on = True                       # held rows have no checkbox (re-arm only)
                    if _held_sh > 0:
                        _act = "stop re-arm only, no buy" if _attach else "skipped (attach off)"
                        _cA.caption(f"• **{_t}** — already held ({_held_sh} sh) · {_act}")
                    else:
                        _fl = " ⚠︎ extended" if _o["extended"] else ""
                        if _o.get("capped"):
                            _fl += " ⚠︎ capped"
                        if _o.get("pivot_frozen") and _o.get("pivot"):
                            _fl += f" · 📌 pivot {_o['pivot']:.2f}"   # stop/zone off frozen level
                        _ew = _earnings_flag(_o.get("earnings_in"))
                        _on = _cA.checkbox(
                            f"**{_t}** {_o['shares']} sh @ ~${_o['price']:.2f} "
                            f"(~${_o['est_value']:,.0f}){_fl}" + (f" · {_ew}" if _ew else ""),
                            value=_buy_default(_o), key=_buy_key(_t),
                            help="Unchecked names are left out of the submit entirely. "
                                 "Earnings-soon names start unchecked (no-fly window).")
                    # Limit plans only; a held row has no buy to limit.
                    _edlim = _o.get("limit_price")
                    if _cL is not None and _held_sh <= 0:
                        _cL.number_input(
                            f"limit {_t}", min_value=0.0,
                            value=float(_edlim) if _edlim else 0.0,
                            step=0.01, format="%.2f", key=f"lim_{_t}_{_nonce}",
                            label_visibility="collapsed", disabled=not _on,
                            help=f"Max fill price for {_t} — defaults to its buy-zone top "
                                 "(pivot × 1.05), the no-chase cap. The order fills at or "
                                 "below this, never above.")
                        _edlim = st.session_state.get(f"lim_{_t}_{_nonce}", _edlim)
                    _cB.number_input(
                        f"stop {_t}", min_value=0.0,
                        value=float(_o["stop_price"]) if _o["stop_price"] else 0.0,
                        step = 0.01, format="%.2f", key=f"stop_{_t}_{_nonce}",
                        label_visibility="collapsed", disabled=not _attach or not _on)
                    _edstop = st.session_state.get(f"stop_{_t}_{_nonce}", _o["stop_price"])
                    # A limit buy can fill anywhere at or below the limit, near the price when
                    # it is marketable. Validation and the risk read use min(limit, price).
                    _basis = (min(_edlim, _o["price"]) if (_is_lim and _edlim)
                              else _o["price"])
                    _paid = _edlim if (_is_lim and _edlim) else _o["price"]
                    if _held_sh > 0 or not _on:
                        pass          # held: stop is a re-arm target; unchecked: not submitted
                    elif _is_lim and (not _edlim or _edlim <= 0):
                        _cB.caption(":red[limit must be > 0]")
                    elif _attach and not stop_is_valid(_edstop, _basis):
                        _cB.caption(":red[stop must be below both limit and price]"
                                    if _is_lim else ":red[stop must be < price]")
                    elif _attach and not stop_within_max_loss(_edstop, _paid):
                        # Submit and arming refuse this row; say so before the click.
                        _cB.caption(f":red[> {MAX_LOSS_FROM_FILL * 100:.0f}% below the "
                                    f"{'limit' if _is_lim else 'price'} — raise to ≥ "
                                    f"{fill_floor(_paid):,.2f}]")
                    elif _attach and _eq and _edstop and _basis > _edstop:
                        # Risk from the edited stop: a stop edit doesn't re-size the shares.
                        _rusd = _o["shares"] * (_basis - _edstop)
                        _cA.caption(f"  ↳ risk to stop ≈ {_rusd / _eq * 100:.2f}% (${_rusd:,.0f})")
                    _rroom = (advisories.stop_room(_o["day_range_pct"] / 100.0, _edstop, _paid)
                              if _o.get("day_range_pct") and _on and _held_sh <= 0
                              and _attach else None)
                    if _rroom and _rroom["warn"]:
                        _cA.caption("  ↳ " + advisories.stop_room_text(
                            _rroom, _o["day_range_pct"] / 100.0))
                    if _o.get("stop_derived") and _on and _held_sh <= 0 and _attach:
                        _cA.caption(f"  ↳ derived stop {_o['stop_price']:,.2f} — "
                                    f"{(_dv.get('stop_pct') or 0) * 100:.1f}% below the "
                                    f"{'limit' if _is_lim else 'price'}")
                    elif _o.get("stop_floored") and _on and _held_sh <= 0 and _attach:
                        _cA.caption(f"  ↳ default stop raised to {_o['stop_price']:,.2f} — "
                                    f"{MAX_LOSS_FROM_FILL * 100:.0f}% below the "
                                    f"{'limit' if _is_lim else 'price'}, the most a fill "
                                    "up there may lose")
                    if (_is_lim and _on and _held_sh <= 0 and _edlim
                            and _edlim < _o["price"]):
                        _cA.caption(f"  ↳ limit {_edlim:,.2f} < last close {_o['price']:,.2f} "
                                    "— fills only on a pullback into the zone")
                    elif (_is_lim and _on and _held_sh <= 0 and _o.get("pivot")
                            and _o["price"] < _o["pivot"]):
                        _cA.caption(f"  ↳ ⚠ price {_o['price']:,.2f} is below the pivot "
                                    f"{_o['pivot']:,.2f} — this limit is marketable and "
                                    f"fills immediately at ~{_o['price']:,.2f}, below "
                                    "the buy zone")
                if any(_o["extended"] for _o in _buys):      # footnotes describe the BUYs only
                    st.caption("⚠︎ *extended* = >5% above the pivot; sized at pivot risk, so "
                               "the real risk to your stop is larger.")
                if any(_o.get("capped") for _o in _buys):
                    st.caption("⚠︎ *capped* = the risk-sized quantity hit the 10%-of-equity "
                               "order cap and was clamped down, so the realized risk sits below "
                               "your target. Lower the risk % or tighten the stop to fit.")
                if any(_earnings_flag(_o.get("earnings_in")) for _o in _buyable):
                    st.caption(f"⚠︎ *earnings in Nd* = a report is scheduled within "
                               f"~{EARNINGS_SOON_DAYS} days. A fresh buy has no profit "
                               "cushion to absorb an earnings gap, so these start "
                               "UNCHECKED — tick one to include it anyway.")
                # Name the account before submit: each paper account has its own keys. The
                # error case renders above.
                if not _account.get("error"):
                    _src = ("Minervini Trader keys" if _account.get("using_dedicated")
                            else "shared ALPACA_* keys — set ALPACA_API_KEY_MINERVINI / "
                                 "ALPACA_API_KEY_SECRET_MINERVINI to target the Minervini account")
                    st.caption(f"Target account **…{str(_account['account_number'])[-4:]}** "
                               f"({_src}) · equity ${_account['equity']:,.0f}")
                _c1, _c2, _c3 = st.columns([2, 2, 1])
                _n_held = sum(1 for _o in _plan if _held.get(_o["ticker"], 0) > 0)
                # A closed gate stamps buys gate_blocked, and submit skips them. The button
                # stays live for held rows: risk-reducing actions MUST NOT be gated.
                if _c1.button("✅ Submit (paper)", key="trade_submit",
                              type="primary", width="stretch",
                              disabled=(not _buys and not _n_held)
                              or (_gate_closed and not _n_held)):
                    # Only checked buy rows are sent; held rows always go, for their stop
                    # re-arm. A row held at Build is stamped rearm_only. If its position
                    # closes before Submit, submit MUST skip it: the preview promised no buy.
                    _final = [{**_o,
                               "rearm_only": _held.get(_o["ticker"], 0) > 0,
                               "gate_blocked": (_gate_closed
                                                and _held.get(_o["ticker"], 0) <= 0),
                               "stop_price": st.session_state.get(
                                   f"stop_{_o['ticker']}_{_nonce}", _o["stop_price"]),
                               "limit_price": (st.session_state.get(
                                   f"lim_{_o['ticker']}_{_nonce}", _o.get("limit_price"))
                                   if _is_lim and _held.get(_o["ticker"], 0) <= 0 else None)}
                              for _o in _plan
                              if _held.get(_o["ticker"], 0) > 0
                              or st.session_state.get(_buy_key(_o["ticker"]), _buy_default(_o))]
                    with st.spinner("Submitting to Alpaca paper…"):
                        try:
                            st.session_state["trade_result"] = submit_buy_plan(
                                _final, attach_stop=_attach)
                        except TradeUnavailable as _e:
                            st.session_state["trade_result"] = {"error": str(_e)}
                    st.session_state.pop("trade_plan", None)
                    st.rerun()
                # Arming buys nothing now. At ~9:26 ET the executor submits at most one row,
                # in list order, after re-checking the gate. So arming is allowed while the
                # gate reads closed: tonight's sell plan may free the book by the open. Limit
                # with a stop only: a market row would buy the open blind.
                if _c2.button(f"Arm {len(_buys)} for open", key="trade_arm",
                              width="stretch",
                              disabled=not _is_lim or not _attach or not _buys,
                              help="Writes tonight's armed entry plan (no orders now). "
                                   "The morning executor submits at most one still-"
                                   "armed row; disarm below anytime before the open. "
                                   "Needs the Limit order type + attached stop."):
                    _armed_rows = [{**_o,
                                    "stop_price": st.session_state.get(
                                        f"stop_{_o['ticker']}_{_nonce}",
                                        _o["stop_price"]),
                                    "limit_price": st.session_state.get(
                                        f"lim_{_o['ticker']}_{_nonce}",
                                        _o.get("limit_price"))}
                                   for _o in _plan
                                   if _held.get(_o["ticker"], 0) <= 0
                                   and st.session_state.get(_buy_key(_o["ticker"]),
                                                            _buy_default(_o))]
                    entries.save_entry_plan(entries.build_entry_plan(_armed_rows))
                    st.rerun()
                if _c3.button("Cancel", key="trade_cancel", width="stretch"):
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

    # --- Armed entries: tonight's plan and overnight disarm -------------------------- #
    # Outside the watchlist block: the morning-after check needs it with no plan built.
    _ep = None
    try:
        _ep = entries.load_latest_entry_plan()
    except Exception:
        _ep = None
    if _ep and _ep.get("rows"):
        st.markdown("**Armed for next open**")
        _armed_state = ("**ARMED** — executes ~9:26 ET"
                        if entries.autobuy_enabled()
                        else "not armed (`AUTOBUY` unset) — plan only")
        st.caption(f"Entry plan **{_ep.get('date')}** · morning submit {_armed_state}. "
                   "At most ONE still-armed row buys (walk order = list order); the "
                   "exposure gate is re-checked fresh at execution.")
        _EICON = {"armed": "🕘", "disarmed": "🚫", "submitted": "✅",
                  "failed": "⚠️", "skipped": "—"}
        for _r in _ep["rows"]:
            _ca, _cb = st.columns([5, 1])
            _lp = _r.get("limit_price")
            _sp = _r.get("stop_price")
            _lp_s = f"{_lp:.2f}" if _lp is not None else "?"
            _sp_s = f"{_sp:.2f}" if _sp is not None else "?"
            _ca.caption(f"{_EICON.get(_r.get('status'), '•')} **{_r['ticker']}** "
                        f"×{_r.get('shares')} · limit {_lp_s} · stop {_sp_s} — "
                        f"{_r.get('status')}"
                        + (f" · {_r['detail']}" if _r.get("detail") else ""))
            if _r.get("status") == entries.ROW_ARMED:
                _cb.button("Disarm", key=f"disarm_{_r['ticker']}_{_ep.get('date')}",
                           width="stretch", on_click=_do_disarm, args=(_r["ticker"],))

    # --- Cancel resting cockpit buys ------------------------------------------------ #
    # The way out of a GTC limit whose setup broke; it would fill on any later pullback.
    # Outside the watchlist block: a pending buy can outlive its watchlist entry.
    if st.button("🗑 Cancel pending cockpit buys", key="cancel_pending",
                 help="Cancels every OPEN cockpit BUY order (SEPA-tagged only — queued "
                      "market buys and resting GTC limits; an unfilled OTO stop leg dies "
                      "with its parent, nothing was bought). Never touches sells, "
                      "protective stops on held positions, or other tools' orders."):
        try:
            _cx = cancel_pending_buys()
            if _cx["cancelled"]:
                st.caption("cancelled: "
                           + ", ".join(c["ticker"] for c in _cx["cancelled"]))
            else:
                st.caption("no pending cockpit buys to cancel")
            for _e in _cx["errors"]:
                st.caption(f":orange[⚠ {_e['ticker']}: {_e['error']}]")
        except TradeUnavailable as _e:
            st.warning(str(_e))

    # --- Latest watchlist trigger check (written by cockpit-refresh.timer) ------------ #
    # A fragment: it re-reads the report once a minute and repaints only itself. The timer
    # ticks only while a browser is connected. Report fields MUST be read with .get(), so a
    # hand-edited or older-schema report degrades instead of crashing the sidebar.
    @st.fragment(run_every="60s")
    def _trigger_report_panel() -> None:
        st.markdown("---")
        # cockpit-refresh's pipeline, in-process, for when the timer missed a run. It MUST
        # run before the report load below, so the new report renders in this pass.
        if st.button("🔔 Check triggers now", key="trigger_check_now",
                     help="Run the watchlist trigger check immediately — tops up the "
                          "watchlist names' daily bars, freezes any missing pivots, and "
                          "writes today's report (same as the scheduled half-hourly run)."):
            with st.spinner("Checking watchlist triggers…"):
                try:
                    from src.stock_screener.cockpit.refresh_job import build_report
                    save_trigger_report(build_report())
                except Exception as _e:      # network/data failure — panel stays alive
                    st.warning(f"Trigger check failed: {_e}")
        _rep = load_latest_trigger_report(cache.TRIGGERS_DIR)
        if not _rep:
            st.caption("No trigger report yet — click 🔔 Check triggers now, or schedule "
                       "cockpit-refresh.timer (HANDOFF §8) for scheduled "
                       "watchlist trigger checks.")
            return
        _hm = str(_rep.get("generated_at", ""))[11:16]   # ISO -> HH:MM, best-effort
        st.markdown(f"**🔔 Trigger check — {_rep.get('date', '?')}"
                    + (f" {_hm}" if _hm else "") + "**")
        if _rep.get("early_close"):
            st.caption("🕐 early close (1pm ET) — half-day session; the volume gate is "
                       "scaled ×1.86 for the short session (thin holiday tape — judge "
                       "any trigger accordingly).")
        if _rep.get("intraday"):
            _settle = "~1pm" if _rep.get("early_close") else "~4pm"
            st.caption(f"⏱ intraday — close/volume provisional until {_settle}; pace = "
                       "volume so far vs expected by this time of day.")
        if _rep.get("all_stale"):
            st.caption("💤 No new bar on the report date (weekend/holiday?) — "
                       "no trigger can fire from a stale bar.")
        _ticons = {"triggered": "🔔", "pullback": "↩", "crossed": "↗", "extended": "⬆",
                   "watch": "👀", "stale": "💤", "no_pivot": "⚠", "no_data": "⚠",
                   "untracked": "🚫"}
        for _n in _rep.get("names", []):
            _st = _n.get("status", "?")
            _t = _n.get("ticker", "?")
            _pv, _cl = _n.get("judged_pivot"), _n.get("close")
            _vr = _n.get("volume_ratio_50")
            _bits = [f"{_ticons.get(_st, '•')} **{_t}** {_st}"]
            if _cl is not None and _pv:
                _bits.append(f"{_cl:,.2f} vs pivot {_pv:,.2f}"
                             + (" (a)" if _n.get("pivot_source") == "auto" else ""))
            if _vr is not None:
                _bits.append(f"vol {_vr:.1f}×")
            _pc = _n.get("volume_pace")
            if _pc is not None and _rep.get("intraday"):
                _bits.append(f"pace {_pc:.1f}×")
            if _n.get("earnings_soon"):
                _bits.append(f"⚠ earnings in {_n.get('earnings_in')}d")
            _cA, _cB = st.columns([6, 1], vertical_alignment="center")
            _cA.caption(" · ".join(_bits))
            _in_scan = _t in res.payloads
            if _cB.button("📈", key=f"trg_chart_{_t}", disabled=not _in_scan,
                          help=(f"Chart {_t}" if _in_scan
                                else f"{_t} is not in the scan table — no chart data")):
                # Jumps the main chart to this name. A fragment needs an app-scope rerun.
                # AppTest runs fragments inline, where that raises; a plain rerun is the
                # fallback.
                st.session_state["chart_pick"] = _t
                try:
                    st.rerun(scope="app")
                except StreamlitAPIException:
                    st.rerun()
        if _rep.get("summary", {}).get("crossed"):
            st.caption("↗ crossed = above its frozen pivot but NOT volume-confirmed — a "
                       "quiet drift is not a buy; wait for a ≥1.5× volume close, or plan "
                       "a pullback/secondary entry off the pivot.")
        if _rep.get("summary", {}).get("pullback"):
            st.caption("↩ pullback = crossed its frozen pivot earlier, now back within "
                       "~2% of it on dry volume (≤0.8×) — the low-risk secondary entry; "
                       "judge the chart, stop goes just below the pivot.")
        if _rep.get("summary", {}).get("untracked"):
            st.caption("🚫 untracked = fell out of the 8/8 trend template — kept on the "
                       "watchlist, but the trigger is not evaluated until it "
                       "re-qualifies.")
        if _rep.get("summary", {}).get("triggered"):
            st.caption("🔔 Triggered = closed above the frozen pivot on ≥1.5× 50-day "
                       "volume — judge it (chart + fuel), then buy at/near the next "
                       "open if it holds up.")

    _trigger_report_panel()

# --------------------------------------------------------------------------- #
# Regime banner
# --------------------------------------------------------------------------- #
reg = res.regime
buy_ok = reg.get("should_generate_buys")
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
           f"{min_criteria}/8 trend template · {len(cand_view)} after filters"
           + (f" · {len(res.errors)} errors" if res.errors else ""))

# --------------------------------------------------------------------------- #
# Candidate table + per-name detail
# --------------------------------------------------------------------------- #
cand = cand_view
if cand is None or len(cand) == 0:
    st.info("No candidates passed the current filters. "
            "Loosen the RS / fundamental filters, or wait for a better tape.")
    st.stop()

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

# A container renders where it is created, not where it is written. This one holds the
# table's slot above the filter, so the filter runs first and applies on the same run.
table_box = st.container()
view = filter_table(view)

if len(view) == 0:
    st.info("No candidates match the current filters. Clear a filter to see more.")
    st.stop()

table_box.caption(f"Showing {len(view)} of {len(cand)} — click a row to chart it.")
# Labels and tooltips are display only; the frame keeps raw names for selection and
# filtering. column_order hides any column not listed.
with table_box:
    col_config = {c: st.column_config.Column(READABLE_COLS.get(c, c), help=COL_HELP.get(c))
                  for c in view.columns}
    # A scan persisted before a column existed MAY lack it.
    event = st.dataframe(view, width="stretch", hide_index=True, height=380,
                         column_config=col_config,
                         column_order=[c for c in DISPLAY_ORDER if c in view.columns],
                         on_select="rerun", selection_mode="single-row", key="cand_table")

# selection.rows are positional indices into the displayed (filtered) frame
_sel = getattr(event, "selection", None)
_rows = (_sel.get("rows", []) if isinstance(_sel, dict)
         else getattr(_sel, "rows", [])) if _sel else []
row_pos = _rows[0] if _rows and _rows[0] < len(view) else 0
pick = view.iloc[row_pos]["ticker"]

# A 📈 jump from the trigger panel overrides the table selection for this run only; the
# key is popped. The button is disabled outside the scan, so a miss is a stale key.
_jump = st.session_state.pop("chart_pick", None)
if _jump in res.payloads:
    pick = _jump

payload = res.payloads[pick]

# The chart on the left; Step 2 and the Step-3 controls stacked on the right. The side
# column MUST be written first, so its control values exist before the chart builds.
colChart, colSide = st.columns([3, 1])

with colSide:
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
            # From data_feed._edgar_backfill: yfinance lacks FY growth and 3-quarter
            # acceleration.
            _fy, _acc, _sp = (f.get("eps_fy_yoy"), f.get("eps_accel_3q"),
                              f.get("last_surprise_pct"))
            _extra = []
            if _fy is not None:
                _extra.append(f"**FY EPS:** {_fy:+.1f}%")
            if _acc is not None:
                _extra.append("3q accel ✅" if _acc else "3q accel —")
            if _sp is not None:
                _extra.append(f"surprise {_sp:+.1f}%")
            if _extra:
                st.markdown(" · ".join(_extra))
            checks = s2.get("checks", {})
            st.markdown(" ".join(
                f"{'✅' if checks.get(k) else '—'} {lbl}"
                for k, lbl in [("revenue_growth", "Rev ≥20%"), ("eps_growth", "EPS ≥20%"),
                               ("eps_accelerating", "EPS accel"),
                               ("margin_expanding", "Margin ↑")]))
            st.caption(f"Score {s2.get('score', 0)}/4")

    # Step-3 controls; the chart renders in colChart.
    with st.container(border=True):
        st.markdown(f"### {pick}")
        # ⭐ adds the name and freezes the charted pivot. 📌 re-freezes an existing entry:
        # picker-added, auto-frozen, or drifted from the chart.
        _app_pivot = payload.get("levels", {}).get("pivot")
        if pick in _wl_tickers():
            st.button(f"✓ In watchlist — remove {pick}", key="wl_toggle",
                      on_click=_wl_remove, args=(pick,), width="stretch")
            _ent = _wl_entry(pick)
            _frozen = _ent.get("judged_pivot") if _ent else None
            if _app_pivot and (_frozen is None or abs(_frozen - _app_pivot) >= 0.005
                               or (_ent or {}).get("pivot_source") == "auto"):
                st.button(f"📌 Freeze pivot @ {_app_pivot:,.2f}", key="wl_freeze",
                          on_click=_wl_freeze, args=(pick, _app_pivot), width="stretch",
                          help="Locks THIS level as the nightly trigger pivot for this name "
                               "(the detected pivot drifts as new bars arrive; your judged "
                               "level overrides an auto-frozen one). Freeze again anytime "
                               "to update it."
                          + (f" Currently frozen @ {_frozen:,.2f}"
                             f" ({(_ent or {}).get('pivot_source') or '?'})."
                             if _frozen else ""))
            elif _frozen:
                st.caption(f"📌 pivot frozen @ {_frozen:,.2f} "
                           f"({(_ent or {}).get('date_added') or '—'}, judged)")
        else:
            st.button(f"⭐ Add {pick} to watchlist", key="wl_toggle", type="primary",
                      on_click=_wl_add, args=(pick,),
                      kwargs={"judged_pivot": _app_pivot}, width="stretch",
                      help="Adds the name AND freezes the current pivot as your judged "
                           "trigger level (shown in Step 4).")
        # A pivot the price is already above arms a trigger whose crossing may be behind
        # it. Warn at freeze time, for ⭐ and 📌 alike.
        _wl_df = payload.get("df")
        _wl_close = (float(_wl_df["Close"].iloc[-1])
                     if _wl_df is not None and len(_wl_df) else None)
        if _app_pivot and _wl_close is not None and _wl_close > _app_pivot:
            st.caption(f"⚠︎ post-breakout: {pick} already trades above this pivot "
                       f"({_wl_close:,.2f} > {_app_pivot:,.2f}) — a freeze here arms a "
                       "trigger that may never re-fire; plan a pullback/secondary entry "
                       "instead (its report row will show ↗ crossed).")
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
    # This value MUST hold at most one '$': two in one st.metric value parse as a LaTeX
    # span and render in a serif font.
    buy_zone = ("n/a" if (bz_lo is None or bz_hi is None)
                else f"${bz_lo:,.2f} – {bz_hi:,.2f}")
    pct = lv.get("pct_to_pivot")
    pct_s = "n/a" if pct is None else f"{pct:+.1f}%"
    vol = lv.get("volume_ratio", 1)
    price_ok = bool(lv.get("breakout_today"))       # price cleared the pivot
    vol_ok = bool(lv.get("volume_confirmed"))        # latest volume >= 1.5x the 20-day avg

    # The level tiles are neutral: they say where to act, not whether. The pivot is the
    # detected one, recomputed every scan. Triggers and trade plans use the frozen 📌
    # level; the tooltip says so.
    _fz_pivot = (_wl_entry(pick) or {}).get("judged_pivot")
    _pivot_help = ("The scan's **detected** pivot — recomputed from the price history on "
                   "every scan, so it drifts as new bars arrive. The buy zone, stop, and "
                   "target tiles derive from it. "
                   + (f"Your **frozen 📌 pivot** for {pick} is **${_fz_pivot:,.2f}** (the "
                      "sidebar's level) — triggers fire and trade-plan orders are priced "
                      "off that frozen level, not this one. If you re-judge the base, "
                      "re-📌 on the chart to update it."
                      if _fz_pivot else
                      "⭐/📌 freezes the level you're judging as the watchlist trigger "
                      "pivot; from then on the frozen level (sidebar), not this drifting "
                      "one, is what triggers and trade plans act on."))
    r1 = st.columns(3)
    r1[0].metric("Pivot", _usd(lv.get("pivot")), border=True, help=_pivot_help)
    r1[1].metric("Buy zone", buy_zone, border=True)
    r1[2].metric("Stop", _usd(lv.get("stop")), border=True)
    r2 = st.columns(2)
    r2[0].metric("Target", _usd(lv.get("target")), border=True)
    r2[1].metric("To pivot", pct_s, border=True)
    spp = lv.get("stop_pct_from_pivot")
    if spp is not None:
        _clamp = (" — capped at the 10% max (logical support sat lower; a base needing a wider "
                  "stop is too loose to risk more than 10%)" if lv.get("stop_clamped") else "")
        _mark = "✅" if spp <= 8.0 + 1e-9 else "⚠️"
        st.caption(f"{_mark} Risk pivot → stop: **{spp:.1f}%** "
                   f"(Minervini: 7–8% ideal, 10% hard max){_clamp}")
    _dr = (payload.get("vcp") or {}).get("median_tr_pct")
    _room = advisories.stop_room((_dr or 0) / 100.0, lv.get("stop"), lv.get("pivot"))
    if _room:
        st.caption(("⚠️ " if _room["warn"] else "") + f"Typical day **{_dr:.1f}%** — the stop "
                   f"is **{_room['room_days']:.1f}** ordinary days below the pivot"
                   + (": inside normal noise, so an ordinary day can shake you out. Wider "
                      "stop and a smaller position, or pass." if _room["warn"] else "."))
    _mfs = lv.get("max_fill_for_stop")
    if _mfs and bz_hi and _mfs < bz_hi:
        # No '$' here: two in one markdown string parse as a LaTeX span.
        st.caption(f"Fills above **{_mfs:,.2f}** raise the stop — the max loss is "
                   f"{MAX_LOSS_FROM_FILL * 100:.0f}% below the price you pay, and the zone "
                   f"runs to {bz_hi:,.2f}.")

    # Two states in sequence on the same two axes, volume and volatility. The base is
    # quiet: both contract. The breakout is loud: both expand.
    vcp_data = payload.get("vcp", {})
    st.markdown("**The base — what you're waiting on (should be _quiet_):**")

    # RMV, then BBWP as a cross-check. Both are point-in-time, so they rise as a breakout
    # fires.
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

    _rs_nh = payload.get("rs_nh")
    if _rs_nh is None:
        st.markdown("**RS line:** n/a — under ~6 months of overlapping SPY history.")
    elif _rs_nh:
        st.markdown("**RS line:** ✅ **at a 52-wk high before price** — outperforming the "
                    "market while still basing (accumulation tell; IBD blue dot).")
    else:
        st.markdown("**RS line:** — not at a new high before price (no divergence signal; "
                    "fine, just no extra confirmation).")

    # volume_quality: % of contractions whose volume ran lighter than the advance into
    # them. Its yardstick is the run-up, not the 1.5× average.
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
    st.markdown(f"- **Price:** {'✅ above pivot' if price_ok else '— below the pivot (no trigger yet)'}")
    vol_txt = f"{vol:.1f}× the 20-day average" if isinstance(vol, (int, float)) else "n/a"
    st.markdown(f"- **Volume:** {'✅' if vol_ok else '—'} {vol_txt} (a breakout needs ≥ 1.5×)")
    if bool(lv.get("squeeze_released")):
        volat_txt = "✅ squeeze released — volatility expanding out of the base"
    elif squeeze_on:
        volat_txt = "— still coiled (squeeze on) — no expansion yet"
    else:
        volat_txt = "— no active squeeze to release"
    st.markdown(f"- **Volatility:** {volat_txt}")
    st.markdown("---")
    # These widgets MUST keep explicit keys. Without one, identity depends on the variable
    # element count above, and a rerun could reset the value.
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
