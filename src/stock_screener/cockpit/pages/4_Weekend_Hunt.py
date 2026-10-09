"""Weekend Hunt page: start a hunt on the hunt PC and review its picks one at a time.

The hunt itself runs on the Windows PC (scripts/hunt/), which pushes each result folder
to ``data/cockpit/hunt/<date>/`` on the Pi. Start leaves a request file the PC's poller
picks up over ssh (``hunt_request``). This page reads those folders. For every
PASS name, bucket by bucket, it shows the chart, the verdict, the Step-2 fundamentals,
the entry numbers and the catalyst read, with an Add-to-watchlist button. Step 4 stays
with the user: nothing here arms or submits an order.

Run the app from the project root: ``streamlit run src/stock_screener/cockpit/app.py`` and
pick "Weekend Hunt" from the page nav.
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
import streamlit.components.v1 as components  # noqa: E402
from streamlit.errors import StreamlitAPIException  # noqa: E402

from src.stock_screener.cockpit import advisories  # noqa: E402 (chart marks)
from src.stock_screener.cockpit import hunt_view  # noqa: E402 (module import → patchable)
from src.stock_screener.cockpit import scan_worker  # noqa: E402
from src.stock_screener.cockpit.charts import build_chart  # noqa: E402
from src.stock_screener.cockpit.export import load_watchlist, watchlist_tickers  # noqa: E402
from src.stock_screener.cockpit import cache  # noqa: E402
from src.stock_screener.cockpit.panels import (render_step2_from_hunt_row,  # noqa: E402
                                               render_step2_panel, step_badge)

# Warm the universe scan in the background, so the charts here come from the latest
# scan. Inert under AppTest.
scan_worker.autostart()

st.set_page_config(page_title="Weekend Hunt", page_icon="🔭", layout="wide")
# Mirror app.py's padding trim so the first row isn't buried under Streamlit's header.
# Tighter gaps and slightly smaller body text than the other pages: this one is read a
# ticker at a time and MUST show the most per screen.
st.markdown(
    "<style>.block-container{padding-top:4rem;padding-bottom:1rem;}"
    'div[data-testid="stVerticalBlock"]{gap:0.4rem;}'
    'div[data-testid="stMarkdownContainer"] p,div[data-testid="stMarkdownContainer"] li'
    "{font-size:0.9rem;line-height:1.4;}</style>",
    unsafe_allow_html=True)

_VERDICT_COLOR = {"PASS": "green", "PASS-": "orange", "FAIL": "red"}
# The page MUST fit a ticker's chart, verdict and entry numbers on one laptop screen
# without scrolling; the scan page keeps the full 720 px chart.
CHART_HEIGHT = 500
SHEET_WIDTH = 560          # the hunt's own panel, for a name the latest scan dropped
REPORT_HEIGHT = 820        # the iframe scrolls itself; the page does not


def _rerun_app() -> None:
    """Rerun the whole app from inside a fragment; AppTest runs fragments inline and
    rejects the scope, so fall back to a plain rerun there."""
    try:
        st.rerun(scope="app")
    except StreamlitAPIException:
        st.rerun()


# ---- callbacks: a click MUST change state before the header renders ------------------ #
def _go(delta: int, n: int) -> None:
    st.session_state["hunt_idx"] = (int(st.session_state.get("hunt_idx", 0)) + delta) % n


def _start_cb() -> None:
    try:
        _, created = hunt_view.request_hunt()
    except OSError as e:
        st.session_state["hunt_msg"] = f"Could not write the request: {e}"
        return
    st.session_state["hunt_msg"] = ("Requested. The hunt PC picks it up within seconds "
                                    "when it is awake; asleep, it runs it when it next wakes."
                                    if created else "A request is already waiting for the PC.")


def _wl_add_cb(pick: hunt_view.Pick, hunt_date: str) -> None:
    merged, added = hunt_view.add_to_watchlist(
        st.session_state.get("watchlist"), pick.ticker, pick.row.get("pivot"),
        hunt_date, pick.verdict, pick.notes)
    if "watchlist" in st.session_state:
        st.session_state["watchlist"] = merged
    # A plan built for the old list is stale once the list changes (app.py does the same).
    st.session_state.pop("trade_plan", None)
    st.session_state.pop("trade_result", None)
    st.session_state["hunt_wl_msg"] = (f"{pick.ticker} added with the pivot frozen at "
                                       f"{float(pick.row.get('pivot') or 0):.2f}."
                                       if added else f"{pick.ticker} is already on the watchlist.")


# ---- header: one row — title, which hunt, Start ---------------------------------------- #
dirs = hunt_view.hunt_dirs()
date_names = [d.name for d in dirs]
progress = hunt_view.hunt_progress()
_active = progress["state"] in ("requested", "claimed", "running")
tcol, dcol, scol = st.columns([2, 2, 1.3], vertical_alignment="center")
tcol.markdown("### 🔭 Weekend Hunt",
              help="Every Tier-A base reviewed on the hunt PC; the PASS names here, one at a "
                   "time. Step 4 — the entry — is yours.")
if date_names:
    if st.session_state.get("hunt_date") not in date_names:
        st.session_state["hunt_date"] = date_names[0]
    hunt_date = dcol.selectbox(
        "Hunt", date_names, key="hunt_date", label_visibility="collapsed",
        format_func=lambda d: f"Hunt of {d}" + (" (latest)" if d == date_names[0] else ""))
scol.button("▶ Start weekend hunt", key="hunt_start", on_click=_start_cb,
            disabled=_active, width="stretch",
            help="The hunt PC watches for requests while awake; asleep, it runs the request "
                 "when it next wakes. The Friday 18:00 hunt is requested by the Pi itself.")
if st.session_state.get("hunt_msg"):
    st.info(st.session_state.pop("hunt_msg"))

_iv = "10s" if _active else None


@st.fragment(run_every=_iv)
def _hunt_status_line() -> None:
    """The request's progress, re-read every 10 s while one is pending. The first sight
    of a finished run reloads the page so its folder shows."""
    p = hunt_view.hunt_progress()
    state = p["state"]
    if state == "requested":
        st.info(f"Waiting for the hunt PC to pick up the request made "
                f"{p.get('requested_at') or '?'} (it must be awake).")
    elif state in ("claimed", "running"):
        st.info(f"Hunt running on the PC · {p.get('message') or 'starting'} …")
    elif state in ("done", "error") and st.session_state.get("hunt_seen") != p.get("updated_at"):
        st.session_state["hunt_seen"] = p.get("updated_at")
        if state == "done":
            st.success("The hunt finished and is on the Pi; loading it.")
        else:
            st.error(f"The hunt failed on the PC: {p.get('message') or 'see its log'}")
        _rerun_app()


_hunt_status_line()

if not dirs:
    st.info("No weekend hunt yet. Run one from the hunt PC, or press Start.")
    st.stop()

hunt_path = dirs[date_names.index(hunt_date)]
state, why = hunt_view.load_state_safe(hunt_path)
if state is None:
    st.error(f"This hunt folder could not be read: {why}")
    st.stop()

meta = state.meta
regime = (meta.get("regime") or {}).get("regime", "?")
st.caption(f"Scan of **{str(meta.get('scan_time', ''))[:10]}** · regime **{regime}** · "
           f"{meta.get('n_eligible', len(state.diag_rows))} reviewed · "
           f":green[PASS {state.n['PASS']}] · :orange[PASS- {state.n['PASS-']}] · "
           f":red[FAIL {state.n['FAIL']}]")

tab_review, tab_narr, tab_report = st.tabs(["Review", "Narrative", "Report"])

# ---- Review: one PASS name at a time ------------------------------------------------- #
with tab_review:
    picks = hunt_view.ordered_picks(state)
    if not picks:
        st.info("No PASS names in this hunt.")
    else:
        if st.session_state.get("hunt_idx_date") != hunt_date:
            st.session_state["hunt_idx"] = 0
            st.session_state["hunt_idx_date"] = hunt_date
        n = len(picks)
        idx = int(st.session_state.get("hunt_idx", 0)) % n
        pick = picks[idx]
        catalysts = hunt_view.load_catalysts(hunt_path)
        payloads = hunt_view.latest_payloads()
        payload = payloads.get(pick.ticker)
        nums = hunt_view.entry_numbers(pick.row)

        _fmt = lambda v, f="{:.2f}": "n/a" if v is None else f.format(v)  # noqa: E731
        wl_now = st.session_state.get("watchlist")
        on_list = pick.ticker in watchlist_tickers(
            wl_now if wl_now is not None else load_watchlist(cache.WATCHLIST_JSON))

        # One strip: where you are, the verdict, and every control, so nothing below the
        # chart has to be reached for.
        pcol, hcol2, wcol, ncol = st.columns([1, 6, 2.6, 1], vertical_alignment="center")
        pcol.button("◀ Prev", key="hunt_prev", on_click=_go, args=(-1, n), width="stretch")
        ncol.button("Next ▶", key="hunt_next", on_click=_go, args=(1, n), width="stretch")
        color = _VERDICT_COLOR.get(pick.verdict, "blue")
        vs = nums["vs_pivot_pct"]
        head = (f"#### {idx + 1} of {n} · {hunt_view.BUCKET_LABEL[pick.bucket]} · "
                f"**{pick.ticker}** :{color}-background[{pick.verdict}]"
                + (f" {vs:+.1f}% vs pivot" if vs is not None else ""))
        hcol2.markdown(head)
        if on_list:
            wcol.markdown(f"✓ {pick.ticker} is on the watchlist.")
        else:
            wcol.button(f"⭐ Add {pick.ticker} · pivot {_fmt(nums['pivot'])}",
                        key="hunt_wl_add", on_click=_wl_add_cb, args=(pick, hunt_date),
                        width="stretch")
        if pick.blocked:
            st.caption(f":orange[⚠︎ Earnings in {pick.earnings_in}d — the hunt bars an entry "
                       f"inside the window.]")
        if st.session_state.get("hunt_wl_msg"):
            st.toast(st.session_state.pop("hunt_wl_msg"), icon="⭐")

        # Row 1: the chart and the verdict with its entry numbers, side by side.
        colChart, colSide = st.columns([2.4, 1])
        with colChart:
            if payload is not None and payload.get("df") is not None:
                scan_lv = payload.get("levels")
                fig = build_chart(pick.ticker, payload["df"], vcp=payload.get("vcp"),
                                  levels=hunt_view.hunt_levels(pick.row, scan_lv),
                                  marks=advisories.step3_marks(payload))
                fig.update_layout(height=CHART_HEIGHT, margin=dict(l=10, r=24, t=30, b=6))
                drift = hunt_view.pivot_drift(pick.row, scan_lv)
                if drift:
                    fig.add_hline(y=drift[0], row=1, col=1, line_dash="dot", line_width=1,
                                  line_color="#9e9e9e", annotation_text="today's pivot",
                                  annotation_position="top left",
                                  annotation_font_color="#9e9e9e")
                st.plotly_chart(fig, width="stretch")
                if drift:
                    st.caption(f":orange[Since the hunt the scan re-detects the pivot at "
                               f"{drift[0]:.2f} ({drift[1]:+.1f}% vs the hunt's "
                               f"{_fmt(nums['pivot'])}; dotted line).] The levels drawn, the "
                               "numbers and ⭐ Add use the hunt's pivot; re-freeze it with 📌 "
                               "on the scan page if you judge the new one.")
                else:
                    st.caption("Today's bars and detected legs; the levels are the hunt's.")
            else:
                sheet = hunt_view.sheet_for(hunt_path, state, pick.ticker)
                if sheet:
                    png, k, total, j = sheet
                    panel = hunt_view.sheet_panel(png, j)
                    st.image(panel if panel is not None else str(png), width=SHEET_WIDTH)
                    st.caption(f"From hunt sheet {k} of {total} ({hunt_view.panel_name(j)}): "
                               f"{pick.ticker} is not in the latest Pi scan.")
                else:
                    st.info("No chart: the name is not in the latest scan and the hunt "
                            "left no sheet for it.")

        with colSide:
            with st.container(border=True):
                st.markdown(step_badge("Step 3", "The hunt's verdict"))
                if pick.notes:
                    st.markdown(pick.notes)
                if nums["depths"] or nums["step3"]:
                    st.caption(" · ".join(x for x in (f"legs {nums['depths']}" if nums["depths"]
                                                      else "", nums["step3"]) if x))
            with st.container(border=True):
                st.markdown(step_badge("Step 4", "Entry numbers at the hunt's close"))
                st.markdown(f"**Pivot** {_fmt(nums['pivot'])} · zone to "
                            f"{_fmt(nums['zone_hi'])} · **stop** {_fmt(nums['stop'])}")
                # Two bare dollar signs in one markdown string render as math.
                st.markdown(f"**ADV** {_fmt(nums['adv_musd'], r'\${:.1f}M')} · max order "
                            f"{_fmt(nums['max_order_usd'], r'\${:,.0f}')}")
                ern = nums["earnings_in"]
                st.markdown(f"**Earnings** {'n/a' if ern is None else f'in {ern}d'} · "
                            f"DD {_fmt(nums['dist_days'], '{:.0f}')} · RS "
                            f"{_fmt(nums['rs'], '{:.0f}')} · Q {_fmt(nums['q'], '{:.0f}')} · "
                            f"F {_fmt(nums['fund'], '{:.0f}')}/{int(nums['f_max'])}")

        # Row 2: Step 2 and the catalyst, side by side.
        colFund, colCat = st.columns(2)
        with colFund:
            if payload is not None:
                render_step2_panel(payload, help_popover=False)
            else:
                render_step2_from_hunt_row(pick.row, help_popover=False)
        # The catalyst: the books' third element, read from the hunt's headlines.
        with colCat, st.container(border=True):
            st.markdown(step_badge("Catalyst", "why would institutions want it now?"),
                        help="A label, never a gate: the catalyst does not change a verdict.")
            cat = catalysts.get(pick.ticker)
            if not cat:
                st.caption("Not read for this hunt.")
            else:
                sent = str(cat.get("sentiment") or "")
                scol_ = {"positive": "green", "negative": "red", "mixed": "orange"}.get(sent, "blue")
                st.markdown(f":blue-background[{cat.get('category') or '?'}] "
                            f":{scol_}-background[{sent or '?'}]  "
                            + str(cat.get("summary") or ""))
                srcs = [s for s in (cat.get("sources") or []) if isinstance(s, dict)]
                if srcs:
                    st.caption("  \n".join(
                        f"{s.get('date') or ''} · [{s.get('title') or s.get('url')}]"
                        f"({s.get('url') or '#'}) — {s.get('publisher') or ''}"
                        for s in srcs))
            st.caption("**Look further:** " + " · ".join(
                f"[{label}]({url})" for label, url in hunt_view.manual_links(pick.ticker)))

# ---- Narrative ---------------------------------------------------------------------- #
with tab_narr:
    narrative = hunt_view.read_text(hunt_path / "narrative.md")
    if narrative:
        st.markdown(narrative)
    else:
        st.info("This hunt left no narrative.")
    summary = hunt_view.read_text(hunt_path / "summary.md")
    if summary:
        with st.expander("The reviewer's closing summary"):
            st.markdown(summary)

# ---- Report ------------------------------------------------------------------------- #
with tab_report:
    html = hunt_view.read_text(hunt_path / "report.html")
    if not html:
        st.info("This hunt left no report.html.")
    else:
        embed = st.checkbox("Embed the review sheets (slow on a Pi)", key="hunt_embed_sheets",
                            value=False)
        components.html(hunt_view.inline_report_images(html, hunt_path, embed),
                        height=REPORT_HEIGHT, scrolling=True)
