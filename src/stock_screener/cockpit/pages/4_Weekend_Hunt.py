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
st.markdown(
    "<style>.block-container{padding-top:4rem;padding-bottom:2rem;}"
    'div[data-testid="stVerticalBlock"]{gap:0.6rem;}</style>',
    unsafe_allow_html=True)

_VERDICT_COLOR = {"PASS": "green", "PASS-": "orange", "FAIL": "red"}


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
    st.session_state["hunt_msg"] = ("Requested. The hunt PC picks it up within a minute "
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


# ---- header: which hunt, and the Start button ------------------------------------------ #
dirs = hunt_view.hunt_dirs()
hcol, scol = st.columns([3, 1])
with hcol:
    st.markdown("## 🔭 Weekend Hunt")
    st.caption("Every Tier-A base reviewed on the hunt PC; the PASS names below, one at a "
               "time. Step 4 — the entry — is yours.")
progress = hunt_view.hunt_progress()
_active = progress["state"] in ("requested", "claimed", "running")
with scol:
    st.button("▶ Start weekend hunt", key="hunt_start", on_click=_start_cb,
              disabled=_active, width="stretch")
    st.caption("The hunt PC checks for requests every 30 s while awake, and the Friday "
               "18:00 hunt is requested by the Pi itself.")
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

date_names = [d.name for d in dirs]
if st.session_state.get("hunt_date") not in date_names:
    st.session_state["hunt_date"] = date_names[0]
hunt_date = st.selectbox("Hunt", date_names, key="hunt_date",
                         format_func=lambda d: f"{d}" + (" (latest)" if d == date_names[0] else ""))
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

        pcol, hcol2, ncol = st.columns([1, 4, 1])
        pcol.button("◀ Prev", key="hunt_prev", on_click=_go, args=(-1, n),
                    width="stretch")
        ncol.button("Next ▶", key="hunt_next", on_click=_go, args=(1, n),
                    width="stretch")
        head = f"### {idx + 1} of {n} · {hunt_view.BUCKET_LABEL[pick.bucket]} · **{pick.ticker}**"
        hcol2.markdown(head)
        if pick.blocked:
            hcol2.warning(f"⚠︎ Earnings in {pick.earnings_in}d — the hunt bars an entry "
                          f"inside the window.")
        if st.session_state.get("hunt_wl_msg"):
            st.success(st.session_state.pop("hunt_wl_msg"))

        colChart, colSide = st.columns([3, 1])
        with colChart:
            if payload is not None and payload.get("df") is not None:
                fig = build_chart(pick.ticker, payload["df"], vcp=payload.get("vcp"),
                                  levels=payload.get("levels"),
                                  marks=advisories.step3_marks(payload))
                st.plotly_chart(fig, width="stretch")
                st.caption("Latest scan on the Pi, with its detected pivot and legs.")
            else:
                sheet = hunt_view.sheet_for(hunt_path, state, pick.ticker)
                if sheet:
                    png, k, total, j = sheet
                    st.image(str(png), width="stretch")
                    st.caption(f"Hunt sheet {k} of {total}: {pick.ticker} is the "
                               f"{hunt_view.panel_name(j)} panel. The name is not in the "
                               "latest Pi scan, so this is the hunt's own chart.")
                else:
                    st.info("No chart: the name is not in the latest scan and the hunt "
                            "left no sheet for it.")

        with colSide:
            with st.container(border=True):
                st.markdown(step_badge("Step 3", "The hunt's verdict"))
                color = _VERDICT_COLOR.get(pick.verdict, "blue")
                vs = nums["vs_pivot_pct"]
                st.markdown(f":{color}-background[{pick.verdict}] · "
                            f"{hunt_view.BUCKET_LABEL[pick.bucket]}"
                            + (f" · {vs:+.1f}% vs pivot at hunt close" if vs is not None else ""))
                if pick.notes:
                    st.markdown(pick.notes)
                if nums["depths"] or nums["step3"]:
                    st.caption(" · ".join(x for x in (f"legs {nums['depths']}" if nums["depths"]
                                                      else "", nums["step3"]) if x))
            if payload is not None:
                render_step2_panel(payload)
            else:
                render_step2_from_hunt_row(pick.row)
            with st.container(border=True):
                st.markdown(step_badge("Step 4", "Entry numbers — advisory"))
                _fmt = lambda v, f="{:.2f}": "n/a" if v is None else f.format(v)  # noqa: E731
                st.markdown(f"**Pivot** {_fmt(nums['pivot'])} · buy zone to "
                            f"{_fmt(nums['zone_hi'])} · **stop** {_fmt(nums['stop'])}")
                # Two bare dollar signs in one markdown string render as math.
                st.markdown(f"**ADV** {_fmt(nums['adv_musd'], r'\${:.1f}M')} · max order "
                            f"{_fmt(nums['max_order_usd'], r'\${:,.0f}')}")
                ern = nums["earnings_in"]
                st.markdown(f"**Earnings** {'n/a' if ern is None else f'in {ern}d'} · "
                            f"DD {_fmt(nums['dist_days'], '{:.0f}')} · RS "
                            f"{_fmt(nums['rs'], '{:.0f}')} · Q {_fmt(nums['q'], '{:.0f}')} · "
                            f"F {_fmt(nums['fund'], '{:.0f}')}/{int(nums['f_max'])}")
                st.caption("At the hunt's close. The scan page has today's levels.")

        # ---- the catalyst: the books' third element, read from the hunt's headlines ---- #
        with st.container(border=True):
            st.markdown(step_badge("Catalyst", "Why would institutions want it now?"))
            cat = catalysts.get(pick.ticker)
            if not cat:
                st.caption("Not read for this hunt.")
            else:
                sent = str(cat.get("sentiment") or "")
                scol_ = {"positive": "green", "negative": "red", "mixed": "orange"}.get(sent, "blue")
                st.markdown(f":blue-background[{cat.get('category') or '?'}] "
                            f":{scol_}-background[{sent or '?'}]")
                st.markdown(str(cat.get("summary") or ""))
                srcs = [s for s in (cat.get("sources") or []) if isinstance(s, dict)]
                if srcs:
                    st.markdown("\n".join(
                        f"- {s.get('date') or ''} · [{s.get('title') or s.get('url')}]"
                        f"({s.get('url') or '#'}) — {s.get('publisher') or ''}"
                        for s in srcs))
            st.markdown("**Look further:** " + " · ".join(
                f"[{label}]({url})" for label, url in hunt_view.manual_links(pick.ticker)))
            st.caption("A label, never a gate: the catalyst does not change a verdict.")

        wl_now = st.session_state.get("watchlist")
        on_list = pick.ticker in watchlist_tickers(
            wl_now if wl_now is not None else load_watchlist(cache.WATCHLIST_JSON))
        if on_list:
            st.caption(f"✓ {pick.ticker} is on the watchlist.")
        else:
            st.button(f"⭐ Add {pick.ticker} to the watchlist (pivot {_fmt(nums['pivot'])})",
                      key="hunt_wl_add", on_click=_wl_add_cb, args=(pick, hunt_date))

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
                        height=1400, scrolling=True)
