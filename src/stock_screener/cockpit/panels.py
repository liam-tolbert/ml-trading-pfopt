"""Panels shared by app.py and the pages.

Pages MUST NOT import app.py: importing it runs the scan page. What the scan page and a
page both render lives here, so the two stay identical.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

from src.stock_screener.cockpit import advisories
from src.stock_screener.cockpit.doctrine import EARNINGS_SOON_DAYS
from src.stock_screener.cockpit.scan import step2_lines

INFO_STEP2 = """
**Step 2 — Fundamentals (the fuel).** A great chart with weak earnings is a trap.
Look for:
- **EPS & revenue YoY ≥ ~20%** and **accelerating** (this quarter ≥ last),
- **stable or expanding margins** (positive *margin trend*).
- **Code 33**: EPS growth, sales growth *and* net margin all rising for three quarters
  running, Minervini's favourite pattern in the numbers (from SEC filings).
- **Annual EPS** higher than the year before, ideally three years running.
- ⚠️ **EPS growth slowing two quarters running**, or **inventory growing faster than
  sales**: the books' warning signs.
- **How the stock took its last report.** A drop of 5% or more on heavy volume after
  the release means big investors were selling the news: the books say stay away.
- **Analyst estimates raised** ≥ 5% over 90 days, and **fund ownership** (how many
  institutions hold it, and whether that count is rising). Fund *quality* still needs a
  look by hand.

`fund_score` (0–8) counts how many of eight checks pass: revenue ≥ 20%, EPS ≥ 20%,
EPS accelerating, margin expanding, Code 33, annual EPS up, estimates raised ≥ 5%, and
the last report *held* (no hard drop on heavy volume). A missing figure counts as a
fail, so read the numbers, not just the score. yfinance often exposes only ~4
quarters, so **YoY may read n/a** — QoQ is the fallback. Use this to rank the
Step-1 list, not as a hard cutoff unless you set "min fundamental checks".

The **next earnings date** shows here too — it's an *entry-timing* input
(see Step 4): don't open a fresh position within ~2–3 weeks of a report.
"""

# The eight Step-2 checks: scan key, chip label, and the hunt diagnostics column.
STEP2_CHECKS = (("revenue_growth", "Rev ≥20%", "f_rev"), ("eps_growth", "EPS ≥20%", "f_eps"),
                ("eps_accelerating", "EPS accel", "f_accel"),
                ("margin_expanding", "Margin ↑", "f_margin"), ("code33", "Code 33", "f_code33"),
                ("annual_eps_up", "Annual EPS ↑", "f_fy"),
                ("estimates_raised", "Estimates ↑", "f_est"), ("report_held", "Report held", "f_react"))


def info_btn(body: str, label: str = "ℹ️ How to use") -> None:
    """A small clickable info popover (falls back to an expander on older Streamlit)."""
    try:
        with st.popover(label):
            st.markdown(body)
    except Exception:
        with st.expander(label):
            st.markdown(body)


def earnings_flag(days) -> str:
    """'⚠︎ earnings in Nd' when a report is 0 to ``EARNINGS_SOON_DAYS`` days out, else ''."""
    return (f"⚠︎ earnings in {int(days)}d"
            if days is not None and 0 <= days <= EARNINGS_SOON_DAYS else "")


def step_badge(step: str, title: str) -> str:
    """A consistent blue step chip + title, e.g. ':blue-background[Step 3]  Judge the VCP'."""
    return f":blue-background[{step}]  {title}"


def _pct(v) -> str:
    """A signed growth percentage, or n/a. Accepts the CSV's strings."""
    try:
        return f"{float(v):+.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _step2_header(help_popover: bool) -> None:
    """The box's title row. Without the popover the same help sits on the title's ⓘ,
    one line shorter, for pages that are short of height."""
    if help_popover:
        st.markdown(step_badge("Step 2", "Fundamentals — the fuel"))
        info_btn(INFO_STEP2)
    else:
        st.markdown(step_badge("Step 2", "Fundamentals — the fuel"), help=INFO_STEP2)


def render_step2_panel(payload: dict, help_popover: bool = True) -> None:
    """The Step-2 box for a scan payload: header, help and the fundamentals read."""
    with st.container(border=True):
        _step2_header(help_popover)
        f = payload.get("fundamentals")
        s2 = payload.get("step2", {})
        if not f:
            st.caption("No fundamental data available (yfinance).")
            return

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
            warn = " ⚠️" if earnings_flag(ei) else ""
            st.markdown(f"**Earnings:** {ne}{when}{warn}")
        # From data_feed._edgar_backfill: yfinance lacks FY growth and 3-quarter
        # acceleration.
        _fy, _acc, _sp, _sd = (f.get("eps_fy_yoy"), f.get("eps_accel_3q"),
                               f.get("last_surprise_pct"), f.get("last_surprise_date"))
        _extra = []
        if _fy is not None:
            _extra.append(f"**FY EPS:** {_fy:+.1f}%")
        if _acc is not None:
            _extra.append("3q accel ✅" if _acc else "3q accel —")
        # An undated surprise comes from a cache that predates the age check and can
        # be a year old, so it is not shown.
        if _sp is not None and _sd:
            _extra.append(f"surprise {_sp:+.1f}% ({_sd})")
        if _extra:
            st.markdown(" · ".join(_extra))
        for _line in step2_lines(f):
            st.markdown(_line)
        _rx = advisories.earnings_reaction_text(payload.get("reaction"),
                                                f.get("last_report"),
                                                f.get("last_report_time"))
        if _rx:
            st.markdown(_rx)
        checks = s2.get("checks", {})
        st.markdown(" ".join(f"{'✅' if checks.get(k) else '—'} {lbl}"
                             for k, lbl, _ in STEP2_CHECKS if k in checks))
        # An older scan's summary holds four checks.
        st.caption(f"Score {s2.get('score', 0)}/{len(checks) or 8}")


def render_step2_from_hunt_row(row: dict, help_popover: bool = True) -> None:
    """The Step-2 box from a hunt's diagnostics row, for a name the latest scan no
    longer holds. Same header and chips; the numbers are the hunt's."""
    with st.container(border=True):
        _step2_header(help_popover)
        st.caption("The hunt's numbers: this name is not in the latest scan.")
        st.markdown(f"**Rev:** {_pct(row.get('rev_yoy'))} YoY")
        st.markdown(f"**EPS:** {_pct(row.get('eps_yoy'))} YoY")
        extra = []
        c33 = row.get("code33")
        if c33 not in (None, ""):
            extra.append(f"**Code 33** {int(float(c33))}/3")
        if str(row.get("inv_flag")) == "True":
            extra.append("⚠️ inventory outrunning sales")
        react = row.get("earn_react")
        if react not in (None, ""):
            extra.append(f"last report {_pct(react)}"
                         + (" ⚠️ hard drop" if row.get("earn_flag") == "hard_drop" else ""))
        est = row.get("est_rev_90d")
        if est not in (None, ""):
            extra.append(f"estimates {_pct(est)} over 90d")
        inst = row.get("inst_count")
        if inst not in (None, ""):
            extra.append(f"{int(float(inst))} funds")
        if extra:
            st.markdown(" · ".join(extra))
        st.markdown(" ".join(f"{'✅' if int(row.get(col) or 0) else '—'} {lbl}"
                             for _, lbl, col in STEP2_CHECKS))
        st.caption(f"Score {row.get('fund', 0)}/{row.get('f_max', 8)}")
