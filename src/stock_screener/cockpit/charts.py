"""Plotly charts for the cockpit — the surface where the *user* judges the VCP.

``build_chart`` renders a daily (or weekly) candlestick with 50/150/200 SMA overlays
and a volume subplot, plus optional VCP contraction shading, an optional Bollinger-band
envelope (toggled from the UI), and the Step-4 advisory levels (pivot / buy-zone / stop /
target). It draws *hints*; it never decides.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.stock_screener.minervini_screener.screening import calculate_sma
from src.stock_screener.cockpit.indicators import (bollinger_bands,
                                                   relative_measured_volatility)

_SMA_STYLE = [(50, "#1f77b4"), (150, "#ff7f0e"), (200, "#2ca02c")]


def _contraction_hover(c: dict) -> str:
    """One-line summary of a VCP contraction for the chart tooltip.

    Surfaces the numbers a user judges a VCP on: how deep the pullback was, the
    peak->trough prices, how long it took, and whether volume dried up (< 1.0×).
    """
    parts = []
    dd = c.get("drawdown_pct")
    if dd is not None:
        parts.append(f"{dd:.1f}% deep")
    pk, tr = c.get("peak_price"), c.get("trough_price")
    if pk is not None and tr is not None:
        parts.append(f"${pk:,.2f} → ${tr:,.2f}")
    dur = c.get("duration_days")
    if dur:
        parts.append(f"{dur}d")
    vr = c.get("volume_ratio")
    if vr is not None:
        parts.append(f"vol {vr:.2f}×")
    return " · ".join(parts)


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to W-FRI (the repo's weekly convention)."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last",
           "Volume": "sum"}
    cols = {k: v for k, v in agg.items() if k in df.columns}
    return df.resample("W-FRI").agg(cols).dropna(subset=["Close"])


def build_chart(ticker: str, df: pd.DataFrame, vcp: Optional[dict] = None,
                levels: Optional[dict] = None, show_overlays: bool = True,
                weekly: bool = False, lookback_days: Optional[int] = None,
                show_bollinger: bool = False) -> go.Figure:
    """Return a 2-row Plotly figure: candlestick + SMAs (top), volume (bottom).

    ``lookback_days`` zooms the VIEW to the last N calendar days so a multi-week VCP
    base is actually visible; SMAs are still computed on the full history (so the
    50/150/200 lines stay correct) and the price y-axis is fit to the window.

    ``show_bollinger`` overlays the 20-period / 2σ Bollinger envelope (upper, 20-SMA
    basis, lower) on the price row — the same bands the TTM-squeeze / BBWP reads are
    built from, so you can eyeball the compression the Step-4 squeeze numbers report.
    """
    d_full = to_weekly(df) if weekly else df

    # SMAs on FULL history, so they're correct even when the view is short.
    smas = [(period, color, calculate_sma(d_full["Close"], period))
            for period, color in _SMA_STYLE if len(d_full) >= period]

    if lookback_days and len(d_full):
        cutoff = d_full.index[-1] - pd.Timedelta(days=int(lookback_days))
        d = d_full.loc[d_full.index >= cutoff]
    else:
        d = d_full

    # 3 rows: price (top), compressed volume pane, RMV (bottom).
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.62, 0.19, 0.19], vertical_spacing=0.03)

    # Bollinger envelope — added BEFORE the candles so it renders beneath them (Scatter has no
    # layer="below"; z-order is trace order). Lower + upper-with-fill draws a band, dotted
    # 20-SMA basis is the midline. Computed on FULL history so bands stay correct when zoomed.
    if show_bollinger and len(d_full) >= 20:
        bb_up, bb_mid, bb_lo = (s.reindex(d.index) for s in bollinger_bands(d_full))
        band = "rgba(75,108,183,0.55)"
        fig.add_trace(go.Scatter(x=d.index, y=bb_lo, name="BB lower", line=dict(width=1, color=band),
                                 showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=bb_up, name="Bollinger (20, 2σ)",
                                 line=dict(width=1, color=band), fill="tonexty",
                                 fillcolor="rgba(75,108,183,0.07)", hoverinfo="skip"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=bb_mid, name="BB basis (SMA20)",
                                 line=dict(width=1, color="rgba(75,108,183,0.8)", dash="dot"),
                                 showlegend=False, hoverinfo="skip"), row=1, col=1)

    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name=ticker, showlegend=False), row=1, col=1)

    for period, color, sma in smas:
        fig.add_trace(go.Scatter(
            x=d.index, y=sma.reindex(d.index), name=f"SMA{period}",
            line=dict(width=1.2, color=color)), row=1, col=1)

    if "Volume" in d.columns:
        # Bars colored by up/down day (close vs prior close): heavy volume on DOWN days is
        # distribution — a VCP disqualifier. First bar of the window has no prior close -> gray.
        _chg = d["Close"].diff()
        _vcol = ["#9aa0a6" if pd.isna(c) else ("#4c9e70" if c >= 0 else "#d96b5f")
                 for c in _chg]
        fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="Volume",
                             marker_color=_vcol, showlegend=False), row=2, col=1)
        # Volume baseline (20-period SMA) + the 1.5× breakout threshold. In the DAILY view
        # this SMA is the engine's 20-day average and the dashed line is the 50%-above level
        # a confirmed breakout must clear (see detect_breakout). Computed on full history.
        if "Volume" in d_full.columns and len(d_full) >= 20:
            unit = "w" if weekly else "d"
            vsma = calculate_sma(d_full["Volume"], 20).reindex(d.index)
            fig.add_trace(go.Scatter(x=d.index, y=vsma, name=f"Vol SMA20{unit}",
                                     line=dict(width=1.1, color="#5f6368")), row=2, col=1)
            fig.add_trace(go.Scatter(x=d.index, y=vsma * 1.5, name="1.5× (breakout)",
                                     line=dict(width=1.1, color="#d93025", dash="dash")),
                          row=2, col=1)

    # Row 3: RMV (Relative Measured Volatility) — normalized 0-100, low = tight base. Computed
    # on FULL history so the min-max normalization window stays stable when zoomed. The shaded
    # band marks the < 25 "tight" zone — the VCP sweet spot.
    if len(d_full) >= 15:
        rmv = relative_measured_volatility(d_full).reindex(d.index)
        fig.add_hrect(y0=0, y1=25, fillcolor="green", opacity=0.08, line_width=0,
                      row=3, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=rmv, name="RMV",
                                 line=dict(width=1.3, color="#8e44ad")), row=3, col=1)

    if show_overlays and vcp:
        # Shade each detected contraction (peak -> trough): a VCP *hint*.
        for c in (vcp.get("contractions") or [])[-6:]:
            try:
                fig.add_vrect(x0=c["peak_date"], x1=c["trough_date"],
                              fillcolor="LightSalmon", opacity=0.15, line_width=0,
                              row=1, col=1)
                # add_vrect is a layout shape and can't carry a tooltip, so overlay an invisible
                # hoverable scatter across the same span (adds a "Contraction N" row under
                # hovermode "x unified"). Restrict x to visible candles so points land on real
                # plotted (rangebreak-safe) positions and don't nudge the y-axis.
                hx = d.index[(d.index >= c["peak_date"]) & (d.index <= c["trough_date"])]
                if len(hx):
                    fig.add_trace(go.Scatter(
                        x=hx, y=[c["peak_price"]] * len(hx), mode="markers",
                        marker=dict(size=6, opacity=0, color="LightSalmon"),
                        name=f"Contraction {c.get('number', '')}".strip(),
                        hoverinfo="text", hovertext=_contraction_hover(c),
                        showlegend=False), row=1, col=1)
            except Exception:
                pass

    if show_overlays and levels:
        piv, stp, tgt = levels.get("pivot"), levels.get("stop"), levels.get("target")
        bz = levels.get("buy_zone")
        if bz:
            fig.add_hrect(y0=bz[0], y1=bz[1], fillcolor="green", opacity=0.07,
                          line_width=0, row=1, col=1)
        # "top right" anchors the label INSIDE the plot so it never clips off the side the
        # way "right" (outside) does.
        if piv:
            fig.add_hline(y=piv, line=dict(color="green", dash="dash"),
                          annotation_text="pivot", annotation_position="top right",
                          annotation_bgcolor="rgba(255,255,255,0.6)", row=1, col=1)
        if stp:
            fig.add_hline(y=stp, line=dict(color="red", dash="dot"),
                          annotation_text="stop", annotation_position="bottom right",
                          annotation_bgcolor="rgba(255,255,255,0.6)", row=1, col=1)
        if tgt:
            fig.add_hline(y=tgt, line=dict(color="royalblue", dash="dot"),
                          annotation_text="target", annotation_position="top right",
                          annotation_bgcolor="rgba(255,255,255,0.6)", row=1, col=1)

    title = f"{ticker} — {'weekly' if weekly else 'daily'}"
    fig.update_layout(
        title=title, height=720, margin=dict(l=10, r=24, t=40, b=10),
        xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02),
        hovermode="x unified")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="RMV", range=[0, 100], row=3, col=1)
    # Collapse non-trading spans (weekends, holidays, no-data days) so candles sit flush —
    # Plotly's date axis spaces points by CALENDAR time and otherwise leaves blank gaps.
    # Purely cosmetic; the VCP math runs on trading-day ROWS. Computed from the data's own
    # dates (no hardcoded calendar), so it works for both daily and weekly views.
    if len(d) > 1:
        idx = pd.DatetimeIndex(d.index).normalize()
        missing = pd.date_range(idx.min(), idx.max(), freq="D").difference(idx)
        if len(missing):
            fig.update_xaxes(rangebreaks=[
                dict(values=missing.strftime("%Y-%m-%d").tolist())])
    # Fit the price pane to the VISIBLE candles so a zoomed-in base fills the chart
    # (SMA lines outside the window simply clip).
    if len(d):
        ylo, yhi = float(d["Low"].min()), float(d["High"].max())
        if yhi > ylo:
            pad = (yhi - ylo) * 0.06
            fig.update_yaxes(range=[ylo - pad, yhi + pad], row=1, col=1)
    return fig
