"""Plotly charts for the cockpit: the surface where the *user* judges the VCP.

``build_chart`` renders a daily or weekly candlestick with 50/150/200 SMA overlays, a
volume pane and an RMV pane. Optional: VCP contraction shading, a Bollinger-band envelope
(toggled from the UI), and the Step-4 advisory levels (pivot / buy-zone / stop / target).
It draws *hints*; it never decides.
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


def _add_marks(fig: go.Figure, d: pd.DataFrame, marks: list) -> None:
    """One marker trace per (pane, symbol) group of ``marks`` on bars inside ``d``."""
    groups: dict = {}
    for m in marks:
        try:
            ts = pd.Timestamp(m["date"])
        except Exception:
            continue
        if ts not in d.index:
            continue
        pane = m.get("pane", "price")
        if pane == "volume" and "Volume" in d.columns:
            y, row = float(d.at[ts, "Volume"]), 2
        else:
            y, row = float(d.at[ts, "Low"]) * 0.985, 1
        key = (row, m.get("symbol", "triangle-up"), m.get("color", "#1a73e8"))
        groups.setdefault(key, []).append((ts, y, m.get("text", "")))
    for (row, symbol, color), pts in groups.items():
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers",
            marker=dict(symbol=symbol, size=9, color=color), hoverinfo="text",
            hovertext=[p[2] for p in pts], showlegend=False), row=row, col=1)


def build_chart(ticker: str, df: pd.DataFrame, vcp: Optional[dict] = None,
                levels: Optional[dict] = None, show_overlays: bool = True,
                weekly: bool = False, lookback_days: Optional[int] = None,
                show_bollinger: bool = False, marks: Optional[list] = None) -> go.Figure:
    """Return a 3-row Plotly figure: candlestick + SMAs (top), volume, RMV (bottom).

    ``marks`` are Step-3 read markers, ``{"date", "pane": "price"|"volume", "text",
    "symbol", "color"}``: a price mark sits under that bar's low, a volume mark on top of
    its bar. Drawn in the daily view only, with the overlays.

    ``lookback_days`` zooms the view to the last N calendar days so a multi-week VCP
    base is visible. SMAs are still computed on the full history, so the 50/150/200
    lines stay correct. The price y-axis is fit to the window.

    ``show_bollinger`` overlays the 20-period / 2σ Bollinger envelope (upper, 20-SMA
    basis, lower) on the price row. These are the bands the TTM-squeeze and BBWP reads
    are built from, so the chart shows the compression the Step-4 squeeze numbers report.
    """
    d_full = to_weekly(df) if weekly else df

    # Full history, so the SMAs are correct even when the view is short.
    smas = [(period, color, calculate_sma(d_full["Close"], period))
            for period, color in _SMA_STYLE if len(d_full) >= period]

    if lookback_days and len(d_full):
        cutoff = d_full.index[-1] - pd.Timedelta(days=int(lookback_days))
        d = d_full.loc[d_full.index >= cutoff]
    else:
        d = d_full

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.62, 0.19, 0.19], vertical_spacing=0.03)

    # The envelope MUST be added before the candles to render beneath them: Scatter has no
    # layer="below", and z-order is trace order. "tonexty" fills the upper trace down to the
    # lower one, so lower comes first. Full history keeps the bands correct when zoomed.
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
        # Colored by close vs prior close: heavy volume on down days is distribution, a VCP
        # disqualifier. The window's first bar has no prior close and is gray.
        _chg = d["Close"].diff()
        _vcol = ["#9aa0a6" if pd.isna(c) else ("#4c9e70" if c >= 0 else "#d96b5f")
                 for c in _chg]
        fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="Volume",
                             marker_color=_vcol, showlegend=False), row=2, col=1)
        # Volume 20-SMA and 1.5× it: in the daily view, roughly the level detect_breakout's
        # volume check wants cleared. The engine averages the 20 bars before the bar; this
        # SMA includes the bar itself. Computed on full history.
        if "Volume" in d_full.columns and len(d_full) >= 20:
            unit = "w" if weekly else "d"
            vsma = calculate_sma(d_full["Volume"], 20).reindex(d.index)
            fig.add_trace(go.Scatter(x=d.index, y=vsma, name=f"Vol SMA20{unit}",
                                     line=dict(width=1.1, color="#5f6368")), row=2, col=1)
            fig.add_trace(go.Scatter(x=d.index, y=vsma * 1.5, name="1.5× (breakout)",
                                     line=dict(width=1.1, color="#d93025", dash="dash")),
                          row=2, col=1)

    # RMV on full history, so its min-max window stays stable when zoomed. The shaded band is
    # the < 25 tight zone, the VCP sweet spot.
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
                # A vrect is a layout shape and can't carry a tooltip, so an invisible scatter
                # spans it and adds a "Contraction N" hover row. Its x MUST be visible candles,
                # so points land on plotted, rangebreak-safe positions.
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

    if show_overlays and marks and not weekly:
        _add_marks(fig, d, marks)

    if show_overlays and levels:
        piv, stp, tgt = levels.get("pivot"), levels.get("stop"), levels.get("target")
        bz = levels.get("buy_zone")
        if bz:
            fig.add_hrect(y0=bz[0], y1=bz[1], fillcolor="green", opacity=0.07,
                          line_width=0, row=1, col=1)
        # "top right" keeps the label inside the plot; "right" sits outside and clips.
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
    # Plotly spaces a date axis by calendar time, leaving gaps for weekends, holidays and
    # no-data days; rangebreaks collapse them. Cosmetic only: the VCP math runs on rows. The
    # breaks come from the data's own dates, so daily and weekly views both work.
    if len(d) > 1:
        idx = pd.DatetimeIndex(d.index).normalize()
        missing = pd.date_range(idx.min(), idx.max(), freq="D").difference(idx)
        if len(missing):
            fig.update_xaxes(rangebreaks=[
                dict(values=missing.strftime("%Y-%m-%d").tolist())])
    # Fit the price pane to the visible candles so a zoomed-in base fills the chart. SMA
    # lines outside that range clip.
    if len(d):
        ylo, yhi = float(d["Low"].min()), float(d["High"].max())
        if yhi > ylo:
            pad = (yhi - ylo) * 0.06
            fig.update_yaxes(range=[ylo - pad, yhi + pad], row=1, col=1)
    return fig
