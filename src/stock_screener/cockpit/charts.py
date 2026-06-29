"""Plotly charts for the cockpit — the surface where the *user* judges the VCP.

``build_chart`` renders a daily (or weekly) candlestick with 50/150/200 SMA overlays
and a volume subplot, plus optional VCP contraction shading and the Step-4 advisory
levels (pivot / buy-zone / stop / target). It draws *hints*; it never decides.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.stock_screener.minervini_screener.screening import calculate_sma

_SMA_STYLE = [(50, "#1f77b4"), (150, "#ff7f0e"), (200, "#2ca02c")]


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to W-FRI (the repo's weekly convention)."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last",
           "Volume": "sum"}
    cols = {k: v for k, v in agg.items() if k in df.columns}
    return df.resample("W-FRI").agg(cols).dropna(subset=["Close"])


def build_chart(ticker: str, df: pd.DataFrame, vcp: Optional[dict] = None,
                levels: Optional[dict] = None, show_overlays: bool = True,
                weekly: bool = False) -> go.Figure:
    """Return a 2-row Plotly figure: candlestick + SMAs (top), volume (bottom)."""
    d = to_weekly(df) if weekly else df

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.76, 0.24], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name=ticker, showlegend=False), row=1, col=1)

    for period, color in _SMA_STYLE:
        if len(d) >= period:
            sma = calculate_sma(d["Close"], period)
            fig.add_trace(go.Scatter(
                x=d.index, y=sma, name=f"SMA{period}",
                line=dict(width=1.2, color=color)), row=1, col=1)

    if "Volume" in d.columns:
        fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="Volume",
                             marker_color="#9aa0a6", showlegend=False), row=2, col=1)

    if show_overlays and vcp:
        # Shade each detected contraction (peak -> trough): a VCP *hint*.
        for c in (vcp.get("contractions") or [])[-6:]:
            try:
                fig.add_vrect(x0=c["peak_date"], x1=c["trough_date"],
                              fillcolor="LightSalmon", opacity=0.15, line_width=0,
                              row=1, col=1)
            except Exception:
                pass

    if show_overlays and levels:
        piv, stp, tgt = levels.get("pivot"), levels.get("stop"), levels.get("target")
        bz = levels.get("buy_zone")
        if bz:
            fig.add_hrect(y0=bz[0], y1=bz[1], fillcolor="green", opacity=0.07,
                          line_width=0, row=1, col=1)
        if piv:
            fig.add_hline(y=piv, line=dict(color="green", dash="dash"),
                          annotation_text="pivot", annotation_position="right",
                          row=1, col=1)
        if stp:
            fig.add_hline(y=stp, line=dict(color="red", dash="dot"),
                          annotation_text="stop", annotation_position="right",
                          row=1, col=1)
        if tgt:
            fig.add_hline(y=tgt, line=dict(color="royalblue", dash="dot"),
                          annotation_text="target", annotation_position="right",
                          row=1, col=1)

    title = f"{ticker} — {'weekly' if weekly else 'daily'}"
    fig.update_layout(
        title=title, height=660, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02),
        hovermode="x unified")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    return fig
