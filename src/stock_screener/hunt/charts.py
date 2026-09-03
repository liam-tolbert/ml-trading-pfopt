"""Review-sheet renderer: 4 candidates per PNG, dense enough to judge Step 3.

Each panel: log-scale candles (~9 months), SMA 50/150/200, the detected pivot
(dashed), the +5% entry ceiling (dotted), the stop, the detected contraction
legs with depth labels, and a volume pane with its 50-day average. The title
carries the numbers the reviewer needs (quality, RS, F, vs-pivot, ADV$, DD,
days to earnings) so a verdict never requires a second lookup.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter, ScalarFormatter

from src.stock_screener.cockpit.doctrine import VOL_AVG_DAYS
from src.stock_screener.cockpit.indicators import prior_volume_average
from .pipeline import BUY_ZONE_MAX_PCT, ScanBundle

BARS = 185           # ~9 months of dailies: full base plus context
PER_FIG = 4
_UP, _DN = "#1a9850", "#d73027"


def render_sheets(bundle: ScanBundle, diag: pd.DataFrame, out_dir: Path,
                  per_fig: int = PER_FIG, bars: int = BARS) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    recs = list(diag.to_dict("records"))
    for gi in range(0, len(recs), per_fig):
        grp = recs[gi:gi + per_fig]
        fig = plt.figure(figsize=(16.5, 18), dpi=110)
        gs = GridSpec(4, 2, height_ratios=[3, 1, 3, 1], hspace=0.28, wspace=0.10)
        for j, r in enumerate(grp):
            _panel(fig, gs, j, bundle, r, bars)
        n = gi // per_fig + 1
        p = out_dir / f"sheet_{n:03d}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def _panel(fig, gs, j, bundle: ScanBundle, r: dict, bars: int) -> None:
    t = r["ticker"]
    p = bundle.result.payloads[t]
    df = p["df"].tail(bars)
    lev, v = p["levels"], p["vcp"]
    o, h, l, c, vol = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close", "Volume"))
    x = np.arange(len(df))
    up = c >= o

    axp = fig.add_subplot(gs[(j // 2) * 2, j % 2])
    axv = fig.add_subplot(gs[(j // 2) * 2 + 1, j % 2], sharex=axp)

    color = np.where(up, _UP, _DN)
    axp.vlines(x, l, h, color=color, lw=0.6)
    axp.vlines(x, np.minimum(o, c), np.maximum(o, c), color=color, lw=2.2)

    full_c = p["df"]["Close"]
    for win, col in ((50, "#1f78b4"), (150, "#ff7f00"), (200, "#6a3d9a")):
        axp.plot(x, full_c.rolling(win).mean().tail(bars).to_numpy(), color=col, lw=1.0)

    piv = float(lev["pivot"])
    axp.axhline(piv, color=_UP, ls="--", lw=1.2)                                  # pivot
    axp.axhline(piv * (1 + BUY_ZONE_MAX_PCT / 100), color=_UP, ls=":", lw=0.8,    # entry ceiling
                alpha=0.6)
    axp.axhline(float(lev["stop"]), color=_DN, ls=":", lw=1.0)                    # stop

    idx = df.index
    for cn in v["contractions"]:
        try:
            i0 = idx.get_indexer([cn["peak_date"]], method="nearest")[0]
            i1 = idx.get_indexer([cn["trough_date"]], method="nearest")[0]
            axp.plot([i0, i1], [cn["peak_price"], cn["trough_price"]],
                     color="black", lw=1.6, marker="o", ms=3)
            axp.annotate(f"{cn['drawdown_pct']:.0f}%", xy=(i1, cn["trough_price"]),
                         fontsize=7, xytext=(0, -10), textcoords="offset points")
        except Exception:
            pass                                       # a leg outside the window is fine

    axp.set_yscale("log")
    axp.yaxis.set_major_formatter(ScalarFormatter())
    axp.yaxis.set_minor_formatter(NullFormatter())
    mk = [i for i in range(1, len(idx)) if idx[i].month != idx[i - 1].month]
    axp.set_xticks(mk)
    axp.set_xticklabels([idx[i].strftime("%b") for i in mk], fontsize=7)

    # The gate's own denominator (prior N bars, today excluded), so the plotted line is
    # the one a confirmation is actually measured against.
    vs50 = prior_volume_average(pd.Series(vol), VOL_AVG_DAYS).to_numpy()
    axv.bar(x, vol, color=color, width=0.8, alpha=0.8)
    axv.plot(x, vs50, color="#1f78b4", lw=1.0)
    axv.set_yticks([])

    star = "*" if r.get("wl") else ""
    try:
        ern = f"{int(float(r['earnings_in']))}d"
    except (TypeError, ValueError):
        ern = "-"
    axp.set_title(
        f"#{r['rank']} {t}{star}  q={r['q']:.0f} RS={r['rs']} F={r['fund']}  "
        f"c={r['close']:.2f} piv={r['pivot']:.2f} ({r['vs_pivot_pct']:+.1f}%)  "
        f"ADV${r['adv_musd']:.1f}M  DD={r['dist_days']}  ern={ern}",
        fontsize=9, loc="left")
