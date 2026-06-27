"""Reporting — reuse the pfopt helpers (at daily frequency) + screener trade stats."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml_stock_prediction.backtest_lib import compute_metrics
from src.stock_screener.momentum_lib import capm_alpha_beta

DAYS_PER_YEAR = 252


def build_report(daily: pd.DataFrame, trades: list, cfg) -> dict:
    net = daily["net"].astype(float)
    spy = daily["spy_ret"].astype(float)

    rep = {
        "strategy": compute_metrics(net, periods_per_year=DAYS_PER_YEAR),
        "spy_buyhold": compute_metrics(spy, periods_per_year=DAYS_PER_YEAR),
        "capm": capm_alpha_beta(net, spy, periods_per_year=DAYS_PER_YEAR),
    }

    n = len(trades)
    if n:
        rets = np.array([t["ret"] for t in trades], dtype=float)
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        if avg_loss != 0:
            payoff = float(avg_win / abs(avg_loss))
        else:
            payoff = float("inf") if len(wins) else 0.0
        rep["trades"] = {
            "n_trades": int(n),
            "win_rate": float(len(wins) / n),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "payoff_ratio": payoff,
            "expectancy": float(rets.mean()),
            "avg_holding_days": float(np.mean([t["holding_days"] for t in trades])),
        }
    else:
        rep["trades"] = {"n_trades": 0, "win_rate": float("nan"), "avg_win": 0.0,
                         "avg_loss": 0.0, "payoff_ratio": float("nan"),
                         "expectancy": float("nan"), "avg_holding_days": float("nan")}

    expo = (daily["invested"] / daily["equity"]).replace([np.inf, -np.inf], np.nan)
    rep["pct_time_in_cash"] = float((daily["n_positions"] == 0).mean())
    rep["average_exposure"] = float(expo.mean())
    return rep
