"""Backtest configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class BacktestConfig:
    """Knobs for the daily event-driven backtest.

    All defaults are daily-frequency oriented. ``sizing`` picks the default Sizer
    when the engine isn't handed one explicitly.
    """
    start: Optional[pd.Timestamp] = None      # inclusive; None = first cache date
    end: Optional[pd.Timestamp] = None        # inclusive; None = last cache date
    initial_equity: float = 100_000.0
    max_positions: int = 20
    scan_every_days: int = 5                  # entry-scan cadence (stops checked daily)
    min_phase2_pct: float = 15.0              # breadth gate for buys (benchmark.should_generate_signals)
    exit_on_regime_flip: bool = False         # liquidate book when SPY enters phase 4
    sizing: str = "risk"                      # "risk" (Minervini) | "equal"
    risk_per_trade_pct: float = 0.01          # risk-based sizing: loss-at-stop = this % of equity
    spread_per_share: float = 0.02            # cost = (spread/2) * shares, per side
    stop_trigger: str = "low"                 # "low" (intraday) | "close"
    min_history_rows: int = 200               # classify_phase needs >= 200 rows
    candidate_cap: Optional[int] = None       # cap candidates/scan (None = unbounded)
    buy_score_min: float = 60.0               # mirror vendored is_buy gate (informational)
    seed: int = 0

    def __post_init__(self):
        if self.start is not None:
            self.start = pd.Timestamp(self.start)
        if self.end is not None:
            self.end = pd.Timestamp(self.end)
