"""Pluggable position sizing. Default = Minervini risk-based.

``weight`` returns the FRACTION of current equity to allocate to a new position; the
engine clamps the total invested to <= 100% (no leverage by default).
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Sizer(ABC):
    @abstractmethod
    def weight(self, sig, equity, n_open, cfg) -> float:
        ...


class EqualWeightSizer(Sizer):
    """1/max_positions per name."""

    def weight(self, sig, equity, n_open, cfg) -> float:
        return 1.0 / max(1, cfg.max_positions)


class RiskBasedSizer(Sizer):
    """Size so the loss at the stop equals ``risk_per_trade_pct`` of equity:
    weight = risk_per_trade_pct / ((entry - stop) / entry).
    Falls back to equal-weight if the stop is missing/degenerate."""

    def weight(self, sig, equity, n_open, cfg) -> float:
        entry = sig.get("entry_price")
        stop = sig.get("stop_loss")
        if not entry or not stop or entry <= 0 or stop >= entry:
            return 1.0 / max(1, cfg.max_positions)
        risk_frac = (entry - stop) / entry
        if risk_frac <= 0:
            return 1.0 / max(1, cfg.max_positions)
        return cfg.risk_per_trade_pct / risk_frac


def make_sizer(cfg) -> Sizer:
    return RiskBasedSizer() if cfg.sizing == "risk" else EqualWeightSizer()
