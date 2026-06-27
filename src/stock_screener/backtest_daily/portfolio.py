"""Position + Portfolio bookkeeping in dollar space; cash is first-class.

Dollar-space cost `(spread/2)*shares` is identical to backtest_lib's fractional
`(spread/2)/price * |dw|` model (since `|dw|*equity = shares*price`), so reported
results stay consistent with the weekly engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class Position:
    permno: int
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    stop: float
    last_phase: int
    reason: str
    last_price: float


class Portfolio:
    def __init__(self, cash: float):
        self.cash = float(cash)
        self.positions: Dict[int, Position] = {}
        self.trades: List[dict] = []          # closed-trade blotter

    def mark(self, permno, price):
        if price is not None and not pd.isna(price):
            self.positions[int(permno)].last_price = float(price)

    def market_value(self) -> float:
        return float(sum(p.shares * p.last_price for p in self.positions.values()))

    def equity(self) -> float:
        return self.cash + self.market_value()

    def enter(self, permno, date, price, shares, stop, cost, reason, phase):
        permno = int(permno)
        self.cash -= shares * price + cost
        self.positions[permno] = Position(
            permno, pd.Timestamp(date), float(price), float(shares), float(stop),
            int(phase), reason, float(price))

    def exit(self, permno, date, price, cost, reason):
        permno = int(permno)
        pos = self.positions.pop(permno)
        price = float(price)
        self.cash += pos.shares * price - cost
        self.trades.append({
            "permno": permno,
            "entry_date": pos.entry_date,
            "exit_date": pd.Timestamp(date),
            "entry_price": pos.entry_price,
            "exit_price": price,
            "shares": pos.shares,
            "ret": (price / pos.entry_price - 1.0) if pos.entry_price else 0.0,
            "pnl": pos.shares * (price - pos.entry_price) - cost,
            "holding_days": int((pd.Timestamp(date) - pos.entry_date).days),
            "entry_reason": pos.reason,
            "exit_reason": reason,
        })
