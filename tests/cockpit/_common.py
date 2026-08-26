"""Shared imports and fixtures for the split cockpit suites.

Every tests/cockpit/test_*.py opens with `from tests.cockpit._common import *`.
The fixtures are underscore-prefixed, which `import *` would normally skip, so
__all__ lists them explicitly — a helper missing from __all__ then fails as a loud
NameError in whichever suite uses it, never silently.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# tests/vcp_labels.py is imported BARE (`from vcp_labels import ...`) by the VCP suites.
# That resolved while these tests lived in tests/test_cockpit.py, because running a script
# puts its own directory on sys.path[0]. From tests/cockpit/ it does not, so put tests/ on
# the path explicitly rather than rewriting the import at every call site.
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import plotly.graph_objects as go  # noqa: E402

from src.stock_screener.backtest_daily.synthetic_provider import make_synthetic  # noqa: E402
from src.stock_screener.cockpit import scan as scan_mod  # noqa: E402
from src.stock_screener.cockpit.charts import build_chart  # noqa: E402
from src.stock_screener.cockpit.scan import ScanConfig, screen_universe  # noqa: E402


def _synthetic_slice(idx=350):
    data = make_synthetic(seed=7)
    cal = data.price.calendar()
    t = cal[idx]
    prices = {}
    for p in data.winners + data.losers + data.late:
        df = data.price.prices(p).loc[:t]
        if len(df) >= 200:
            prices[data.price.ticker(p)] = df
    spy = data.price.spy().loc[:t]
    return prices, spy, data


def _submit_fakes():
    """Build (FakeClient, _Order) for the submit_buy_plan tests — ONE fake Alpaca stack
    shared by the market/limit/rearm-only/pending-buy variants (consolidated 2026-08-10
    from four near-identical per-test copies; every assertion kept). ``get_orders``
    honors BOTH ``filter.symbols`` (the per-symbol open-stop query) and ``filter.side``
    (the pending-cockpit-BUY sweep); ``open_orders`` is a flat list of ``_Order``."""
    from alpaca.trading.enums import OrderSide

    class _Acct:
        equity = "100000"; cash = "100000"; account_number = "PA000123"

    class _Pos:
        def __init__(self, symbol, qty):
            self.symbol, self.qty = symbol, qty

    class _Asset:
        tradable = True

    class _Order:
        def __init__(self, oid, symbol, otype=None, stop_price=None,
                     side=OrderSide.SELL, coid=""):
            self.id, self.symbol, self.type = oid, symbol, otype
            self.side, self.stop_price = side, stop_price
            self.client_order_id = coid

    class _Resp:
        def __init__(self, oid):
            self.id = oid

    class FakeClient:
        def __init__(self, positions=None, open_orders=None):
            self._positions = positions or {}       # {sym: qty_str}
            self._open = open_orders or []          # flat [_Order, ...]
            self.submitted, self.cancelled, self._n = [], [], 0

        def get_account(self):
            return _Acct()

        def get_all_positions(self):
            return [_Pos(s, q) for s, q in self._positions.items()]

        def get_asset(self, t):
            return _Asset()

        def get_orders(self, filter=None):
            side = getattr(filter, "side", None)
            syms = getattr(filter, "symbols", None)
            return [od for od in self._open
                    if (side is None or od.side == side)
                    and (not syms or od.symbol in syms)]

        def cancel_order_by_id(self, oid):
            self.cancelled.append(str(oid))

        def submit_order(self, order_data=None):
            self.submitted.append(order_data)
            self._n += 1
            return _Resp(f"id-{self._n}")

    return FakeClient, _Order


def _submit_entry(t, shares, price, stop, limit=None, **extra):
    """A plan row for _run_submit; est_value uses the limit (worst-case fill) when given,
    mirroring build_buy_plan's basis."""
    e = {"ticker": t, "shares": shares, "price": price, "pivot": price,
         "est_value": round(shares * (limit or price), 2), "extended": False,
         "stop_price": stop, **extra}
    if limit is not None:
        e["limit_price"] = limit
    return e


def _run_submit(plan, fake, attach=True):
    from src.stock_screener.cockpit import trade
    orig = trade._connect_paper
    trade._connect_paper = lambda: (fake, True)
    try:
        return trade.submit_buy_plan(plan, attach_stop=attach)
    finally:
        trade._connect_paper = orig


def _positions_offline(**pos):
    """One-position offline fetch_positions payload shared by the Positions-page
    AppTests (consolidated 2026-08-20 from three near-identical per-test copies).
    Override per-position fields via kwargs; `df` is deliberately absent (the
    bare-dict `.get` tolerance path); the account's total P&L follows the position."""
    p = {"symbol": "AAA", "qty": 10, "avg_entry": 100.0, "current_price": 101.0,
         "market_value": 1010.0, "cost_basis": 1000.0, "unrealized_pl": 10.0,
         "unrealized_plpc": 0.01, "lastday_price": 101.0, "current_stop": 92.0,
         "has_stop": True, "sma_50": 95.0, "last_close": 101.0, "volume_ratio": 1.0,
         "gain_pct": 0.01, "below_sma50": False, "next_earnings": None,
         "earnings_in": None, "stage": "fresh", "advisories": [],
         "template_criteria": 8}
    p.update(pos)
    return {"account": {"account_number": "PA00SZOE", "equity": 50000.0,
                        "cash": 10000.0, "using_dedicated": True,
                        "positions_count": 1,
                        "total_unrealized_pl": p["unrealized_pl"]},
            "positions": [p]}


def _rendered_text(at) -> str:
    """An AppTest run's markdown + caption text, joined for substring asserts."""
    return " ".join(str(getattr(m, "value", ""))
                    for m in list(at.markdown) + list(getattr(at, "caption", [])))


def _pos_fakes():
    """Build (Client, _Pos, _Order) fakes for the positions/re-arm tests. The Client's
    get_orders honors filter.symbols=None -> ALL open orders (the batched query rearm/fetch use);
    _Pos carries the alpaca-py Position P&L attrs; _Order carries symbol/type/stop_price."""
    from alpaca.trading.enums import OrderSide, OrderType

    class _Acct:
        equity = "50000"; cash = "10000"; account_number = "PA00SZOE"

    class _Pos:
        def __init__(self, symbol, qty, **kw):
            self.symbol, self.qty = symbol, str(qty)
            for k, v in kw.items():
                setattr(self, k, v)

    class _Order:
        def __init__(self, oid, symbol, stop_price, otype=None, qty=None):
            self.id, self.symbol = oid, symbol
            self.type = otype or OrderType.STOP
            self.stop_price, self.side = stop_price, OrderSide.SELL
            self.qty = qty                                    # None -> unreadable (no re-quantify)

    class _Resp:
        def __init__(self, oid):
            self.id = oid

    class Client:
        def __init__(self, positions, open_orders=None):
            self._positions = positions
            self._open = open_orders or {}
            self.submitted, self.cancelled, self._n = [], [], 0

        def get_account(self):
            return _Acct()

        def get_all_positions(self):
            return list(self._positions)

        def get_orders(self, filter=None):
            syms = getattr(filter, "symbols", None)
            out = []
            if syms:
                for s in syms:
                    out.extend(self._open.get(s, []))
            else:                                     # batched query: every open order
                for lst in self._open.values():
                    out.extend(lst)
            return out

        def cancel_order_by_id(self, oid):
            self.cancelled.append(str(oid))

        def submit_order(self, order_data=None):
            self.submitted.append(order_data)
            self._n += 1
            return _Resp(f"id-{self._n}")

    return Client, _Pos, _Order


def _trigger_frame(end, closes, vols=None):
    """Daily OHLCV ending exactly at ``end`` for deterministic stale-bar checks."""
    import pandas as pd
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=len(closes))
    vols = vols if vols is not None else [1000] * len(closes)
    return pd.DataFrame({"Open": closes, "High": [c * 1.01 for c in closes],
                         "Low": [c * 0.99 for c in closes], "Close": closes,
                         "Volume": vols}, index=idx)


def _lin_series(segments, start=100.0):
    """Build an OHLCV frame whose Close walks piecewise-linearly through `segments`
    (list of (n_days, end_price)); High=Low=Open=Close so swings are exact/deterministic."""
    import pandas as pd
    closes = [start]
    for days, end in segments:
        s = closes[-1]
        closes += [s + (end - s) * k / days for k in range(1, days + 1)]
    idx = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=len(closes))
    c = pd.Series(closes, index=idx)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c,
                         "Volume": 1_000_000}, index=idx)


def _range_frame(wide=150, tight=60):
    """OHLCV with real intrabar range (unlike ``_lin_series`` where H=L=C): a volatile body
    (closes oscillate ±2.5) then a near-flat tail (closes ±0.1) that still has intraday range.
    Needed for the TTM squeeze, which fires only when close dispersion falls below the range."""
    import pandas as pd
    close, high, low = [], [], []
    for i in range(wide):
        c = 100.0 + (2.5 if i % 2 else -2.5)
        close.append(c); high.append(c + 1.0); low.append(c - 1.0)
    for i in range(tight):
        c = 100.0 + (0.1 if i % 2 else -0.1)
        close.append(c); high.append(c + 1.5); low.append(c - 1.5)
    idx = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=len(close))
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close,
                         "Volume": 1_000_000}, index=idx)


def _ohlc_series(segments, start=100.0, band=0.01):
    """Like ``_lin_series`` but with real intrabar range: High/Low sit ±band around the
    close walk. Needed for adaptive-mode (thr=None) detector tests — H=L=C frames have no
    true range, so they'd false-trigger the dead-tape guard."""
    import pandas as pd
    closes = [start]
    for days, end in segments:
        s = closes[-1]
        closes += [s + (end - s) * k / days for k in range(1, days + 1)]
    idx = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=len(closes))
    c = pd.Series(closes, index=idx)
    return pd.DataFrame({"Open": c, "High": c * (1 + band), "Low": c * (1 - band),
                         "Close": c, "Volume": 1_000_000}, index=idx)


def _log_record(msg: str):
    import logging
    return logging.LogRecord("cockpit.t", logging.INFO, __file__, 1, msg, None, None)



def run_suite(ns, label="cockpit") -> int:
    """Run every ``test_*`` in ``ns`` (a module's globals()); 0 when all pass.

    Shared by each category file's ``__main__`` block and by tests/test_cockpit.py, so a
    suite behaves identically alone or as part of the gate. Sorted by name for a
    deterministic order, and the first failure propagates — the gate wants a non-zero exit
    and a traceback, not a summary of what else it tried afterwards."""
    tests = [v for k, v in sorted(ns.items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} {label} tests passed")
    return 0


__all__ = [
    'Path',
    'ROOT',
    'ScanConfig',
    '_lin_series',
    '_log_record',
    '_ohlc_series',
    '_pos_fakes',
    '_positions_offline',
    '_range_frame',
    '_rendered_text',
    '_run_submit',
    '_submit_entry',
    '_submit_fakes',
    '_synthetic_slice',
    '_trigger_frame',
    'build_chart',
    'go',
    'make_synthetic',
    'os',
    'run_suite',
    'scan_mod',
    'screen_universe',
    'subprocess',
    'sys',
]
