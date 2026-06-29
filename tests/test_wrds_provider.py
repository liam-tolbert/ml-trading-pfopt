"""Tests for the WRDS-backed providers using a FIXTURE cache (no live WRDS).

Writes a tiny cache (via the same cache_io the ingester uses), loads the WRDS
providers from it, checks they satisfy the provider contract, and runs the engine
end-to-end. This keeps wrds_provider.py CI-testable without credentials or network.

Run: `python tests/test_wrds_provider.py`  (or under pytest).
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stock_screener.backtest_daily.cache_io import write_table              # noqa: E402
from src.stock_screener.backtest_daily.config import BacktestConfig            # noqa: E402
from src.stock_screener.backtest_daily.engine import BacktestEngine            # noqa: E402
from src.stock_screener.backtest_daily.wrds_provider import (                  # noqa: E402
    load_wrds_providers, WrdsPriceProvider, WrdsUniverseProvider, WrdsFundamentalsProvider,
)
from src.stock_screener.backtest_daily.synthetic_provider import (             # noqa: E402
    _ohlcv, _winner_close, _spy_close,
)

N = 420
START = "2015-01-02"
WINNERS = (1001, 1002)


def _write_fixture_cache(d):
    """Build a tiny but valid cache in the shape ingest_wrds.py produces."""
    cal = pd.bdate_range(START, periods=N)

    # prices: two winners (base -> Stage-2 uptrend)
    rows = []
    for permno, seed in zip(WINNERS, (1, 2)):
        # base -> uptrend (buyable) -> decline (triggers the sell-signal exit)
        f = _ohlcv(_winner_close(N, 200, 120, 80, np.random.default_rng(seed)), cal)
        for dt, r in f.iterrows():
            rows.append({"permno": permno, "date": dt, "open": r.Open, "high": r.High,
                         "low": r.Low, "close": r.Close, "volume": r.Volume})
    write_table(pd.DataFrame(rows), d, "prices")

    # spy: uptrend; its dates define the calendar
    spy = _ohlcv(_spy_close(N, np.random.default_rng(9)), cal, band=0.004, vol=5e6)
    spy = spy.reset_index().rename(columns={"Date": "date", "Open": "open", "High": "high",
                                            "Low": "low", "Close": "close", "Volume": "volume"})
    write_table(spy, d, "spy")

    # universe: both names eligible from each quarterly rebalance
    rebals = cal[::63]
    write_table(pd.DataFrame([{"rebalance_date": rd, "permno": p}
                              for rd in rebals for p in WINNERS]), d, "universe")

    # delist: none (empty table with the right columns)
    write_table(pd.DataFrame(columns=["permno", "delist_date", "dlret"]), d, "delist")

    # fundamentals: growing revenue/eps, rdq lagged 45d
    frows = []
    for permno in WINNERS:
        rev, eps, inv = 100.0, 1.0, 50.0
        for k in range(0, N, 63):
            dd = cal[k]
            frows.append({"permno": permno, "datadate": dd, "rdq": dd + pd.Timedelta(days=45),
                          "revtq": rev, "eps": eps, "invtq": inv})
            rev *= 1.08
            eps *= 1.08
    write_table(pd.DataFrame(frows), d, "fundamentals")


def test_wrds_providers_read_cache_and_satisfy_contract():
    with tempfile.TemporaryDirectory() as d:
        _write_fixture_cache(d)
        price, uni, fund = load_wrds_providers(d)

        assert isinstance(price, WrdsPriceProvider)
        assert isinstance(uni, WrdsUniverseProvider)
        assert isinstance(fund, WrdsFundamentalsProvider)

        # price contract
        assert set(price.permnos()) == set(WINNERS)
        assert len(price.calendar()) > 300
        f = price.prices(1001)
        assert list(f.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert f.index.is_monotonic_increasing
        assert price.delisting(1001) is None        # empty delist table

        # universe PIT
        assert uni.members_asof(price.calendar()[100]) == set(WINNERS)
        assert uni.members_asof(pd.Timestamp("1990-01-01")) == set()

        # fundamentals PIT (rdq gate)
        assert fund.fundamentals_asof(1001, pd.Timestamp("2015-01-05")) is None  # before first rdq
        late = fund.fundamentals_asof(1001, price.calendar()[-1])
        assert late is not None and len(late["quarterly_revenue"]) >= 2
        print("contract OK: 2 names, calendar=%d days, fundamentals PIT honored"
              % len(price.calendar()))


def test_engine_runs_on_wrds_cache():
    with tempfile.TemporaryDirectory() as d:
        _write_fixture_cache(d)
        price, uni, fund = load_wrds_providers(d)
        res = BacktestEngine(price, uni, fund,
                             config=BacktestConfig(max_positions=2)).run()
        assert len(res["daily"]) > 200
        assert np.isfinite(res["report"]["strategy"]["sharpe"])
        assert res["n_trades"] >= 1, "winners should produce at least one trade"
        print("engine-on-cache OK: trades=%d sharpe=%.2f"
              % (res["n_trades"], res["report"]["strategy"]["sharpe"]))


if __name__ == "__main__":
    test_wrds_providers_read_cache_and_satisfy_contract()
    test_engine_runs_on_wrds_cache()
    print("\nAll wrds_provider fixture tests passed.")
