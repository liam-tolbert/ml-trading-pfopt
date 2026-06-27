"""Synthetic tests for the daily Minervini backtest harness.

Runs as a plain script (`python tests/test_backtest_daily.py`) or under pytest.
No disk/network. The load-bearing test is `test_engine_decisions_leak_free`:
running the engine on data truncated at D_mid must reproduce, bit-for-bit, the
decisions of a full run up to D_mid — the guard against any future-data leak.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stock_screener.backtest_daily.config import BacktestConfig          # noqa: E402
from src.stock_screener.backtest_daily.engine import BacktestEngine          # noqa: E402
from src.stock_screener.backtest_daily.portfolio import Portfolio            # noqa: E402
from src.stock_screener.backtest_daily.metrics import build_report           # noqa: E402
from src.stock_screener.backtest_daily.sizing import RiskBasedSizer          # noqa: E402
from src.stock_screener.backtest_daily.indicators_cache import IndicatorsCache  # noqa: E402
from src.stock_screener.backtest_daily.fundamentals_adapter import compustat_to_scorer_dict  # noqa: E402
from src.stock_screener.backtest_daily.providers import (                    # noqa: E402
    InMemoryPriceProvider, StaticUniverseProvider, NullFundamentalsProvider,
)
from src.stock_screener.backtest_daily.synthetic_provider import (           # noqa: E402
    make_synthetic, _ohlcv, _winner_close, _spy_close,
)
from src.ml_stock_prediction.backtest_lib import compute_metrics            # noqa: E402
from src.stock_screener.momentum_lib import capm_alpha_beta                 # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _single(close, permno=5001, delist=None, start="2015-01-02", spy_close=None):
    """One-name in-memory provider trio (price/universe/fundamentals) with a
    market benchmark. Default SPY is an uptrend so the regime gate allows buys."""
    n = len(close)
    cal = pd.bdate_range(start, periods=n)
    frame = _ohlcv(np.asarray(close, dtype=float), cal)
    if spy_close is None:
        spy_close = _spy_close(n, np.random.default_rng(123))
    spy_df = _ohlcv(np.asarray(spy_close, dtype=float), cal, band=0.004, vol=5e6)
    dels = {permno: delist} if delist else {}
    price = InMemoryPriceProvider({permno: frame}, spy_df, delistings=dels,
                                  calendar=cal, tickers={permno: "TST"})
    return price, StaticUniverseProvider.always([permno]), NullFundamentalsProvider()


def _eng(data, **kw):
    return BacktestEngine(data.price, data.universe, data.fundamentals,
                          config=BacktestConfig(**kw))


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_runs_end_to_end():
    data = make_synthetic(seed=7, periods=600, n_winners=8, n_losers=4, n_late=1)
    res = _eng(data, max_positions=10, risk_per_trade_pct=0.0125).run()
    assert len(res["daily"]) > 300
    assert res["n_trades"] > 0, "expected trades on synthetic winners"
    rep = res["report"]
    assert np.isfinite(rep["strategy"]["sharpe"])
    assert 0.0 <= rep["pct_time_in_cash"] <= 1.0
    print("end_to_end OK: trades=%d sharpe=%.2f cash%%=%.0f exposure%%=%.0f" % (
        res["n_trades"], rep["strategy"]["sharpe"],
        rep["pct_time_in_cash"] * 100, rep["average_exposure"] * 100))


def test_engine_decisions_leak_free():
    data = make_synthetic(seed=3, periods=560, n_winners=6, n_losers=3, n_late=1)
    cal = data.price.calendar()
    mid = cal[len(cal) // 2 + 30]

    full = _eng(data, max_positions=8, end=mid).run()
    trunc_price = data.price.truncate(mid)
    trunc = BacktestEngine(trunc_price, data.universe, data.fundamentals,
                           config=BacktestConfig(max_positions=8)).run()

    a, b = full["daily"]["equity"], trunc["daily"]["equity"]
    common = a.index.intersection(b.index)
    assert len(common) > 100
    pd.testing.assert_series_equal(a.loc[common], b.loc[common],
                                   check_exact=False, rtol=1e-9)
    assert full["n_trades"] == trunc["n_trades"]
    print("leak_free OK: %d common days, trades %d == %d"
          % (len(common), full["n_trades"], trunc["n_trades"]))


def test_slice_contract():
    data = make_synthetic(seed=1, periods=400, n_winners=4, n_losers=2, n_late=1)
    cache = IndicatorsCache.build(data.price, BacktestConfig())
    cal = data.price.calendar()
    for permno in data.price.permnos()[:3]:
        for t in (cal[250], cal[-1]):
            df = cache.ohlcv_upto(permno, t)
            if df is None or len(df) == 0:
                continue
            assert df.index.max() <= t
            assert float(df["Close"].iloc[-1]) == cache.close(permno, df.index.max())
    print("slice_contract OK")


def test_fundamentals_pit():
    q = pd.DataFrame({
        "datadate": pd.to_datetime(["2019-03-31", "2019-06-30", "2019-09-30",
                                    "2019-12-31", "2020-03-31"]),
        "rdq": pd.to_datetime(["2019-05-15", "2019-08-15", "2019-11-15",
                               "2020-02-15", "2020-05-15"]),
        "revtq": [100, 108, 117, 126, 136], "eps": [1.0, 1.1, 1.2, 1.3, 1.4],
        "invtq": [50, 50, 50, 50, 50],
    })
    asof = pd.Timestamp("2020-03-01")            # rdq<=asof => first 4 quarters
    d = compustat_to_scorer_dict(q, asof)
    assert len(d["quarterly_revenue"]) == 4      # the 2020-05-15 rdq quarter is hidden
    q2 = q.copy(); q2.loc[4, "revtq"] = 9999     # mutating the future quarter ...
    assert compustat_to_scorer_dict(q2, asof)["quarterly_revenue"] == d["quarterly_revenue"]
    assert compustat_to_scorer_dict(q, pd.Timestamp("2019-01-01")) is None  # before any rdq
    print("fundamentals_pit OK")


def test_universe_pit_and_delisting_forced_exit():
    rng = np.random.default_rng(9)
    close = _winner_close(360, 200, 150, 0, rng)            # rises, bought, keeps rising
    cal = pd.bdate_range("2015-01-02", periods=360)
    ddate = cal[340]
    price, uni, fund = _single(close, permno=7001, delist=(ddate, -0.85))
    # PIT eligibility: not a member before listing window? (always-member here) — assert
    # delisting forces the exit at the delisting return.
    res = BacktestEngine(price, uni, fund,
                         config=BacktestConfig(max_positions=3)).run()
    recs = res["blotter"].to_dict("records") if len(res["blotter"]) else []
    delist_trades = [t for t in recs if t["exit_reason"] == "DELIST"]
    assert len(delist_trades) == 1, recs
    assert delist_trades[0]["exit_price"] < delist_trades[0]["entry_price"]  # big loss
    print("delisting_forced_exit OK: ret=%.2f" % delist_trades[0]["ret"])


def test_buy_gate_downtrend_not_bought():
    close = 100.0 * 0.999 ** np.arange(400)                 # steady downtrend
    price, uni, fund = _single(close)
    res = BacktestEngine(price, uni, fund,
                         config=BacktestConfig(max_positions=5)).run()
    assert res["n_trades"] == 0
    print("buy_gate OK: downtrend never bought")


def test_cash_when_no_signals():
    rng = np.random.default_rng(4)
    close = _winner_close(400, 200, 150, 0, rng)            # would be buyable ...
    spy_down = 1000.0 * 0.999 ** np.arange(400)             # ... but market is bearish
    price, uni, fund = _single(close, spy_close=spy_down)
    res = BacktestEngine(price, uni, fund,
                         config=BacktestConfig(max_positions=5)).run()
    assert res["n_trades"] == 0
    eq = res["daily"]["equity"].to_numpy()
    assert np.allclose(eq, eq[0])                            # flat: all cash
    assert res["report"]["pct_time_in_cash"] == 1.0
    print("cash_when_no_signals OK")


def test_stop_exit():
    rng = np.random.default_rng(8)
    up = _winner_close(320, 200, 120, 0, rng)               # rises, gets bought
    crash = up[-1] * 0.55                                    # -45% gap, well below any entry stop
    close = np.concatenate([up, [crash], np.full(39, crash)])
    price, uni, fund = _single(close)
    res = BacktestEngine(price, uni, fund,
                         config=BacktestConfig(max_positions=3, scan_every_days=5)).run()
    recs = res["blotter"].to_dict("records") if len(res["blotter"]) else []
    reasons = [t["exit_reason"] for t in recs]
    assert "STOP" in reasons, reasons
    print("stop_exit OK")


def test_cost_model_closed_form():
    pf = Portfolio(100_000.0)
    spread, shares, price = 0.02, 100.0, 50.0
    cost = (spread / 2.0) * shares
    pf.enter(1, "2020-01-02", price, shares, stop=45.0, cost=cost, reason="BUY", phase=2)
    assert abs(pf.cash - (100_000.0 - shares * price - cost)) < 1e-9
    pf.exit(1, "2020-02-03", price, cost=cost, reason="SELL")   # flat price round trip
    assert abs(pf.cash - (100_000.0 - 2 * cost)) < 1e-9
    assert len(pf.trades) == 1 and pf.trades[0]["exit_reason"] == "SELL"
    print("cost_model OK")


def test_sizing_identity():
    cfg = BacktestConfig(sizing="risk", risk_per_trade_pct=0.01, max_positions=20)
    w = RiskBasedSizer().weight({"entry_price": 100.0, "stop_loss": 92.0}, 1e5, 0, cfg)
    risk_frac = (100.0 - 92.0) / 100.0
    assert abs(w - cfg.risk_per_trade_pct / risk_frac) < 1e-12
    assert abs(w * risk_frac - cfg.risk_per_trade_pct) < 1e-12   # loss-at-stop == risk%
    print("sizing_identity OK: w=%.4f" % w)


def test_report_matches_helpers():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(5)
    net = pd.Series(rng.normal(0.0005, 0.01, 300), index=idx)
    spy = pd.Series(rng.normal(0.0003, 0.009, 300), index=idx)
    daily = pd.DataFrame({"net": net, "spy_ret": spy,
                          "equity": (1 + net).cumprod() * 1e5,
                          "invested": 5e4, "n_positions": 5}, index=idx)
    rep = build_report(daily, [], BacktestConfig())
    m = compute_metrics(net, periods_per_year=252)
    cap = capm_alpha_beta(net, spy, periods_per_year=252)
    assert abs(rep["strategy"]["sharpe"] - m["sharpe"]) < 1e-9
    assert abs(rep["capm"]["beta"] - cap["beta"]) < 1e-9
    print("report_matches_helpers OK")


def test_reproducible():
    d1 = make_synthetic(seed=11, periods=420, n_winners=4, n_losers=2, n_late=1)
    d2 = make_synthetic(seed=11, periods=420, n_winners=4, n_losers=2, n_late=1)
    r1 = _eng(d1, max_positions=8).run()
    r2 = _eng(d2, max_positions=8).run()
    pd.testing.assert_frame_equal(r1["blotter"], r2["blotter"])
    print("reproducible OK")


if __name__ == "__main__":
    test_runs_end_to_end()
    test_engine_decisions_leak_free()
    test_slice_contract()
    test_fundamentals_pit()
    test_universe_pit_and_delisting_forced_exit()
    test_buy_gate_downtrend_not_bought()
    test_cash_when_no_signals()
    test_stop_exit()
    test_cost_model_closed_form()
    test_sizing_identity()
    test_report_matches_helpers()
    test_reproducible()
    print("\nAll backtest_daily synthetic tests passed.")
