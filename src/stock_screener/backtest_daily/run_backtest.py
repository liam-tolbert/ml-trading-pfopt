"""Run the daily Minervini backtest on synthetic data and print a report.

    python src/stock_screener/backtest_daily/run_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.stock_screener.backtest_daily.config import BacktestConfig          # noqa: E402
from src.stock_screener.backtest_daily.engine import BacktestEngine          # noqa: E402
from src.stock_screener.backtest_daily.synthetic_provider import make_synthetic  # noqa: E402


def main():
    data = make_synthetic(seed=7)
    cfg = BacktestConfig(max_positions=10, risk_per_trade_pct=0.0125, scan_every_days=5)
    res = BacktestEngine(data.price, data.universe, data.fundamentals, config=cfg).run()

    daily, rep = res["daily"], res["report"]
    s, spy, capm, tr = rep["strategy"], rep["spy_buyhold"], rep["capm"], rep["trades"]

    print("=" * 64)
    print("DAILY MINERVINI SCREENER BACKTEST  (synthetic data)")
    print("=" * 64)
    print(f"window   : {daily.index.min().date()} -> {daily.index.max().date()}  "
          f"({len(daily)} trading days)")
    print(f"strategy : CAGR {s['cagr']*100:6.2f}%  vol {s['ann_vol']*100:5.1f}%  "
          f"Sharpe {s['sharpe']:.2f}  maxDD {s['max_drawdown']*100:6.1f}%")
    print(f"SPY B&H  : CAGR {spy['cagr']*100:6.2f}%  vol {spy['ann_vol']*100:5.1f}%  "
          f"Sharpe {spy['sharpe']:.2f}  maxDD {spy['max_drawdown']*100:6.1f}%")
    print(f"CAPM     : alpha {capm['alpha_ann']*100:+.2f}%/yr (t {capm['t_alpha']:+.2f})  "
          f"beta {capm['beta']:.2f}")
    print(f"trades   : n {tr['n_trades']}  win {tr['win_rate']*100:.0f}%  "
          f"payoff {tr['payoff_ratio']:.2f}  avg hold {tr['avg_holding_days']:.0f}d")
    print(f"exposure : avg {rep['average_exposure']*100:.0f}%  "
          f"time in cash {rep['pct_time_in_cash']*100:.0f}%")
    print("=" * 64)


if __name__ == "__main__":
    main()
