"""SEPA Cockpit — a human-in-the-loop Minervini screener.

The automated Minervini selection has no out-of-sample alpha (see
``src/stock_screener/HANDOFF.md``); its edge in Steps 2-4 is discretionary. This
package does the mechanical work — scan a universe for Trend-Template passers
(Step 1) and highlight fundamental quality (Step 2) — then hands the *user*
interactive charts to judge the VCP (Step 3) and advisory entry levels (Step 4).

Design rules:
- Reuse ONLY the pure rule functions from ``minervini_screener.screening`` (they
  import on numpy/pandas; the package's ``data/`` layer is NOT imported because its
  ``__init__`` eager-loads SQLAlchemy, which is absent from the ``ml-trading`` env).
- All live data comes from this package's own thin yfinance layer (``data_feed``).
"""
