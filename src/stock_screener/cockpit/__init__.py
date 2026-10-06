"""SEPA Cockpit: a human-in-the-loop Minervini screener.

Steps 2-4 of the method are discretionary in the books, so this package does the
mechanical work and leaves the judgment to the user. It scans a universe for
Trend-Template passers (Step 1) and highlights fundamental quality (Step 2). The *user*
judges the VCP on interactive charts (Step 3) and gets advisory entry levels (Step 4).

Design rules:
- Only the pure rule functions from ``minervini_screener.screening`` MAY be reused.
  They import on numpy/pandas alone, and they are all that is left of that package.
- Live market data MUST come from this package's thin yfinance layer (``data_feed``).
"""
