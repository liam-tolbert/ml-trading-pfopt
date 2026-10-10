"""SEPA Cockpit: a Minervini screener where a person makes every trading decision.

In Minervini's books, Steps 2-4 of the method are judgment calls, so this package does
the mechanical work and leaves the judgment to the user. It scans a universe (a list of
tickers) for Trend-Template passers (Step 1) and highlights fundamental quality (Step 2).
The *user* judges the VCP on interactive charts (Step 3) and gets suggested entry levels
(Step 4).

Design rules:
- Only the rule functions from ``minervini_screener.screening`` MAY be reused. They have
  no file or network access and depend on numpy and pandas only, and that package holds
  nothing else.
- Live market data MUST come from this package's small yfinance wrapper (``data_feed``).
"""
