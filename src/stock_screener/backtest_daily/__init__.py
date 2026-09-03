"""Event-driven DAILY backtest harness for the (vendored) Minervini screener.

This package drives the vendored screener's pure rule functions
(`stock_screener.minervini_screener.screening`) over point-in-time, daily price +
fundamentals data, simulating Minervini *as he trades it*: enter on a buy signal,
exit on a stop / sell-signal / regime flip / delisting, and sit in CASH when nothing
qualifies. It is decoupled from any data source via provider interfaces; ship with a
synthetic provider so it runs/tests today, with a stubbed WRDS/CRSP provider for the
survivorship-free data pull.

Leak-safety is the #1 correctness property: every rule call receives only data with
index <= the decision date, and fundamentals lagged to their report (rdq) date.

**This module deliberately re-exports NOTHING.** Import submodules by their full path
instead. The cockpit test suite pulls `synthetic_provider.make_synthetic` for its
offline price fixture, and that fixture ships in the Pi's runtime image; a re-export
here would make importing it drag in `engine` -> `metrics` -> `momentum_lib` +
`ml_stock_prediction.backtest_lib`, forcing two parked research tracks into the
production image to satisfy one test helper.
"""
