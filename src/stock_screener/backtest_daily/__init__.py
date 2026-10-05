"""Synthetic market data for the cockpit's offline tests.

What is left of the daily backtest harness: the provider contracts, the point-in-time
fundamentals adapter and the seeded synthetic provider. `tests/cockpit/_common.py`
builds its price fixture from `synthetic_provider.make_synthetic`, and that fixture
ships in the Pi's runtime image because the test gate runs there.

**This module MUST re-export nothing.** Import submodules by their full path, so the
fixture's import closure stays these files and numpy/pandas.
"""
