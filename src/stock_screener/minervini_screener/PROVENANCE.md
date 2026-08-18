# Vendored third-party code — Minervini stock screener

**Source:** sibling repo `../../../../stock-screener` (locally cloned), commit `397e555`
**Original author:** Ryan Hamby
**License:** MIT — (c) 2024 Ryan Hamby. See ./LICENSE. Retained per its terms.

Verbatim copy of that repo's `src/` Python package, vendored 2026-06-26 as the rule
engine for a survivorship-free historical backtest. This is NOT original work; the
copyright notice is preserved as MIT requires.

## Copied vs excluded
- **Copied:** the whole `src/` library — `screening/`, `data/`, `analysis/`, `notifications/`.
- **Excluded (code-only request):** `tests/`, `.github/` CI/CD, `scripts/`, `examples/`,
  `*.md` docs, `config.yaml`, `.env.example`, `requirements.txt`, `*.sh`, and the root
  live-scan entry points (`run_optimized_scan.py`, `manage_positions.py`,
  `automated_position_report.py`).

## Changes from the original (no business logic changed)
1. Absolute imports `from src.X` were rewritten to relative `from ..X` so the package works
   mounted here instead of at the original `src/`.
2. `screening/__init__.py` imports and re-exports ONLY the pure rule modules (numpy/pandas).
   The legacy DB-backed `.screener` import was initially wrapped in `try/except ImportError`,
   then REMOVED entirely on 2026-07-22 (host-repo review item 20): with SQLAlchemy present
   (a transitive dep of wrds) the guard silently succeeded and every host process eagerly
   imported the dead data layer (~1.7s, sys.modules pollution). The only in-repo consumer of
   those re-exports was the equally-unused vendored `notifications/scheduler.py` (inert by
   design). `.screener` remains in the tree and importable directly. No files were deleted;
   no function bodies were modified.
3. `screening/phase_indicators.py`'s local `calculate_sma` (a byte-duplicate of the one in
   `screening/indicators.py`, which is what the package `__init__` exports) was replaced with
   `from .indicators import calculate_sma` on 2026-07-22 (host-repo review item 32 — the two
   copies fed different consumers and could silently drift). Import-only dedup; return values
   identical (the surviving copy logs a warning on short series). `from .phase_indicators
   import calculate_sma` still resolves via the re-export.

## Backtest-relevant modules (pure rule logic, no live fetching)
- `screening/phase_indicators.py` — classify_phase, validate_minervini_trend_template,
  detect_vcp_pattern, detect_breakout, calculate_relative_strength
- `screening/signal_engine.py` — score_buy_signal, score_sell_signal, calculate_stop_loss
- `screening/benchmark.py` — analyze_spy_trend, calculate_market_breadth, should_generate_signals
- `screening/indicators.py` — RSI/SMA/EMA/volume helpers

`data/`, `notifications/`, `analysis/`, and the batch processors / `quant_engine.py` are
live-fetch + notify orchestration: kept for completeness, not needed for the backtest.
