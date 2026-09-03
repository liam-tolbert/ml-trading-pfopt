# Vendored third-party code — Minervini stock screener

**Source:** sibling repo `../../../../stock-screener` (locally cloned), commit `397e555`
**Original author:** Ryan Hamby
**License:** MIT — (c) 2024 Ryan Hamby. See ./LICENSE. Retained per its terms.

Verbatim copy of that repo's `src/` Python package, vendored 2026-06-26 as the rule
engine for a survivorship-free historical backtest. This is NOT original work; the
copyright notice is preserved as MIT requires.

## Copied vs excluded
- **Copied (2026-06-26):** the whole `src/` library — `screening/`, `data/`, `analysis/`,
  `notifications/`. Most of it was later deleted — see "Removed from this copy" below.
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
   design). No function bodies were modified. (`.screener` stayed in the tree, importable
   directly, until the 2026-09-02 removal below.)
3. `screening/phase_indicators.py`'s local `calculate_sma` (a byte-duplicate of the one in
   `screening/indicators.py`, which is what the package `__init__` exports) was replaced with
   `from .indicators import calculate_sma` on 2026-07-22 (host-repo review item 32 — the two
   copies fed different consumers and could silently drift). Import-only dedup; return values
   identical (the surviving copy logs a warning on short series). `from .phase_indicators
   import calculate_sma` still resolves via the re-export.

## Removed from this copy (2026-09-02)

An audit found that **only four of the 27 vendored modules were reachable** from this repo —
everything the host uses arrives through `screening/__init__.py`. The rest was deleted rather
than carried as unreachable weight that still had to be kept out of the runtime image by hand,
mis-described itself as importable, and answered every repo-wide grep. MIT permits
modification; the notice and this record are what it requires.

**Deleted:** `data/` (11 modules — the SQLAlchemy/yfinance live-fetch layer),
`notifications/` (4 — email/Slack/scheduler), `analysis/` (2 — `position_manager`),
`screening/screener.py` (the legacy DB-backed value screener), `screening/quant_engine.py`,
`screening/batch_processor.py`, `screening/optimized_batch_processor.py`.

**Why each was unreachable:**
- Their upstream entry points (`run_optimized_scan.py`, `manage_positions.py`,
  `automated_position_report.py`) were never vendored — see "Copied vs excluded" — so nothing
  could call the batch processors, the quant engine or the position manager.
- `notifications/` had been **import-broken since the 2026-07-22 change above**:
  `scheduler.py` does `from ..screening import screen_candidates`, and that name stopped being
  re-exported when the `.screener` import was removed. Importing `notifications` raised
  ImportError from then on, so the package cannot have been in use.
- The cockpit substitutes its own VCP detector (`cockpit/vcp.py`) and its own data layer
  (`cockpit/data_feed.py`) for the vendored equivalents.

**Kept:** `screening/{__init__,phase_indicators,signal_engine,benchmark,indicators}.py`, plus
`LICENSE` and this file. **No function body has ever been modified** — the only edits remain
the three documented above, plus shrinking the `screening/__init__.py` re-export list to the
names this repo actually consumes.

**Recovering anything deleted:** it is all in this repo's git history, and upstream `397e555`
is the authoritative copy.

## Backtest-relevant modules (pure rule logic, no live fetching)
- `screening/phase_indicators.py` — classify_phase, validate_minervini_trend_template,
  detect_vcp_pattern, detect_breakout, calculate_relative_strength
- `screening/signal_engine.py` — score_buy_signal, score_sell_signal, calculate_stop_loss
- `screening/benchmark.py` — analyze_spy_trend, calculate_market_breadth, should_generate_signals
- `screening/indicators.py` — RSI/SMA/EMA/volume helpers

`data/`, `notifications/`, `analysis/`, and the batch processors / `quant_engine.py` were
live-fetch + notify orchestration, needed by neither the backtest nor the cockpit; they
were deleted on 2026-09-02 (above).
