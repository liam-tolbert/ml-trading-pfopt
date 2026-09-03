"""Stock screening package.

Pure rule logic (numpy/pandas only) lives in ``phase_indicators``, ``signal_engine``,
``benchmark`` and ``indicators``, and is re-exported below. These four modules are the
whole of this package: the DB-backed value screener, the batch processors, the quant
engine and the live ``data``/``notifications``/``analysis`` layers were deleted from
this copy because nothing here could reach them and they needed extras (SQLAlchemy,
yfinance, slack) that the runtime does not install. See PROVENANCE.md.

Only names with a consumer in this repo are exported. An unexported helper is still
importable from its own module — that is deliberate, so this list stays a map of what
is actually used.

NOTE (vendored — see PROVENANCE.md): modifications from the upstream MIT source are
limited to (1) ``from src.`` -> relative imports and (2) this import/re-export shim.
No function body was ever changed.
"""

# --- Pure rule logic: numpy/pandas only, always importable ------------------
from .phase_indicators import (
    classify_phase,
    validate_minervini_trend_template,
    detect_vcp_pattern,
    detect_breakout,
    calculate_relative_strength,
)
from .signal_engine import (
    score_buy_signal,
    score_sell_signal,
    calculate_stop_loss,
)
from .benchmark import (
    analyze_spy_trend,
    calculate_market_breadth,
    should_generate_signals,
)
from .indicators import calculate_sma

__all__ = [
    # pure rule logic (Minervini screen)
    "classify_phase",
    "validate_minervini_trend_template",
    "detect_vcp_pattern",
    "detect_breakout",
    "calculate_relative_strength",
    "score_buy_signal",
    "score_sell_signal",
    "calculate_stop_loss",
    "analyze_spy_trend",
    "calculate_market_breadth",
    "should_generate_signals",
    # technical-indicator helpers
    "calculate_sma",
]
