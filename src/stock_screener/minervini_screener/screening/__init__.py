"""Stock screening package.

Pure rule logic (numpy/pandas only) lives in ``phase_indicators``, ``signal_engine``,
``benchmark`` and ``indicators`` and is re-exported below for convenience. The legacy
DB-backed value screener (``.screener``) is NOT imported here at all: it needs the live
data layer (SQLAlchemy, yfinance, ...), and with those extras installed a guarded import
silently dragged the whole dead layer into every process (~1.7s + sys.modules pollution).
Import ``.screener`` directly if it is ever needed.

NOTE (vendored — see PROVENANCE.md): modifications from the upstream MIT source are
limited to (1) ``from src.`` -> relative imports and (2) this import/re-export shim.
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
from .indicators import (
    calculate_rsi,
    calculate_sma,
    calculate_ema,
    detect_volume_spike,
    find_swing_lows,
)

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
    "calculate_rsi",
    "calculate_sma",
    "calculate_ema",
    "detect_volume_spike",
    "find_swing_lows",
]
