"""Stock screening package.

Pure rule logic (numpy/pandas only) lives in ``phase_indicators``, ``signal_engine``,
``benchmark`` and ``indicators`` and is re-exported below for convenience. The legacy
DB-backed value screener (``.screener``) depends on the live data layer (SQLAlchemy,
yfinance, ...); its import is GUARDED so the rule modules stay importable in
environments without those extras (e.g. the backtest harness env).

NOTE (vendored — see PROVENANCE.md): modifications from the upstream MIT source are
limited to (1) ``from src.`` -> relative imports and (2) this guard + re-exports.
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

# --- Legacy value screener: needs the live data layer (SQLAlchemy). Guard so a
#     missing extra doesn't break importing the pure rule modules above. --------
try:
    from .screener import (
        calculate_value_score,
        detect_support_levels,
        calculate_support_score,
        screen_candidates,
    )
    __all__ += [
        "calculate_value_score",
        "detect_support_levels",
        "calculate_support_score",
        "screen_candidates",
    ]
except ImportError:
    # data-layer extras not installed; pure rule logic remains available.
    pass
