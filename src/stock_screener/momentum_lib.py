"""Momentum / trend-following signal library — pure, data-agnostic core.

Phase 0 of the momentum-validation effort (see docs/HANDOFF.md and the plan).
This module deliberately contains NO data fetching and NO backtest loop. It only
produces leak-free, point-in-time momentum FEATURES from weekly OHLCV and exposes
rules-based SCORERS that masquerade as a model (``predict_proba(X)[:, 1]``) so the
existing, tested ``backtest_lib`` engine can rank/trade them with ZERO changes.

The signal ideas are ported from the sibling ``stock-screener`` repo
(``src/screening/phase_indicators.py`` + ``signal_engine.py``) — Minervini
Stage-2 trend structure, relative-strength slope vs SPY, distance-from-52-week
high, breakout/volume — but adapted from DAILY bars to this project's WEEKLY
(W-FRI) panel (windows divided by ~5) and made fully vectorised/point-in-time so
every feature at date t depends only on data at or before t.

Leak-safety is the #1 correctness property here: a forward-looking feature makes
momentum look falsely good. Guarantees enforced below:
  - every rolling window is TRAILING (no ``center=True``);
  - only positive ``.shift(k)`` is ever used (never ``.shift(-k)``);
  - the 52-week high uses ``High.rolling(52).max()`` (NOT a whole-series max);
  - SPY is aligned by date with ``method='ffill'`` (past only — never bfill);
  - no feature reads the realized forward return (``Returns-future-1wk``).
``tests/test_momentum_lib.py::test_features_leak_free`` is the regression guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Window constants — DAILY screener windows converted to WEEKLY (÷~5)
# --------------------------------------------------------------------------- #
SMA_SHORT = 10   # ~50-day SMA
SMA_MID = 30     # ~150-day SMA
SMA_LONG = 40    # ~200-day SMA
HIGH_52W_WEEKS = 52
RS_LEVEL_WEEKS = 13      # ~63-day RS level horizon (unused as a column but documents intent)
RS_SLOPE_WEEKS = 13      # RS-slope lookback
SMA_SLOPE_WEEKS = 5      # ~20-day SMA-slope lookback
VOL_AVG_WEEKS = 10       # ~20-day volume average
BREAKOUT_WEEKS = 20      # base-high lookback for breakout
MOM_LOOKBACK = 12        # classic momentum lookback (weeks)
MOM_SKIP = 1             # skip the most recent week (dodge 1-wk reversal)

# The exact feature columns produced by create_momentum_features and consumed by
# the scorers. Export once and reuse for both the panel build and every engine
# call so the column set can never drift between them.
MOM_FEATURE_COLS = [
    "mom_12_1",
    "above_sma10", "above_sma40", "sma_stack",
    "dist_sma10", "dist_sma40",
    "slope_sma10", "slope_sma40",
    "dist_52w_high",
    "rs_level", "rs_slope",
    "vol_ratio", "breakout",
    "phase",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Trailing OLS slope of ``series`` over the last ``window`` points, expressed
    as percent-per-step relative to the window mean (so it is scale-free across
    tickers). Returns a FULL point-in-time history (one slope per date), unlike
    the screener's ``calculate_slope`` which returns only a tail scalar.

    Closed form for a fixed abscissa x = 0..w-1: slope = Σ(x-x̄)·y / Σ(x-x̄)².
    The numerator is a fixed-weight dot product over the trailing window; we
    compute it with ``rolling().apply(raw=True)``. NaN for the first w-1 rows.
    Trailing-only → leak-free.
    """
    w = int(window)
    if w < 2:
        raise ValueError("window must be >= 2")
    k = np.arange(w, dtype=float)
    k -= k.mean()                      # centered abscissa
    sxx = float(np.dot(k, k))          # Σ(x-x̄)²  (constant)

    def _slope(arr):
        return float(np.dot(arr, k) / sxx)

    raw_slope = series.rolling(w, min_periods=w).apply(_slope, raw=True)
    mean = series.rolling(w, min_periods=w).mean()
    out = raw_slope / mean.replace(0.0, np.nan) * 100.0
    return out


def _vectorized_phase(close, sma10, sma40, slope_sma10, dist_sma10) -> pd.Series:
    """Weekly 4-phase classification (priority 4→2→3→1), vectorised.

    Mirrors ``stock-screener``'s ``classify_phase`` with weekly SMAs:
      P4 downtrend : close<sma10 & close<sma40 & sma10<sma40
      P2 uptrend   : close>sma10 & sma10>sma40 & slope_sma10>0
      P3 distrib.  : close>sma10 & dist_sma10>25
      P1 base      : otherwise
    Warmup rows (sma40 NaN) → phase 0.
    """
    close = np.asarray(close, dtype=float)
    sma10 = np.asarray(sma10, dtype=float)
    sma40 = np.asarray(sma40, dtype=float)
    slope_sma10 = np.asarray(slope_sma10, dtype=float)
    dist_sma10 = np.asarray(dist_sma10, dtype=float)

    p4 = (close < sma10) & (close < sma40) & (sma10 < sma40)
    p2 = (close > sma10) & (sma10 > sma40) & (slope_sma10 > 0)
    p3 = (close > sma10) & (dist_sma10 > 25)
    phase = np.select([p4, p2, p3], [4, 2, 3], default=1).astype(float)
    phase[np.isnan(sma40) | np.isnan(slope_sma10)] = 0.0
    return phase


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def create_momentum_features(ohlcv: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    """One ticker's weekly OHLCV (+ weekly SPY Close) -> point-in-time momentum
    features, indexed exactly like ``ohlcv`` (NOT dropna'd here; the caller drops
    warmup rows after assembling the panel). See module docstring for the
    leak-safety guarantees. Columns produced == ``MOM_FEATURE_COLS``.
    """
    close = ohlcv["Close"].astype(float)
    high = ohlcv["High"].astype(float)
    volume = ohlcv["Volume"].astype(float)

    # SPY aligned by DATE, forward-filled (past only). Never bfill.
    spy_aligned = spy_close.reindex(close.index, method="ffill").astype(float)

    sma10 = close.rolling(SMA_SHORT, min_periods=SMA_SHORT).mean()
    sma30 = close.rolling(SMA_MID, min_periods=SMA_MID).mean()
    sma40 = close.rolling(SMA_LONG, min_periods=SMA_LONG).mean()

    out = pd.DataFrame(index=ohlcv.index)

    # classic 12-1 total-return momentum (skip the most recent week)
    out["mom_12_1"] = close.shift(MOM_SKIP) / close.shift(MOM_SKIP + MOM_LOOKBACK) - 1.0

    # trend structure
    out["above_sma10"] = (close > sma10).astype(float)
    out["above_sma40"] = (close > sma40).astype(float)
    out["sma_stack"] = ((sma10 > sma30) & (sma30 > sma40)).astype(float)
    out["dist_sma10"] = (close / sma10 - 1.0) * 100.0
    out["dist_sma40"] = (close / sma40 - 1.0) * 100.0
    out["slope_sma10"] = _rolling_slope(sma10, SMA_SLOPE_WEEKS)
    out["slope_sma40"] = _rolling_slope(sma40, SMA_SLOPE_WEEKS)

    # distance from the trailing 52-week high (<=0; near 0 == near the high)
    high_52w = high.rolling(HIGH_52W_WEEKS, min_periods=20).max()
    out["dist_52w_high"] = (close / high_52w - 1.0) * 100.0

    # relative strength vs SPY
    rs_level = close / spy_aligned * 100.0
    out["rs_level"] = rs_level
    out["rs_slope"] = _rolling_slope(rs_level, RS_SLOPE_WEEKS)

    # volume + breakout (both exclude the current bar from their baseline)
    out["vol_ratio"] = volume / volume.shift(1).rolling(VOL_AVG_WEEKS).mean()
    base_high = close.rolling(BREAKOUT_WEEKS, min_periods=10).max().shift(1)
    out["breakout"] = (close > base_high).astype(float)

    # phase (uses sma10/40 + slope_sma10 + dist_sma10 already computed)
    out["phase"] = _vectorized_phase(close, sma10, sma40,
                                      out["slope_sma10"], out["dist_sma10"])

    # the boolean-cast columns are 0.0 even during warmup; NaN them where the
    # underlying SMA is undefined so the panel dropna removes true warmup rows.
    out.loc[sma10.isna(), ["above_sma10", "dist_sma10"]] = np.nan
    out.loc[sma40.isna(), ["above_sma40", "dist_sma40", "sma_stack"]] = np.nan
    return out[MOM_FEATURE_COLS]


def compute_beta(asset_log_ret: pd.Series, mkt_log_ret: pd.Series,
                 window: int = 26) -> pd.Series:
    """Trailing rolling beta = Cov(asset, mkt) / Var(mkt) over ``window`` weeks of
    LOG returns. Point-in-time (NaN until ``window`` observations).

    Default window=26 to name-match the engine's ``Beta_26wk`` column. NOTE the
    notebook's ``compute_beta`` defaulted to window=20; we use 26 here and report
    it as such. Either is point-in-time and unbiased; only the smoothing differs.
    """
    mkt = mkt_log_ret.reindex(asset_log_ret.index, method="ffill")
    cov = asset_log_ret.rolling(window, min_periods=window).cov(mkt)
    var = mkt.rolling(window, min_periods=window).var()
    return cov / var.replace(0.0, np.nan)


# --------------------------------------------------------------------------- #
# Scorers — stateless, continuous, UNGATED (mirror tests' StubModel)
# --------------------------------------------------------------------------- #
# Both expose predict_proba(X) -> np.column_stack([1-p, p]) of shape (n, 2), with
# p a continuous momentum-strength score in (0, 1) that is monotone in trend
# strength. They are UNGATED on purpose: the beta-neutral L/S shorts the bottom
# ranks, so a hard "phase!=2 -> 0" gate (Minervini's long-only filter) would tie
# the entire short leg at 0 and produce a degenerate hedge. The engine only needs
# ranks for top_n/quantile; the squash sets where buy_threshold=0.5 falls.

CLASSIC_SCALE = 10.0  # sigmoid spread for the 12-1 scorer; rank-preserving

# Screener-composite weights (echo signal_engine.score_buy_signal's emphasis:
# trend dominant, then RS + 52w-high proximity + breakout/volume). Fundamentals,
# R/R and stops are dropped (no fundamentals in this panel; R/R & stops are trade
# management, not a cross-sectional signal).
W_TREND, W_SLOPE, W_RS, W_HIGH, W_BRK, W_EXTEND = 0.35, 0.20, 0.20, 0.15, 0.10, 0.10
COMPOSITE_GAIN = 6.0  # sigmoid gain around s=0.5


class ClassicMomentumScorer:
    """Academic 12-week-minus-1-week total-return cross-sectional momentum.
    p = sigmoid(scale * mom_12_1); NaN momentum -> neutral 0.5 (rank-neutral)."""

    def __init__(self, scale: float = CLASSIC_SCALE):
        self.scale = scale

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        mom = pd.to_numeric(X["mom_12_1"], errors="coerce").to_numpy(dtype=float)
        p = _sigmoid(self.scale * mom)
        p = np.where(np.isnan(mom), 0.5, p)
        return np.column_stack([1.0 - p, p])


class ScreenerCompositeScorer:
    """Ungated screener-style composite: trend structure + SMA slope + RS slope +
    52w-high proximity + breakout/volume, minus an over-extension penalty.
    Each sub-term is bounded; NaN inputs fall back to a neutral contribution."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        def col(name):
            return pd.to_numeric(X[name], errors="coerce").to_numpy(dtype=float)

        sma_stack = np.nan_to_num(col("sma_stack"), nan=0.0)
        above10 = np.nan_to_num(col("above_sma10"), nan=0.0)
        above40 = np.nan_to_num(col("above_sma40"), nan=0.0)
        slope10 = col("slope_sma10")
        rs_slope = col("rs_slope")
        dist_high = col("dist_52w_high")
        vol_ratio = col("vol_ratio")
        breakout = np.nan_to_num(col("breakout"), nan=0.0)
        dist10 = col("dist_sma10")

        trend = 0.5 * sma_stack + 0.25 * above10 + 0.25 * above40          # 0..1
        slope = np.clip(np.nan_to_num(slope10, nan=0.0), -1, 1) * 0.5 + 0.5
        rs = np.clip(np.nan_to_num(rs_slope, nan=0.0), -1, 1) * 0.5 + 0.5
        highp = np.clip((np.nan_to_num(dist_high, nan=-25.0) + 25.0) / 25.0, 0, 1)
        brk = 0.5 * breakout + 0.5 * np.clip(
            (np.nan_to_num(vol_ratio, nan=1.0) - 1.0) / 1.0, 0, 1)
        extend = np.clip((np.nan_to_num(dist10, nan=0.0) - 25.0) / 15.0, 0, 1)

        s = (W_TREND * trend + W_SLOPE * slope + W_RS * rs
             + W_HIGH * highp + W_BRK * brk - W_EXTEND * extend)
        p = _sigmoid(COMPOSITE_GAIN * (s - 0.5))
        return np.column_stack([1.0 - p, p])


def momentum_fit_fn(scorer):
    """Return a no-op ``fit_fn(X, y) -> scorer`` compatible with the backtest
    engine. The scorer is rules-based, so there is nothing to train and the
    engine's refit cadence is irrelevant — every "fit" returns the same scorer."""
    def fit_fn(X, y):
        return scorer
    return fit_fn


# --------------------------------------------------------------------------- #
# CAPM alpha/beta with HAC (Newey-West) standard errors
# --------------------------------------------------------------------------- #
def capm_alpha_beta(net_weekly: pd.Series, spy_weekly: pd.Series,
                    periods_per_year: int = 52) -> dict:
    """Regress weekly net returns on SPY with Newey-West (HAC) covariance and
    return annualized alpha, beta, and the HAC t-stat on alpha. This is the
    decomposition the HANDOFF says lived in a stripped notebook cell; it is not
    in backtest_lib.py, so it lives here.

    Regressing a beta-hedged L/S series on SPY also soaks any residual hedge-miss
    beta into the beta term -> robust to the 26-vs-20 beta-window choice.
    """
    import statsmodels.api as sm

    y = pd.Series(net_weekly).dropna()
    x = pd.Series(spy_weekly).reindex(y.index)
    df = pd.concat([y.rename("y"), x.rename("spy")], axis=1).dropna()
    n = len(df)
    nan = float("nan")
    if n < 10:
        return {"alpha_ann": nan, "beta": nan, "t_alpha": nan,
                "alpha_weekly": nan, "n": n, "maxlags": 0}

    maxlags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    X = sm.add_constant(df["spy"])
    fit = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    alpha_w = float(fit.params["const"])
    beta = float(fit.params["spy"])
    return {
        "alpha_ann": (1.0 + alpha_w) ** periods_per_year - 1.0,
        "beta": beta,
        "t_alpha": float(fit.tvalues["const"]),
        "alpha_weekly": alpha_w,
        "n": n,
        "maxlags": maxlags,
    }
