"""Synthetic-data tests for src/momentum_lib.

Runs as a plain script (`python tests/test_momentum_lib.py`) or under pytest.
No disk / network access — all data is generated. The single most important test
is `test_features_leak_free`: it is the regression guard against a forward-looking
feature, which would make momentum look falsely good in the backtest.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import momentum_lib as ml  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _dates(n):
    return pd.date_range("2015-01-02", periods=n, freq="W-FRI")


def _ohlcv(close, dates, high=None, volume=None):
    close = np.asarray(close, dtype=float)
    high = close if high is None else np.asarray(high, dtype=float)
    volume = (np.full(len(close), 1_000_000.0) if volume is None
              else np.asarray(volume, dtype=float))
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": close, "Close": close, "Volume": volume},
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def _flat_spy(dates):
    return pd.Series(100.0, index=pd.DatetimeIndex(dates, name="Date"))


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_uptrend_is_phase2_and_scores_high():
    n = 70
    dates = _dates(n)
    close = 100.0 * 1.01 ** np.arange(n)
    feats = ml.create_momentum_features(_ohlcv(close, dates), _flat_spy(dates))
    last = feats.iloc[-1]
    assert last["phase"] == 2.0, last["phase"]
    assert last["sma_stack"] == 1.0
    assert last["above_sma10"] == 1.0 and last["above_sma40"] == 1.0
    assert abs(last["dist_52w_high"]) < 1e-6, last["dist_52w_high"]  # at the high

    X = feats.iloc[[-1]]
    for scorer in (ml.ClassicMomentumScorer(), ml.ScreenerCompositeScorer()):
        p = scorer.predict_proba(X)[:, 1][0]
        assert p > 0.7, (type(scorer).__name__, p)
    print("test_uptrend_is_phase2_and_scores_high OK — phase 2, both scorers > 0.7")


def test_downtrend_scores_low():
    n = 70
    dates = _dates(n)
    close = 100.0 * 0.99 ** np.arange(n)
    feats = ml.create_momentum_features(_ohlcv(close, dates), _flat_spy(dates))
    last = feats.iloc[-1]
    assert last["phase"] == 4.0, last["phase"]
    X = feats.iloc[[-1]]
    for scorer in (ml.ClassicMomentumScorer(), ml.ScreenerCompositeScorer()):
        p = scorer.predict_proba(X)[:, 1][0]
        assert p < 0.3, (type(scorer).__name__, p)
    print("test_downtrend_scores_low OK — phase 4, both scorers < 0.3 "
          "(ungated short leg gets weak names)")


def test_features_leak_free():
    """Feature at row t depends only on rows <= t. Computing on a truncated series
    must not change any earlier row -> the guard against center=True, .shift(-k),
    whole-series .max(), or a bfill SPY align."""
    rng = np.random.default_rng(7)
    n = 120
    dates = _dates(n)
    rets = rng.normal(0.001, 0.025, size=n)
    close = 100.0 * np.cumprod(1.0 + rets)
    high = close * (1.0 + np.abs(rng.normal(0, 0.01, size=n)))
    vol = rng.uniform(5e5, 2e6, size=n)
    spy = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0008, 0.015, size=n)),
                    index=pd.DatetimeIndex(dates, name="Date"))

    full = ml.create_momentum_features(_ohlcv(close, dates, high, vol), spy)
    k = 95
    trunc = ml.create_momentum_features(
        _ohlcv(close[:k], dates[:k], high[:k], vol[:k]), spy)

    pd.testing.assert_frame_equal(full.iloc[:k], trunc, check_exact=False, rtol=1e-9)
    print("test_features_leak_free OK — first", k,
          "rows identical with future bars truncated (point-in-time)")


def test_scorer_predict_proba_shape_and_monotone():
    rows = 5
    base = {c: np.full(rows, np.nan) for c in ml.MOM_FEATURE_COLS}
    # constant strong structure; vary the continuous momentum drivers upward
    base["sma_stack"] = np.ones(rows)
    base["above_sma10"] = np.ones(rows)
    base["above_sma40"] = np.ones(rows)
    base["dist_sma10"] = np.full(rows, 5.0)
    base["dist_sma40"] = np.full(rows, 10.0)
    base["slope_sma40"] = np.full(rows, 0.5)
    base["dist_52w_high"] = np.full(rows, -5.0)
    base["rs_level"] = np.full(rows, 100.0)
    base["vol_ratio"] = np.full(rows, 1.2)
    base["breakout"] = np.ones(rows)
    base["phase"] = np.full(rows, 2.0)
    base["mom_12_1"] = np.linspace(-0.2, 0.2, rows)     # increasing
    base["slope_sma10"] = np.linspace(-1.0, 1.0, rows)  # increasing
    base["rs_slope"] = np.linspace(-1.0, 1.0, rows)     # increasing
    X = pd.DataFrame(base)[ml.MOM_FEATURE_COLS]

    for scorer in (ml.ClassicMomentumScorer(), ml.ScreenerCompositeScorer()):
        proba = scorer.predict_proba(X)
        assert proba.shape == (rows, 2), (type(scorer).__name__, proba.shape)
        assert np.allclose(proba.sum(axis=1), 1.0)
        p = proba[:, 1]
        assert np.all((p > 0.0) & (p < 1.0))
        assert np.all(np.diff(p) > 0), (type(scorer).__name__, p)

    # classic: NaN momentum -> neutral 0.5
    Xn = X.copy()
    Xn.loc[Xn.index[0], "mom_12_1"] = np.nan
    p0 = ml.ClassicMomentumScorer().predict_proba(Xn)[:, 1][0]
    assert abs(p0 - 0.5) < 1e-9, p0
    print("test_scorer_predict_proba_shape_and_monotone OK — (n,2), rows sum to 1, "
          "monotone in trend, NaN->0.5")


def test_compute_beta_recovers_known_beta():
    rng = np.random.default_rng(11)
    n = 60
    dates = _dates(n)
    mkt = pd.Series(rng.normal(0.0, 0.02, size=n), index=pd.DatetimeIndex(dates))
    true_beta = 1.4
    asset = true_beta * mkt
    beta = ml.compute_beta(asset, mkt, window=26)
    assert abs(beta.iloc[-1] - true_beta) < 1e-6, beta.iloc[-1]
    assert np.isnan(beta.iloc[24]), "beta must be NaN before `window` obs"
    print("test_compute_beta_recovers_known_beta OK — recovered beta =",
          round(float(beta.iloc[-1]), 4))


def test_capm_alpha_beta_recovers_known_alpha():
    rng = np.random.default_rng(13)
    n = 250
    dates = _dates(n)
    spy = pd.Series(rng.normal(0.0015, 0.02, size=n), index=pd.DatetimeIndex(dates))
    alpha_w, beta = 0.001, 1.1
    strat = alpha_w + beta * spy + rng.normal(0.0, 0.002, size=n)
    strat = pd.Series(strat, index=pd.DatetimeIndex(dates))

    res = ml.capm_alpha_beta(strat, spy)
    assert abs(res["beta"] - beta) < 0.05, res["beta"]
    assert abs(res["alpha_weekly"] - alpha_w) < 5e-4, res["alpha_weekly"]
    expected_ann = (1.0 + alpha_w) ** 52 - 1.0
    assert abs(res["alpha_ann"] - expected_ann) < 0.03, res["alpha_ann"]
    assert np.isfinite(res["t_alpha"])

    # zero-alpha control: alpha estimate should be ~0
    strat0 = beta * spy + rng.normal(0.0, 0.002, size=n)
    res0 = ml.capm_alpha_beta(pd.Series(strat0, index=pd.DatetimeIndex(dates)), spy)
    assert abs(res0["alpha_weekly"]) < 5e-4, res0["alpha_weekly"]
    print("test_capm_alpha_beta_recovers_known_alpha OK — beta",
          round(res["beta"], 3), "alpha_ann", round(res["alpha_ann"], 4),
          "t", round(res["t_alpha"], 2))


if __name__ == "__main__":
    test_uptrend_is_phase2_and_scores_high()
    test_downtrend_scores_low()
    test_features_leak_free()
    test_scorer_predict_proba_shape_and_monotone()
    test_compute_beta_recovers_known_beta()
    test_capm_alpha_beta_recovers_known_alpha()
    print("\nAll momentum_lib synthetic tests passed.")
