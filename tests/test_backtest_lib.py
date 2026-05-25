"""Synthetic-data tests for src/backtest_lib.walk_forward_backtest.

Runs as a plain script (`python tests/test_backtest_lib.py`) or under pytest.
Uses a deterministic stub model so portfolio returns are hand-checkable and the
training window can be asserted leak-free without depending on XGBoost/pypfopt.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import backtest_lib as bl  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic world
# --------------------------------------------------------------------------- #
TICKERS = [f"T{i}" for i in range(6)]
N_WEEKS = 80
FEATURES = ["score", "noise"]


def build_world(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-02", periods=N_WEEKS, freq="W-FRI")

    # wide close frame: independent geometric random walks
    closes = pd.DataFrame(index=dates, columns=TICKERS, dtype=float)
    for tk in TICKERS:
        rets = rng.normal(0.001, 0.02, size=N_WEEKS)
        closes[tk] = 100.0 * np.cumprod(1.0 + rets)

    # long panel
    rows = []
    for tk in TICKERS:
        c = closes[tk]
        fwd1 = c.shift(-1) / c - 1.0  # realized t -> t+1 return
        for d in dates:
            rows.append({
                "Date": d,
                "Stock": tk,
                # 'score' is a fixed per-ticker rank signal so top-N is deterministic
                "score": float(TICKERS.index(tk)),
                "noise": float(rng.normal()),
                "Signal": int(rng.integers(0, 2)),
                "Returns-future-1wk": fwd1.loc[d],
            })
    panel = pd.DataFrame(rows).set_index("Date").sort_index()
    panel = panel.dropna(subset=["Returns-future-1wk"])  # drop last week (no fwd)
    return panel, closes


class StubModel:
    """Deterministic: P_Buy is a monotonic function of the 'score' feature.
    Records the max training Date so tests can assert no future-label leakage.
    """
    def __init__(self):
        self.max_train_date = None

    def fit_record(self, X, y):
        self.max_train_date = X.index.max()
        return self

    def predict_proba(self, X):
        # higher score -> higher P_Buy, squashed to (0,1)
        p = 1.0 / (1.0 + np.exp(-(X["score"].to_numpy() - 2.5)))
        return np.column_stack([1.0 - p, p])


def make_fit_fn(record):
    def fit_fn(X, y):
        m = StubModel()
        m.fit_record(X, y)
        record["models"].append((X.index.min(), X.index.max(), len(X)))
        return m
    return fit_fn


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_runs_and_shapes():
    panel, closes = build_world()
    record = {"models": []}
    spy = panel.groupby("Date")["Returns-future-1wk"].mean()  # toy benchmark
    res = bl.walk_forward_backtest(
        panel, closes, FEATURES, make_fit_fn(record),
        spy_fwd_returns=spy, schemes={"equal": bl.equal_weights},
        top_n=3, refit_every=13, label_buffer=2, min_train_rows=10,
        backtest_start=panel.index.unique()[20],
    )
    assert len(res["weekly"]) > 0, "no weeks simulated"
    assert "equal" in res["metrics"].index
    assert "spy" in res["metrics"].index
    assert (res["equity"]["equal"] > 0).all()
    print("test_runs_and_shapes OK — weeks:", len(res["weekly"]),
          "refits:", len(res["refit_dates"]))


def test_no_future_label_leak():
    """Each model's training rows must end at or before (refit_date - label_buffer)."""
    panel, closes = build_world()
    record = {"models": []}
    label_buffer = 2
    res = bl.walk_forward_backtest(
        panel, closes, FEATURES, make_fit_fn(record),
        schemes={"equal": bl.equal_weights},
        top_n=3, refit_every=13, label_buffer=label_buffer, min_train_rows=10,
        backtest_start=panel.index.unique()[20],
    )
    all_dates = panel.index.unique().sort_values()
    pos = {d: i for i, d in enumerate(all_dates)}
    assert len(record["models"]) == len(res["refit_dates"])
    for (tr_min, tr_max, n), refit_date in zip(record["models"], res["refit_dates"]):
        expected_cutoff = all_dates[pos[refit_date] - label_buffer]
        assert tr_max <= expected_cutoff, (
            f"LEAK: model refit at {refit_date.date()} trained on data up to "
            f"{tr_max.date()} > cutoff {expected_cutoff.date()}")
    print("test_no_future_label_leak OK —", len(record["models"]),
          "refits all respect the", label_buffer, "week label buffer")


def test_equal_weight_matches_hand_calc():
    """For the equal-weight scheme, gross return == mean of the realized
    next-week returns of the top-N scored tickers."""
    panel, closes = build_world()
    top_n = 3
    res = bl.walk_forward_backtest(
        panel, closes, FEATURES, make_fit_fn({"models": []}),
        schemes={"equal": bl.equal_weights},
        top_n=top_n, refit_every=13, label_buffer=2, min_train_rows=10,
        backtest_start=panel.index.unique()[20], tc_one_way=0.0,
    )
    # 'score' == ticker index, so the top-3 are always T5,T4,T3.
    top = ["T5", "T4", "T3"]
    for t in res["weekly"].index:
        wk = panel.loc[[t]].set_index("Stock")["Returns-future-1wk"]
        expected = wk.loc[top].mean()
        got = res["weekly"].loc[t, "equal_gross"]
        assert abs(got - expected) < 1e-9, f"{t}: {got} != {expected}"
    print("test_equal_weight_matches_hand_calc OK — exact match on",
          len(res["weekly"]), "weeks")


def test_hold_and_drift_no_cost():
    """With a single rebalance (rebalance_every huge), the book is bought once
    then held: first-week turnover == 1.0, every later week has zero turnover,
    zero cost, and net == gross (the drift path trades nothing)."""
    panel, closes = build_world()
    res = bl.walk_forward_backtest(
        panel, closes, FEATURES, make_fit_fn({"models": []}),
        schemes={"equal": bl.equal_weights},
        top_n=3, refit_every=13, label_buffer=2, min_train_rows=10,
        backtest_start=panel.index.unique()[20], rebalance_every=10_000,
        tc_one_way=0.001,
    )
    wk = res["weekly"]
    assert (wk["equal_net"] <= wk["equal_gross"] + 1e-12).all()
    first = wk.iloc[0]
    assert abs(first["equal_turnover"] - 1.0) < 1e-9, first["equal_turnover"]
    later = wk.iloc[1:]
    assert (later["equal_turnover"].abs() < 1e-12).all(), "held book should not trade"
    assert np.allclose(later["equal_net"], later["equal_gross"])
    assert int(wk["rebalanced"].sum()) == 1, "expected exactly one rebalance"
    print("test_hold_and_drift_no_cost OK — bought once, held, net==gross after")


def test_rebalance_frequency_reduces_turnover():
    """Lower rebalance frequency -> fewer trades -> less total cost drag."""
    panel, closes = build_world()
    common = dict(schemes={"equal": bl.equal_weights}, top_n=3, refit_every=13,
                  label_buffer=2, min_train_rows=10,
                  backtest_start=panel.index.unique()[20], tc_one_way=0.001)
    weekly = bl.walk_forward_backtest(panel, closes, FEATURES,
                                      make_fit_fn({"models": []}),
                                      rebalance_every=1, **common)["weekly"]
    monthly = bl.walk_forward_backtest(panel, closes, FEATURES,
                                       make_fit_fn({"models": []}),
                                       rebalance_every=4, **common)["weekly"]
    cost_w = (weekly["equal_gross"] - weekly["equal_net"]).sum()
    cost_m = (monthly["equal_gross"] - monthly["equal_net"]).sum()
    assert cost_m < cost_w, f"monthly cost {cost_m} should be < weekly {cost_w}"
    assert monthly["rebalanced"].sum() < weekly["rebalanced"].sum()
    print(f"test_rebalance_frequency_reduces_turnover OK — "
          f"total cost weekly={cost_w:.4f} > monthly={cost_m:.4f}")


def test_backtest_end_truncates():
    panel, closes = build_world()
    end = panel.index.unique()[-15]
    res = bl.walk_forward_backtest(
        panel, closes, FEATURES, make_fit_fn({"models": []}),
        schemes={"equal": bl.equal_weights},
        top_n=3, refit_every=13, label_buffer=2, min_train_rows=10,
        backtest_start=panel.index.unique()[20], backtest_end=end,
    )
    assert res["weekly"].index.max() <= end
    print("test_backtest_end_truncates OK — last week", res["weekly"].index.max().date())


if __name__ == "__main__":
    test_runs_and_shapes()
    test_no_future_label_leak()
    test_equal_weight_matches_hand_calc()
    test_hold_and_drift_no_cost()
    test_rebalance_frequency_reduces_turnover()
    test_backtest_end_truncates()
    print("\nAll backtest_lib synthetic tests passed.")
