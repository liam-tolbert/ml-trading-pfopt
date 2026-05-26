"""Walk-forward weekly portfolio backtest — pure, data-agnostic core.

This module deliberately contains NO feature engineering and NO data fetching.
All of that lives in `Main.ipynb` (the authoritative feature pipeline, with the
three look-ahead fixes already baked in). The notebook builds the inputs and
hands them here; this file only runs the walk-forward simulation so the loop
logic is unit-testable on synthetic data.

Inputs (built by the notebook):
  - panel:  long DataFrame indexed by Date. Must contain `feature_cols`, plus
            'Stock', 'Signal' (0/1, NaN allowed for unlabeled rows), and
            'Returns-future-1wk' (the realized t -> t+1 return; the PnL source).
            Because the notebook's create_stock_features is point-in-time, every
            feature column here is already leak-free; the only leak-control jobs
            left for the backtest are (a) train each model on past-only labels
            and (b) keep the Markowitz covariance trailing.
  - closes: wide DataFrame, Date x ticker, weekly Close (covariance / mu source).
  - feature_cols: the exact training-time feature column order.
  - fit_fn(X_df, y_series) -> fitted model exposing predict_proba(X)[:, 1].

Rebalancing model:
  - The portfolio is REFORMED only on rebalance weeks (every `rebalance_every`
    weeks). On a rebalance week the model scores the universe, picks top-N by
    P_Buy, computes target weights, and pays turnover cost vs the (drifted)
    current book.
  - Between rebalances the book is HELD: weights drift with realized returns
    (no trading, no cost). This is what makes lower rebalance frequencies cut
    turnover — the whole point of the `rebalance_every` knob.

Leak controls enforced here:
  - At each refit, training rows are restricted to Date <= (refit_date shifted
    back `label_buffer` weeks) so the label's forward return is already realized.
  - Markowitz covariance/mu use only closes up to and including the rebalance week.
  - The realized return for week t (`Returns-future-1wk`) is the OUTCOME, applied
    after weights are chosen from <= t information. It is never a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52


# --------------------------------------------------------------------------- #
# Weighting schemes
# --------------------------------------------------------------------------- #
def equal_weights(buy_tickers, probs_series, close_window, alpha=0.5):
    """1/N over the candidate tickers. Ignores probs/closes."""
    if not buy_tickers:
        return {}
    w = 1.0 / len(buy_tickers)
    return {t: w for t in buy_tickers}


def max_sharpe_weights(buy_tickers, probs_series, close_window, alpha=0.5,
                       risk_free_rate=0.02):
    """Replicates run_model's optimizer: P_Buy-adjusted mu, sample covariance,
    max-Sharpe on the EfficientFrontier. Annualized at weekly frequency
    (frequency=52) — note production run_model leaves pypfopt's default
    frequency=252, which is a latent bug on weekly data.

    Falls back to equal weight if the optimizer fails or there is too little
    usable price history (singular covariance, <2 tickers, etc.).
    """
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier

    cols = [t for t in buy_tickers if t in close_window.columns]
    cw = close_window[cols].dropna(axis=1, how="any")
    if cw.shape[1] < 2 or cw.shape[0] < 3:
        return equal_weights(list(cw.columns) or buy_tickers, probs_series,
                             close_window, alpha)
    try:
        baseline_mu = expected_returns.mean_historical_return(
            cw, frequency=WEEKS_PER_YEAR)
        common = baseline_mu.index.intersection(probs_series.index)
        adjusted_mu = baseline_mu.loc[common] * (
            1 + alpha * (probs_series.loc[common] - 0.5))
        S = risk_models.sample_cov(cw, frequency=WEEKS_PER_YEAR)
        S = S.loc[adjusted_mu.index, adjusted_mu.index]
        ef = EfficientFrontier(adjusted_mu, S)
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        clean = ef.clean_weights()
        weights = {k: v for k, v in clean.items() if v != 0.0}
        if not weights:
            raise ValueError("optimizer returned all-zero weights")
        return weights
    except Exception:
        return equal_weights(list(cw.columns), probs_series, close_window, alpha)


DEFAULT_SCHEMES = {
    "max_sharpe": max_sharpe_weights,
    "equal": equal_weights,
}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def compute_metrics(weekly_returns: pd.Series, periods_per_year: int = WEEKS_PER_YEAR,
                    risk_free_rate: float = 0.0) -> dict:
    """Summary stats for a weekly net-return series."""
    r = pd.Series(weekly_returns).dropna()
    if len(r) == 0:
        return {"n_weeks": 0}
    equity = (1.0 + r).cumprod()
    years = len(r) / periods_per_year
    total = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    ann_ret = float(r.mean() * periods_per_year)
    sharpe = float((ann_ret - risk_free_rate) / vol) if vol > 0 else np.nan
    return {
        "n_weeks": int(len(r)),
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "ann_return": ann_ret,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(equity),
    }


# --------------------------------------------------------------------------- #
# Walk-forward simulation
# --------------------------------------------------------------------------- #
def _drift_weights(weights: dict, fwd_by_ticker: pd.Series, gross: float) -> dict:
    """Carry weights to next week under realized returns (no trading).
    New w_i = w_i * (1 + r_i) / (1 + gross), so the book stays fully invested
    and re-normalized. Missing tickers are treated as 0% return (held flat)."""
    denom = 1.0 + gross
    if not weights or denom == 0:
        return dict(weights)
    out = {}
    for tk, w in weights.items():
        r_i = fwd_by_ticker.get(tk, 0.0)
        r_i = 0.0 if pd.isna(r_i) else float(r_i)
        out[tk] = w * (1.0 + r_i) / denom
    return out


def walk_forward_backtest(
    panel: pd.DataFrame,
    closes: pd.DataFrame,
    feature_cols,
    fit_fn,
    *,
    spy_fwd_returns: pd.Series | None = None,
    schemes=None,
    top_n: int = 10,
    buy_threshold: float | None = None,
    alpha: float = 0.5,
    cov_lookback: int = 52,
    refit_every: int = 13,
    rebalance_every: int = 1,
    label_buffer: int = 2,
    min_train_rows: int = 500,
    backtest_start=None,
    backtest_end=None,
    tc_one_way: float = 0.0005,
    spread_per_share: float = 0.02,
    signal_col: str = "Signal",
    stock_col: str = "Stock",
    fwd_ret_col: str = "Returns-future-1wk",
    risk_free_rate: float = 0.0,
    verbose: bool = False,
):
    """Run the walk-forward backtest.

    Parameters of note
    -------------------
    refit_every : weeks between model retrains (independent of rebalancing).
    top_n : number of names to hold, selected by highest P_Buy (rank-based).
        Ignored when `buy_threshold` is set.
    buy_threshold : if set (e.g. 0.5), select EVERY name with P_Buy > threshold
        instead of a fixed top_n -- a variable-size "all buy signals" book. In
        weeks where no name clears the threshold the book goes flat (cash, zero
        return that week). When None, the fixed `top_n`-by-rank rule applies.
    rebalance_every : weeks between portfolio reforms. 1 = reform every week
        (full turnover); larger = hold the book and let it drift, paying cost
        only on rebalance weeks.
    tc_one_way : flat one-way transaction cost as a fraction of traded notional
        (cost = tc_one_way * Σ|Δw|). Used when `spread_per_share` is None.
    spread_per_share : if set (e.g. 0.02 for a $0.02 bid-ask spread), cost is
        computed PER TICKER from that week's close: (spread/2)/price_i × |Δw_i|,
        summed over names. This makes cheap stocks cost more bps than expensive
        ones (the real spread effect; commission = $0). Names with a missing or
        non-positive price fall back to the flat `tc_one_way` rate. When None,
        the flat `tc_one_way` model applies and behavior is unchanged.
    backtest_start / backtest_end : restrict the live simulation window
        (e.g. backtest_end='2025-08-15' to match a benchmark that ends there).

    Returns
    -------
    dict with 'weekly' (per-week rows), 'equity' (curves per scheme + 'spy'),
    'metrics' (stats per scheme + 'spy'), and 'refit_dates'.
    """
    schemes = dict(DEFAULT_SCHEMES if schemes is None else schemes)

    feature_cols = list(feature_cols)
    panel = panel.sort_index()
    all_dates = panel.index.unique().sort_values()
    date_pos = {d: i for i, d in enumerate(all_dates)}

    live_dates = list(all_dates)
    if backtest_start is not None:
        bs = pd.Timestamp(backtest_start)
        live_dates = [d for d in live_dates if d >= bs]
    if backtest_end is not None:
        be = pd.Timestamp(backtest_end)
        live_dates = [d for d in live_dates if d <= be]

    labeled = panel[panel[signal_col].notna()]

    records = []
    cur_weights = {s: {} for s in schemes}   # current held book per scheme (drifts)
    model = None
    refit_dates = []

    weeks_since_refit = refit_every       # force a fit on the first live week
    weeks_since_rebal = rebalance_every   # force a rebalance on the first live week

    for t in live_dates:
        # ---- refit on a quarterly cadence, past-only labels ----
        if weeks_since_refit >= refit_every or model is None:
            cutoff_pos = date_pos[t] - label_buffer
            if cutoff_pos < 0:
                weeks_since_refit += 1
                weeks_since_rebal += 1
                continue
            cutoff_date = all_dates[cutoff_pos]
            train_rows = labeled[labeled.index <= cutoff_date]
            if len(train_rows) < min_train_rows:
                weeks_since_refit += 1
                weeks_since_rebal += 1
                continue
            model = fit_fn(train_rows[feature_cols], train_rows[signal_col].astype(int))
            refit_dates.append(t)
            weeks_since_refit = 0
            if verbose:
                print(f"[refit] {t.date()} on {len(train_rows)} rows "
                      f"(labels <= {cutoff_date.date()})")

        if model is None or t not in panel.index:
            weeks_since_refit += 1
            weeks_since_rebal += 1
            continue

        # realized t -> t+1 returns for every ticker available this week
        week_rows = panel.loc[[t]]
        fwd_by_ticker = pd.Series(week_rows[fwd_ret_col].to_numpy(),
                                  index=week_rows[stock_col].to_numpy())

        # ---- decide whether to reform the book this week ----
        do_rebalance = weeks_since_rebal >= rebalance_every and len(week_rows) >= 2
        if do_rebalance:
            probs = model.predict_proba(week_rows[feature_cols])[:, 1]
            tickers = week_rows[stock_col].to_numpy()
            probs_series = pd.Series(probs, index=tickers)
            if buy_threshold is not None:
                # every name the model rates a "buy" -> variable-size book
                buy_tickers = list(probs_series[probs_series > buy_threshold].index)
            else:
                buy_tickers = list(
                    probs_series.nlargest(min(top_n, len(probs_series))).index)
            close_window = closes.loc[:t].tail(cov_lookback)

        row = {"Date": t, "rebalanced": int(do_rebalance)}
        for scheme_name, weight_fn in schemes.items():
            if do_rebalance:
                target = weight_fn(buy_tickers, probs_series, close_window, alpha=alpha)
                keys = set(target) | set(cur_weights[scheme_name])
                turnover = sum(abs(target.get(k, 0.0) - cur_weights[scheme_name].get(k, 0.0))
                               for k in keys)
                if spread_per_share is None:
                    cost = tc_one_way * turnover
                else:
                    # price-dependent: (spread/2)/price_i × |Δw_i| per name.
                    # close_window's last row is the most recent close <= t.
                    price_row = close_window.iloc[-1]
                    cost = 0.0
                    for k in keys:
                        dw = abs(target.get(k, 0.0) - cur_weights[scheme_name].get(k, 0.0))
                        if dw == 0.0:
                            continue
                        price = price_row.get(k, np.nan)
                        if pd.isna(price) or price <= 0:
                            cost += tc_one_way * dw        # fallback: flat rate
                        else:
                            cost += (spread_per_share / 2.0) / float(price) * dw
                cur_weights[scheme_name] = dict(target)
            else:
                turnover = 0.0
                cost = 0.0

            book = cur_weights[scheme_name]
            gross = 0.0
            for tk, w in book.items():
                r_i = fwd_by_ticker.get(tk, np.nan)
                if pd.notna(r_i):
                    gross += w * float(r_i)
            net = gross - cost

            # drift the book into next week (held, no trading)
            cur_weights[scheme_name] = _drift_weights(book, fwd_by_ticker, gross)

            row[f"{scheme_name}_gross"] = gross
            row[f"{scheme_name}_net"] = net
            row[f"{scheme_name}_turnover"] = turnover
            row[f"{scheme_name}_n"] = len(book)

        if spy_fwd_returns is not None:
            row["spy"] = float(spy_fwd_returns.get(t, np.nan))

        records.append(row)
        weeks_since_refit += 1
        # elapsed weeks since last rebalance, as seen at the START of next week:
        # 1 if we just rebalanced, else one more than before.
        weeks_since_rebal = 1 if do_rebalance else weeks_since_rebal + 1

    weekly = pd.DataFrame(records).set_index("Date").sort_index()

    # ---- equity curves + metrics ----
    equity = pd.DataFrame(index=weekly.index)
    metrics = {}
    for scheme_name in schemes:
        net = weekly[f"{scheme_name}_net"]
        equity[scheme_name] = (1.0 + net.fillna(0.0)).cumprod()
        metrics[scheme_name] = compute_metrics(net, risk_free_rate=risk_free_rate)
    if spy_fwd_returns is not None and "spy" in weekly:
        equity["spy"] = (1.0 + weekly["spy"].fillna(0.0)).cumprod()
        metrics["spy"] = compute_metrics(weekly["spy"], risk_free_rate=risk_free_rate)

    metrics_df = pd.DataFrame(metrics).T

    return {
        "weekly": weekly,
        "equity": equity,
        "metrics": metrics_df,
        "refit_dates": refit_dates,
    }
