import sys
from collections import OrderedDict
import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import indicators
import lib
import xgboost as xgb

#TODO: Use argparse to make a better CLI

BUY_THRESHOLD = 0.55  # P(Buy) must exceed this to enter the portfolio
TOP_N = 10            # cap portfolio inclusion at top N by P_Buy (after threshold filter)
ALPHA = 0.5           # P(Buy) influence on expected returns in compute_adjusted_mu

def run_model(stock_portfolio, stock_data):
    # Part 1: Getting recommendations
    #download_and_fix_yfinance_data(stock_portfolio)
    real_df = lib.create_stock_features(stock_portfolio, stock_data, lib.get_sp500('2015-01-01'))

    real_df = real_df.sort_values(by='Date')
    today_stocks_features = real_df.loc[[real_df.index.max()]]
    todays_probs = model.predict_proba(today_stocks_features[lib.features])[:, 1]
    tickers = today_stocks_features["Stock"].values

    recommendations = pd.DataFrame({
        "Stock": tickers,
        "P_Buy": todays_probs,
        "Recommendation": np.where(todays_probs > BUY_THRESHOLD, "Buy", "Sell"),
    })

    # Part 2: Getting the Markowitz mean-variance portfolio

    buy_candidates = (
        recommendations.loc[recommendations.Recommendation == "Buy"]
        .nlargest(TOP_N, "P_Buy")
    )
    rec_array = buy_candidates["Stock"].tolist()
    n_above_threshold = int((recommendations["Recommendation"] == "Buy").sum())
    print(f"{n_above_threshold} candidates cleared P(Buy) > {BUY_THRESHOLD}; taking top {len(rec_array)} by P_Buy")
    if len(rec_array) < 2:
        print("Fewer than 2 candidates — skipping optimizer")
        return recommendations, OrderedDict()

    buy_stocks_history = lib.extract_ticker_dataframe(stock_data, rec_array[0])["Close"]
    for i in range(1, len(rec_array)):
        a = lib.extract_ticker_dataframe(stock_data, rec_array[i])["Close"]
        buy_stocks_history = pd.concat([buy_stocks_history, a], axis=1, join='inner')
    buy_stocks_history.columns = rec_array

    def compute_adjusted_mu(buy_probs, baseline_mu, alpha=0.01):
        common = baseline_mu.index.intersection(buy_probs.index)
        return baseline_mu.loc[common] * (1 + alpha * (buy_probs.loc[common] - 0.5))

    buy_probs = pd.Series(todays_probs, index=tickers).loc[rec_array]

    baseline_mu = expected_returns.mean_historical_return(buy_stocks_history)

    # Alpha is a strength parameter for the adjustment. Larger values will make the adjustment more aggressive.
    adjusted_mu = compute_adjusted_mu(buy_probs, baseline_mu, alpha=ALPHA)

    S = risk_models.sample_cov(buy_stocks_history)

    # Ensure that the adjusted_mu and S use the same tickers
    common_tickers = adjusted_mu.index.intersection(S.index)

    S = S.loc[common_tickers, common_tickers]

    print(adjusted_mu.shape)
    print(S.shape)

    ef = EfficientFrontier(adjusted_mu, S)
    ef.max_sharpe()
    clean_weights = ef.clean_weights()

    # Final portfolio weights
    weights = OrderedDict()
    for key, value in clean_weights.items():
        if value != 0.0:
            weights[key] = value

    return recommendations, weights

model_filename = "model.pkl"

model = xgb.XGBClassifier()
model.load_model(model_filename)

importance2 = model.get_booster().get_score(importance_type='gain')
print(importance2)

if len(sys.argv) < 3:
    print("Usage: ./model.sh <path/to/tickers.txt> <path/to/stock_data.csv>")
    sys.exit(1)

tickers_file = sys.argv[1]
stock_data_file = sys.argv[2]

with open(tickers_file) as f:
    tickers = [line.rstrip() for line in f]

(recommendations, weights) = run_model(tickers, stock_data_file)

recommendations.to_csv("recommendations.csv")
print("Saved trade recommendations to recommendations.csv")

print(weights)