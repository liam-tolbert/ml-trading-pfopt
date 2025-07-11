import sys
from collections import OrderedDict
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pandas.core.interchange.dataframe_protocol import DataFrame
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import indicators
import lib
import importlib
importlib.reload(indicators)
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from collections import OrderedDict

#TODO: Use argparse to make a better CLI

def run_model(stock_portfolio, stock_data):
    # Part 1: Getting recommendations
    #download_and_fix_yfinance_data(stock_portfolio)
    real_df = lib.create_stock_features(stock_portfolio, stock_data, lib.get_sp500('2015-01-01'))

    real_df = real_df.sort_values(by='Date')
    today_stocks_features = real_df.loc[[real_df.index.max()]]
    todays_pred = model.predict(today_stocks_features[lib.features])
    stock_col = today_stocks_features["Stock"].reset_index().drop(columns="Date")
    pred_col = pd.DataFrame(todays_pred)

    # Making recommendations per stock
    recommendations = stock_col.join(pred_col)
    recommendations.columns = ["Stock", "Recommendation"]
    recommendations["Recommendation"] = recommendations["Recommendation"].map({0: 'Buy', 1: 'Sell', 2: 'Hold'})

    # Part 2: Getting the Markowitz mean-variance portfolio

    # Isolating buys
    buy_recommendations = recommendations[recommendations.Recommendation == "Buy"].drop(columns="Recommendation")
    rec_array = buy_recommendations.to_numpy().flatten().tolist()
    buy_stocks_history = lib.extract_ticker_dataframe(stock_data, rec_array[0])["Close"]
    for i in range(1, len(rec_array)):
        a = lib.extract_ticker_dataframe(stock_data, rec_array[i])["Close"]
        buy_stocks_history = pd.concat([buy_stocks_history, a], axis=1, join='inner')
    buy_stocks_history.columns = rec_array

    def compute_adjusted_mu(buy_probs, baseline_mu, alpha=0.01):
        tickers = baseline_mu.index.intersection(buy_probs.index)
        adjusted_mu = baseline_mu.loc[tickers] * (1 + alpha * (buy_probs.loc[tickers] - 0.5))
        return adjusted_mu

    probs = model.predict_proba(today_stocks_features[lib.features])
    probs_db = pd.DataFrame(probs, columns=["Hold", "Buy", "Sell"], index=today_stocks_features["Stock"].values)
    buy_probs = probs_db["Buy"]

    baseline_mu = expected_returns.mean_historical_return(buy_stocks_history)

    # Alpha is a strength parameter for the adjustment. Larger values will make the adjustment more aggressive.
    adjusted_mu = compute_adjusted_mu(buy_probs, baseline_mu, alpha=0.05)

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

importance = model.get_booster().get_score(importance_type='weight')  # or 'gain', 'cover'
importance2 = model.get_booster().get_score(importance_type='gain')  # or 'gain', 'cover'
importance3 = model.get_booster().get_score(importance_type='cover')  # or 'gain', 'cover'

print(importance2)

if len(sys.argv) < 3:
    print("Usage: ./model.sh <path/to/tickers.txt> <path/to/stock_data.csv>")
    sys.exit(1)

with open("tickers_k.txt") as f:
    tickers = [line.rstrip() for line in f]

"""tickers = [
    "AAPL", "ABNB", "ACN", "ALAB", "AMD", "AMZN", "ANET", "AOSL", "APP",
    "ASAN", "ASML", "AVGO", "BAH", "BITO", "BWXT", "CLS", "COHR", "COIN", "COST",
    "COWG", "CPRX", "CRDO", "CRM", "CRWV", "DAVE", "DELL", "DKNG", "DOCS",
    "DXPE", "EPD", "FBTC", "FVRR", "GOOG", "GRNY", "HOOD", "IHAK", "INTA", "IONQ",
    "JPM", "LITE", "LQDT", "LUNR", "META", "MRVL", "MSFT", "MU", "NBIS", "NEE",
    "NFLX", "NLR", "NNE", "NUTX", "NVDA", "NVDY", "NVO", "OUST", "OXY", "PANW",
    "PEP", "PLD", "PLTR", "PYPL", "QCOM", "QTUM", "RBRK", "RDDT", "RDNT", "REAL",
    "RGTI", "S", "SAIC", "SCHD", "SEZL", "SKYW", "SMCI", "SMTC", "SNOW", "SOXL",
    "SYM", "TEAM", "TEM", "TOST", "TSM", "U", "UBER", "UPST", "URA", "VIST",
    "VRT", "WMT", "WRD", "XYZ", "HIMS", "OSCR"
]"""

(recommendations, weights) = run_model(tickers, "stocks.csv")

recommendations.to_csv("recommendations.csv")
print("Saved trade recommendations to recommendations.csv")

print(weights)