from collections import OrderedDict

# imports
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
import importlib
importlib.reload(indicators)
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from collections import OrderedDict

features = ["SMA_5v20", "RSI", "MACD", "Bollinger_Bands", "ATR", "Stochastic", "OBV", "ADX", "Aroon", "Bull_Probability", "Returns-3wk-1wklag", "Returns-1wk-0wklag"]

def download_and_fix_yfinance_data(stocks): # Takes raw yfinance dataframe and fixes it if it has that weird date syncing problem
    data = {}

    for t in stocks:
        df = yf.download(t, interval='1d', start='2015-01-01', auto_adjust=True)
        df = df.resample('W-FRI').agg({ # everything needs to be on the thursday interval. weekly prices sometimes start not on thursday
          ('Open', t): 'first',
          ('High', t): 'max',
          ('Low', t): 'min',
          ('Close', t): 'last',
          ('Volume', t): 'sum'
        })
        df.name = t
        data[t] = df

    # Combine and align on the same date index
    combined = pd.concat(data.values(), axis=1, join='outer')
    combined = combined.ffill()  # or .bfill(), depending on your use case

    non_nan_counts = combined.count()
    cols_to_drop = non_nan_counts[non_nan_counts < 20].index
    y = combined.drop(columns=cols_to_drop)

    stockData = y.copy()

    # Rename columns, joining multi-level column names into a single string with "_".
    stockData.columns = ["_".join(col) if isinstance(col, tuple) else col for col in stockData.columns]

    # Remove unnecessary columns such as "level_0" or "index" that may have been carried over.
    stockData = stockData.loc[:, ~stockData.columns.isin(["level_0", "index"])]
    stockData.to_csv("stocks.csv")

    return stockData


def download_and_fix_sp500():
    y = yf.download("^GSPC", start="2015-01-01", interval="1d")
    y = y.resample('W-FRI').agg({
        ('Open', "^GSPC"): 'first',
        ('High', "^GSPC"): 'max',
        ('Low', "^GSPC"): 'min',
        ('Close', "^GSPC"): 'last',
        ('Volume', "^GSPC"): 'sum'
    })
    sp500 = y.copy()

    sp500.columns = ["_".join(col) if isinstance(col, tuple) else col for col in sp500.columns]

    sp500 = sp500.loc[:, ~sp500.columns.isin(["level_0", "index"])]
    sp500.to_csv("SP500.csv")

def extract_ticker_dataframe(csv_filepath: str, ticker: str) -> pd.DataFrame:
    """
    Reads a multi-ticker CSV file (like one from yfinance) and isolates the data
    for the specified ticker. This updated version assumes that the CSV contains the
    dates as the index (first column), so we use that as the Date information.

    The CSV is expected to have a two-row header:
      - The first row contains field names (e.g., "Open", "High", etc.).
      - The second row contains the ticker symbols for each column.

    The returned DataFrame's columns will be ordered as needed for Backtrader:
      Date, Open, High, Low, Close, Volume.
    """
    # Use the first column as the index so that the dates are read from the CSV.
    df = pd.read_csv(csv_filepath)#, header=[0, 1], index_col=0)

    # Convert the index to datetime in case it's not already
    df.index = pd.to_datetime(df["Date"], errors="coerce")
    df = df.drop("Date", axis=1)

    # Identify all columns for the specified ticker (matching on the lower level).
    ticker_cols = [col for col in df.columns if ticker == str(col).strip().split("_")[-1]]
    if not ticker_cols:
        raise ValueError(f"Ticker '{ticker}' not found in the CSV file.")

    # Extract the ticker's columns
    df_ticker = df.loc[:, ticker_cols].copy()
    #df_ticker.columns = [col[0].strip() for col in df_ticker.columns]

    # Removing ticker name from column names
    for col in df_ticker.columns:
        df_ticker.rename(columns={col: str(col).split("_")[0]}, inplace=True)

    desired_order = ["Open", "High", "Low", "Close", "Volume"]

    available_order = [col for col in desired_order if col in df_ticker.columns]
    df_ticker = df_ticker[available_order]

    df_ticker = df_ticker.reset_index().rename(columns={"index": "Date"}).set_index("Date")

    return df_ticker.dropna()

def classify_regimes(sp500):
    model = MarkovRegression(sp500['Log_Returns'], k_regimes=2, trend='c', switching_variance=True)
    result = model.fit()
    #print(result.summary())
    smoothed_probs = result.smoothed_marginal_probabilities
    sp500['Regime'] = smoothed_probs.idxmax(axis=1)
    sp500['Bull_Prob'] = smoothed_probs[0]

    """if show_regimes:
        plt.plot(sp500.index, smoothed_probs[0], label="Probability of Bull Market")
        plt.fill_between(sp500.index, 0, 1, where=sp500['Bull_Prob'] > 0.5, color='green', alpha=0.3)
        plt.fill_between(sp500.index, 0, 1, where=sp500['Bull_Prob'] <= 0.5, color='red', alpha=0.3)
        plt.legend()
        plt.title("Bull vs. Bear Market Probability")
        plt.show()"""

    return sp500["Bull_Prob"].to_frame()
    # 0 -> Bull, 1 -> Bear

# Compute rolling portfolio weights using a lookback period (e.g., 52 weeks)
def compute_rolling_portfolio_weights(data, lookback_window=52):
    """
    Computes portfolio weights for each date using historical data up to that date.

    Args:
        data (pd.DataFrame): DataFrame with dates as index and stocks as columns.
        lookback_window (int): Number of days to look back for the optimization.

    Returns:
        pd.DataFrame: A DataFrame with dates as index and stocks as columns, containing weights.
    """
    weights_list = []
    dates = []
    for date in data.index[lookback_window:]:
        window_data = data.loc[:date].tail(lookback_window)
        mu = expected_returns.mean_historical_return(window_data)
        S = risk_models.sample_cov(window_data)
        ef = EfficientFrontier(mu, S)
        try:
            ef.max_sharpe()
            clean_weights = ef.clean_weights()
        except Exception as e:
            # In case optimization fails, assign zero weights.
            clean_weights = {stock: 0 for stock in data.columns}
        weights_list.append(clean_weights)
        dates.append(date)
    weights_df = pd.DataFrame(weights_list, index=dates)
    return weights_df

# 2. Feature engineering: Technical indicators, fundamentals
# SMA, RSI, MACD, etc. etc. basically, seeing which indicator sticks
def create_stock_features(stocks, stock_data_filename, sp500):
    feature_rows = []
    regime_df = classify_regimes(sp500)

    for stock in stocks:
        # will be filled with indicators for one stock
        try:
            prices = extract_ticker_dataframe(stock_data_filename, stock)
        except ValueError:
            continue
        # weekly return
        prices["Return"] = prices["Close"].pct_change(periods=2)

        features = pd.DataFrame(index=prices.index)

        # Simple Moving Avg Comparison (5 vs 20)
        sma = indicators.sma_strategy(prices["Close"].to_frame(), 5, 20)
        features["SMA_5v20"] = sma["signal_raw"]

        # Relative Strength Index (RSI)
        rsi = indicators.rsi_strategy(prices["Close"].to_frame())
        features["RSI"] = rsi["rsi"]

        # Moving Average Convergence Divergence (MACD)
        macd = indicators.macd_strategy(prices["Close"].to_frame())
        features["MACD"] = macd["signal_raw"] # only using signal, not other columns, since I don't want to have weird "fitting" of the model to the components of the macd signal

        # Bollinger Bands
        bands = indicators.bollinger_strategy(prices["Close"].to_frame())
        features["Bollinger_Bands"] = bands["signal_ternary"]

        # Average True Range (ATR)
        atr = indicators.atr_indicator(prices[["High", "Low", "Close"]])
        features["ATR"] = atr["signal"]

        # Stochastic Oscillator Strategy
        stochastic = indicators.stochastic_strategy(prices[["High", "Low", "Close"]])
        features["Stochastic"] = stochastic["signal"]

        # OBV (On-Balance Volume)
        obv = indicators.obv_strategy(prices[["Close", "Volume"]])
        features["OBV"] = obv["obv"]

        # ADX (Average Directional Index)
        adx = indicators.adx_strategy(prices[["High", "Low", "Close"]])
        features["ADX"] = adx["adx"]

        # Aroon Indicator
        aroon = indicators.aroon_strategy(prices[["High", "Low"]])
        features["Aroon"] = aroon["aroon_oscillator"]

        # Returns features. Overall 4 week percent change, split into 3 week period, 1 week lag and 1 week period, 0 week lag
        features["Returns-3wk-1wklag"] = prices["Close"].shift(1).pct_change(periods=3)
        features["Returns-1wk-0wklag"] = prices["Close"].pct_change()

        # Other technical indicators and fundamentals?

        # rolling return (2 week window)
        features["Returns-2wk"] = prices["Return"]
        features["Bull_Probability"] = regime_df["Bull_Prob"]

        features["Stock"] = stock
        features = features.reset_index().rename(columns={"index": "Date"})
        feature_rows.append(features)

    # Combine data for all stocks into one dataframe
    features_long = pd.concat(feature_rows, ignore_index=True).set_index("Date").dropna()



    return features_long

def get_sp500(start_date: str):
    if start_date[4] != '-' or start_date[7] != '-':
        raise ValueError("Invalid start date format")

    sp500 = pd.read_csv("SP500.csv")
    sp500.set_index("Date", inplace=True)
    sp500.index = pd.to_datetime(sp500.index)
    sp500['Log_Returns'] = np.log(sp500['Close_^GSPC'] / sp500['Close_^GSPC'].shift(1)).dropna()
    sp500 = sp500.loc[start_date:]
    return sp500.dropna()