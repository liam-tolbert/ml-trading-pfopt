# indicators for use by my main notebook
import math

import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def sma_strategy(data: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """
    Implements a simple moving average crossover strategy.
    It computes short-term and long-term SMAs and identifies buy/sell signals 
    based on SMA crossovers.
    """
    data['short_sma'] = data['Close'].rolling(window=short_window).mean()
    data['long_sma'] = data['Close'].rolling(window=long_window).mean()

    data['signal_ternary'] = 0  # Default to no signal
    data.loc[data['short_sma'] > data['long_sma'], 'signal_ternary'] = 1
    data.loc[data['short_sma'] < data['long_sma'], 'signal_ternary'] = -1

    data['signal_raw'] = (data['short_sma'] - data['long_sma']).abs()

    return data

def rsi_strategy(data: pd.DataFrame, window: int = 14, overbought: int = 70, oversold: int = 30) -> pd.DataFrame:
    """
    Implements a Relative Strength Index (RSI) strategy.
    It calculates the RSI and identifies overbought and oversold signals.
     Args:
        data (pd.DataFrame): DataFrame of a single stock in yfinance format.
        window (int): Window size for RSI calculation (default is 14).
        overbought (int): RSI threshold for overbought condition (default is 70).
        oversold (int): RSI threshold for oversold condition (default is 30).

    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    data['signal_ternary'] = 0  # Default to no signal
    data.loc[data['rsi'] > overbought, 'signal_ternary'] = -1
    data.loc[data['rsi'] < oversold, 'signal_ternary'] = 1

    return data

def macd_strategy(data: pd.DataFrame, short_span: int = 12, long_span: int = 26, signal_span: int = 9) -> pd.DataFrame:
    """
    Implements a MACD strategy.
    Uses the 'Close' price column from yfinance data.

    Args:
        data (pd.DataFrame): DataFrame of a single stock in yfinance format.
        short_span (int): Period for the short-term EMA (default is 12).
        long_span (int): Period for the long-term EMA (default is 26).
        signal_span (int): Period for the MACD signal line (default is 9).

    Returns:
        pd.DataFrame: DataFrame with additional columns for EMAs, MACD, signal line, and signals.
    """
    data['ema_short'] = data['Close'].ewm(span=short_span, adjust=False).mean()
    data['ema_long'] = data['Close'].ewm(span=long_span, adjust=False).mean()
    data['macd'] = data['ema_short'] - data['ema_long']
    data['macd_signal'] = data['macd'].ewm(span=signal_span, adjust=False).mean()

    data['signal_ternary'] = 0  # Default to no signal
    data.loc[data['macd'] > data['macd_signal'], 'signal_ternary'] = 1
    data.loc[data['macd'] < data['macd_signal'], 'signal_ternary'] = -1

    data['signal_raw'] = (data['macd'] - data['macd_signal']).abs()

    return data

def bollinger_strategy(data: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    """
    Implements a Bollinger Bands strategy.
    Uses the 'Close' price column from yfinance data.

    Args:
        data (pd.DataFrame): DataFrame of a single stock in yfinance format.
        window (int): Window size for moving average and standard deviation (default is 20).
        num_std (int): Number of standard deviations for band width (default is 2).

    Returns:
        pd.DataFrame: DataFrame with additional columns for SMA, upper/lower bands, and signals.
    """
    data['sma'] = data['Close'].rolling(window=window).mean()
    data['std'] = data['Close'].rolling(window=window).std()
    data['upper_band'] = data['sma'] + num_std * data['std']
    data['lower_band'] = data['sma'] - num_std * data['std']

    data['signal_ternary'] = 0  # Default to no signal
    data.loc[data['Close'] > data['upper_band'], 'signal_ternary'] = -1  # Sell signal
    data.loc[data['Close'] < data['lower_band'], 'signal_ternary'] = 1  # Buy signal

    return data

def atr_indicator(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculates the Average True Range (ATR) for volatility measurement.
    Uses the 'High', 'Low', and 'Close' columns from yfinance data.

    Returns:
        pd.DataFrame: DataFrame with an additional column 'atr'.
    """
    data['high_low'] = data['High'] - data['Low']
    data['high_prev_close'] = (data['High'] - data['Close'].shift()).abs()
    data['low_prev_close'] = (data['Low'] - data['Close'].shift()).abs()
    data['true_range'] = data[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
    data['signal'] = data['true_range'].rolling(window=window).mean()

    # Clean up intermediate columns
    data.drop(['high_low', 'high_prev_close', 'low_prev_close', 'true_range'], axis=1, inplace=True)

    return data

def stochastic_strategy(data: pd.DataFrame, window: int = 14, overbought: int = 80, oversold: int = 20) -> pd.DataFrame:
    """
    Implements the stochastic oscillator strategy.
    Uses the 'High', 'Low', and 'Close' columns from yfinance data.
    """
    data['low_min'] = data['Low'].rolling(window=window).min()
    data['high_max'] = data['High'].rolling(window=window).max()
    data['stochastic'] = ((data['Close'] - data['low_min']) / (data['high_max'] - data['low_min'])) * 100

    data['signal'] = 0  # Default to no signal
    data.loc[data['stochastic'] > overbought, 'signal'] = -1  # Sell signal
    data.loc[data['stochastic'] < oversold, 'signal'] = 1  # Buy signal

    # Clean up intermediate columns
    data.drop(['low_min', 'high_max'], axis=1, inplace=True)

    return data

def obv_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the On-Balance Volume (OBV) indicator.
    Adds an 'obv' column to the DataFrame.
    """
    data = data.copy()
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    data['obv'] = obv
    return data

def adx_strategy(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculates the Average Directional Index (ADX) indicator.
    Adds 'adx', 'plus_di', and 'minus_di' columns to the DataFrame.
    """
    data = data.copy()
    # Previous period values
    data['prev_high'] = data['High'].shift(1)
    data['prev_low'] = data['Low'].shift(1)
    data['prev_close'] = data['Close'].shift(1)
    
    # True Range (TR)
    data['tr'] = data[['High', 'prev_close']].max(axis=1) - data[['Low', 'prev_close']].min(axis=1)
    
    # Directional Movements
    data['plus_dm'] = data['High'] - data['prev_high']
    data['minus_dm'] = data['prev_low'] - data['Low']
    data['plus_dm'] = data.apply(lambda row: row['plus_dm'] if (row['plus_dm'] > row['minus_dm'] and row['plus_dm'] > 0) else 0, axis=1)
    data['minus_dm'] = data.apply(lambda row: row['minus_dm'] if (row['minus_dm'] > row['plus_dm'] and row['minus_dm'] > 0) else 0, axis=1)
    
    # Smoothed averages using Wilder's smoothing method (approximated with EWMA)
    atr = data['tr'].ewm(alpha=1/window, adjust=False).mean()
    plus_dm_smoothed = data['plus_dm'].ewm(alpha=1/window, adjust=False).mean()
    minus_dm_smoothed = data['minus_dm'].ewm(alpha=1/window, adjust=False).mean()
    
    data['plus_di'] = 100 * (plus_dm_smoothed / atr)
    data['minus_di'] = 100 * (minus_dm_smoothed / atr)
    data['dx'] = 100 * (abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di']).replace(0, np.nan))
    data['adx'] = data['dx'].ewm(alpha=1/window, adjust=False).mean()
    
    # Clean up intermediate columns
    data.drop(['prev_high', 'prev_low', 'prev_close', 'tr', 'plus_dm', 'minus_dm', 'dx'], axis=1, inplace=True)
    return data

def aroon_strategy(data: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    """
    Calculates the Aroon Up, Aroon Down, and Aroon Oscillator indicators.
    Adds 'aroon_up', 'aroon_down', and 'aroon_oscillator' columns to the DataFrame.
    """
    data = data.copy()
    aroon_up = [np.nan] * (window - 1)
    aroon_down = [np.nan] * (window - 1)
    
    for i in range(window - 1, len(data)):
        window_data = data.iloc[i - window + 1:i + 1]
        high_values = window_data['High'].values
        low_values = window_data['Low'].values
        days_since_high = window - 1 - high_values.argmax()
        days_since_low = window - 1 - low_values.argmin()
        up = ((window - days_since_high) / window) * 100
        down = ((window - days_since_low) / window) * 100
        aroon_up.append(up)
        aroon_down.append(down)
    
    data['aroon_up'] = aroon_up
    data['aroon_down'] = aroon_down
    data['aroon_oscillator'] = data['aroon_up'] - data['aroon_down']
    return data