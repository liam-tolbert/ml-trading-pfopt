# indicators for use by my main notebook
import math
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def sma_strategy(data: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    data = data.copy()
    data['short_sma'] = data['Close'].rolling(window=short_window).mean()
    data['long_sma'] = data['Close'].rolling(window=long_window).mean()

    data['signal_ternary'] = 0
    data.loc[data['short_sma'] > data['long_sma'], 'signal_ternary'] = 1
    data.loc[data['short_sma'] < data['long_sma'], 'signal_ternary'] = -1

    data['signal_raw'] = (data['short_sma'] - data['long_sma']).abs()
    return data

def rsi_strategy(data: pd.DataFrame, window: int = 14, overbought: int = 70, oversold: int = 30) -> pd.DataFrame:
    data = data.copy()
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    data['signal_ternary'] = 0
    data.loc[data['rsi'] > overbought, 'signal_ternary'] = -1
    data.loc[data['rsi'] < oversold, 'signal_ternary'] = 1
    return data

def macd_strategy(data: pd.DataFrame, short_span: int = 12, long_span: int = 26, signal_span: int = 9) -> pd.DataFrame:
    data = data.copy()
    data['ema_short'] = data['Close'].ewm(span=short_span, adjust=False).mean()
    data['ema_long'] = data['Close'].ewm(span=long_span, adjust=False).mean()
    data['macd'] = data['ema_short'] - data['ema_long']
    data['macd_signal'] = data['macd'].ewm(span=signal_span, adjust=False).mean()

    data['signal_ternary'] = 0
    data.loc[data['macd'] > data['macd_signal'], 'signal_ternary'] = 1
    data.loc[data['macd'] < data['macd_signal'], 'signal_ternary'] = -1
    data['signal_raw'] = (data['macd'] - data['macd_signal']).abs()
    return data

def bollinger_strategy(data: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    data = data.copy()
    data['sma'] = data['Close'].rolling(window=window).mean()
    data['std'] = data['Close'].rolling(window=window).std()
    data['upper_band'] = data['sma'] + num_std * data['std']
    data['lower_band'] = data['sma'] - num_std * data['std']

    data['signal_ternary'] = 0
    data.loc[data['Close'] > data['upper_band'], 'signal_ternary'] = -1
    data.loc[data['Close'] < data['lower_band'], 'signal_ternary'] = 1
    return data

def atr_indicator(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    data = data.copy()
    high_low = data['High'] - data['Low']
    high_prev_close = (data['High'] - data['Close'].shift()).abs()
    low_prev_close = (data['Low'] - data['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    data['signal'] = true_range.rolling(window=window).mean()
    return data

def stochastic_strategy(data: pd.DataFrame, window: int = 14, overbought: int = 80, oversold: int = 20) -> pd.DataFrame:
    data = data.copy()
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    data['stochastic'] = ((data['Close'] - low_min) / (high_max - low_min)) * 100

    data['signal'] = 0
    data.loc[data['stochastic'] > overbought, 'signal'] = -1
    data.loc[data['stochastic'] < oversold, 'signal'] = 1
    return data

def obv_strategy(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    close = data['Close'].values
    volume = data['Volume'].values
    obv = [0]
    for i in range(1, len(data)):
        if close[i] > close[i - 1]:
            obv.append(obv[-1] + volume[i])
        elif close[i] < close[i - 1]:
            obv.append(obv[-1] - volume[i])
        else:
            obv.append(obv[-1])
    data['obv'] = obv
    return data

def adx_strategy(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    data = data.copy()
    prev_high = data['High'].shift(1)
    prev_low = data['Low'].shift(1)
    prev_close = data['Close'].shift(1)

    tr = pd.concat([
        data['High'] - data['Low'],
        (data['High'] - prev_close).abs(),
        (data['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)

    plus_dm = data['High'] - prev_high
    minus_dm = prev_low - data['Low']

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr = tr.ewm(alpha=1/window, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/window, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/window, adjust=False).mean() / atr)

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    data['adx'] = dx.ewm(alpha=1/window, adjust=False).mean()
    data['plus_di'] = plus_di
    data['minus_di'] = minus_di
    return data

def aroon_strategy(data: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    data = data.copy()
    aroon_up = []
    aroon_down = []

    for i in range(len(data)):
        if i < window - 1:
            aroon_up.append(np.nan)
            aroon_down.append(np.nan)
        else:
            high_idx = data['High'].iloc[i - window + 1:i + 1].values.argmax()
            low_idx = data['Low'].iloc[i - window + 1:i + 1].values.argmin()
            aroon_up.append(((window - 1 - high_idx) / window) * 100)
            aroon_down.append(((window - 1 - low_idx) / window) * 100)

    data['aroon_up'] = aroon_up
    data['aroon_down'] = aroon_down
    data['aroon_oscillator'] = data['aroon_up'] - data['aroon_down']
    return data
