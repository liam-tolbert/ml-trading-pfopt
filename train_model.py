import datetime
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
import lib
import importlib
importlib.reload(indicators)
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from collections import OrderedDict

# 1. Data download + prep

stocks = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "AMD", "INTC",
    "JPM", "GS", "BAC", "C", "WFC", "V", "MA", "AXP", "BRK-B",
    "UNH", "JNJ", "PFE", "LLY", "ABBV", "TMO", "DHR", "BMY", "GILD",
    "XOM", "CVX", "COP", "OXY", "SLB", "HAL", "BP", "SHEL", "EOG",
    "WMT", "COST", "HD", "LOW", "MCD", "SBUX", "TGT", "NKE", "PG", "KO",
    "ADBE", "CRM", "ORCL", "QCOM", "AVGO", "TXN", "INTU", "CSCO", "IBM", "MU",
    "PANW", "NOW", "CDNS", "ANET", "LRCX", "SNPS", "ACN", "FTNT", "MSI", "ZM",
    "BLK", "TFC", "USB", "BK", "SCHW", "SPGI", "MS", "ICE", "PGR", "AIG",
    "CB", "MMC", "MET", "ALL", "PRU", "CME", "COF", "DFS", "FITB", "MTB",
    "ABT", "MRK", "ZBH", "ISRG", "MDT", "CVS", "CI", "REGN", "VRTX", "SYK",
    "PSX", "MPC", "VLO", "KMI", "WMB", "CTRA", "DVN", "HES", "FANG",
    "LIN", "APD", "ECL", "SHW", "NEM", "FCX", "DD", "ALB", "LYB", "MLM",
    "PEP", "CL", "KMB", "EL", "MNST", "KR", "DG", "DLTR", "GIS", "MDLZ",
    "TSCO", "ROST", "TJX", "YUM", "WBA", "PM", "MO", "UL", "HSY", "HRL",
    "CAT", "DE", "HON", "LMT", "RTX", "BA", "GE", "NOC", "ETN", "EMR",
    "UNP", "NSC", "FDX", "UPS", "CSX", "WM", "EXC", "DUK", "NEE",
    "PLD", "AMT", "CCI", "EQIX", "SPG", "PSA", "O", "DLR", "VTR", "ARE"
]

# Alternatively, use stocks2 if you want to
stocks2 = [
    "ADBE", "CRM", "ORCL", "SAP", "NOW", "SHOP", "SQ", "ZM", "CRWD", "DDOG",
    "TXN", "QCOM", "AVGO", "MU", "LRCX", "KLAC", "NXPI", "ADI", "MRVL", "SWKS",
    "PYPL", "INTU", "FISV", "ADP", "VEEV", "TEAM", "WDAY", "ZS", "OKTA", "MDB",
    "T", "VZ", "TMUS", "CHTR", "CMCSA", "DIS", "ROKU", "LYV", "TTWO", "ATVI",
    "PEP", "KMB", "CL", "HSY", "MDLZ", "GIS", "MO", "PM", "EL", "STZ"
]

# Uncomment this if you want to refresh the dataset(s)
"""
x = yf.download(stocks, start="2015-01-01", end="2025-04-15", interval="1wk")

sp500 = yf.download("^GSPC", start="2015-01-01", interval="1wk")
sp500 = sp500.resample('W-FRI').agg({
      ('Open', "^GSPC"): 'first',
      ('High', "^GSPC"): 'max',
      ('Low', "^GSPC"): 'min',
      ('Close', "^GSPC"): 'last',
      ('Volume', "^GSPC"): 'sum'
    })
sp500.columns = ["_".join(col) if isinstance(col, tuple) else col for col in sp500.columns]

sp500 = sp500.loc[:, ~sp500.columns.isin(["level_0", "index"])]
sp500.to_csv("SP500.csv")
"""

'''
stockData = x.copy()

# Rename columns, joining multi-level column names into a single string with "_".
stockData.columns = ["_".join(col) if isinstance(col, tuple) else col for col in stockData.columns]

# Remove unnecessary columns such as "level_0" or "index" that may have been carried over.
stockData = stockData.loc[:, ~stockData.columns.isin(["level_0", "index"])]
stockData.to_csv("50stocks.csv")
'''

sp500 = lib.get_sp500('2000-01-01') # it's also needed in predict.py, so I put it in lib

# 2. Cleaning data, train/val/test split
df = lib.create_stock_features(stocks, "50stocks.csv", sp500)

def label_signal(return_val, buy_thresh=0.01, sell_thresh=-0.01):
    if return_val > buy_thresh:
        return 0
    elif return_val < sell_thresh:
        return 1
    else:
        return 2

df['Signal'] = df['Returns-2wk'].apply(label_signal)

# Sort by date?
df = df.sort_values(by='Date')

# 70% train, 15% val, 15% test
split_1 = int(len(df) * 0.7)
split_2 = int(len(df) * 0.85)

train = df.iloc[:split_1]
val = df.iloc[split_1:split_2]
test = df.iloc[split_2:]

X_train = train[lib.features]
y_train = train['Signal']
X_val = val[lib.features]
y_val = val['Signal']

# tuning class weights b/c the val set has underrepresented sell orders
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))
print("Class Weights:", class_weight_dict)

# Map class weights to each sample in training set
sample_weights = y_train.map(class_weight_dict)
# Increase sample weights when Bull_Probability is extreme (close to 0 or 1)
bull_prob_multiplier = X_train['Bull_Probability'].apply(lambda p: 2.0 if p < 0.2 or p > 0.8 else 1.0)
sample_weights = sample_weights * bull_prob_multiplier

# Train the model
model = xgb.XGBClassifier(
    objective='multi:softprob',  # for multi-class
    num_class=len(classes),
    eval_metric='mlogloss',
    tree_method='hist',
    subsample=0.6,
    colsample_bytree=0.9,
    reg_lambda=5,
    reg_alpha=5,
    booster='gbtree',
    learning_rate=0.01,
    gamma=0.5,
    min_child_weight=5,
    n_estimators=200,
    use_label_encoder=False,
    random_state=42,
    max_depth = 12,
)

model.fit(X_train, y_train, sample_weight=sample_weights)

X_test = test[lib.features]
y_test = test['Signal']

y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))

filename = "model.pkl"
try:
    model.save_model(filename)
except Exception as e:
    print("Error saving model: " + str(e))

print("Model saved as " + filename)