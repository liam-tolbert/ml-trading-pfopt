from collections import OrderedDict

# imports
import numpy as np
import pandas as pd
import yfinance as yf
import indicators
import lib
import xgboost as xgb
from sklearn.metrics import classification_report

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

sp500 = lib.get_sp500('2000-01-01') # it's also needed in predict.py, so I put it in lib

# 2. Cleaning data, train/val/test split
df = lib.create_stock_features(stocks, "data/50stocks.csv", sp500)

def label_signal(row):
    r1 = row['Returns-future-1wk']
    if r1 > 0.01:
        return 1   # Buy
    elif r1 < -0.01:
        return 0   # Sell
    else:
        return np.nan  # drop — ambiguous

df['Signal'] = df.apply(label_signal, axis=1)
df = df.dropna(subset=['Returns-future-1wk', 'Returns-future-2wk', 'Signal'])
df['Signal'] = df['Signal'].astype(int)

# Sort by date
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

# Train the model
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='hist',
    max_depth=4,
    n_estimators=150,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2,
    reg_alpha=1,
    min_child_weight=20,
    use_label_encoder=False,
    random_state=42,
)

model.fit(X_train, y_train)

X_test = test[lib.features]
y_test = test['Signal']

y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))

filename = "data/model.pkl"
try:
    model.save_model(filename)
except Exception as e:
    print("Error saving model: " + str(e))

print("Model saved as " + filename)