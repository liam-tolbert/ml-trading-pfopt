from collections import OrderedDict

# imports
import numpy as np
import pandas as pd
import yfinance as yf
import indicators
import lib
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

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
    r1 = row['Returns-1wk-0wklag']
    r2 = row['Returns-2wk']
    if r1 < -0.01 and r2 < -0.015:
        return 0  # Sell
    elif r1 > 0.01 and r2 > 0.015:
        return 1  # Buy
    else:
        return 2  # Hold

df['Signal'] = df.apply(label_signal, axis=1)

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