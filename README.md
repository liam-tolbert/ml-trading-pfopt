# Stock Trading and Portfolio Optimization using Technical Analysis and Machine Learning

1. Overview
This project seeks to combine technical indicators, regime switching models, machine learning, and portfolio optimization to build a stock recommendation and weight allocation system. By providing a list of stocks to invest in, a user will receive recommendations on which stocks to either buy, hold, or sell. Additionally, the user will receive a recommendation on how much money to spend on these assets as a percentage of the total money the user plans to spend. As stock movements change, the model will update its predictions every week based on the new data. 

2. Methodology
2.1	Data Acquisition
Historical stock prices for a set of technology and financial stocks are downloaded using the yfinance package. Historical prices start from January 1st, 2015, until the current date, which the user sets. The user can specify which tickers are used, and can change them anytime. Ideally, the user will set the tickers to whichever stocks they have long positions in. 
2.2	Financial Tools Utilized
This project uses two well-established financial models: Hamilton’s Regime Switching Model  and the Markowitz Portfolio Optimization Model . The model also utilizes many technical indicators, such as:
-	Simple Moving Average (SMA)
-	Relative Strength Index (RSI)
-	Moving Average Convergence-Divergence (MACD)
-	 Bollinger Bands
-	Average True Range (ATR)
-	Stochastic Oscillator 
-	On-Balance Volume (OBV)
-	Average Directional Index (ADX)
-	Aroon Indicator

2.3	Feature Engineering
The project reads multi-ticker CSV files (e.g., generated from yfinance) using a custom function extract_ticker_name(). This function reformats the data to prepare it for feature engineering. Next, a Hamilton regime switching model is applied to the S&P 500 data. The Hamilton model outputs a bull market probability, which is then used as an input feature. Then, various technical indicators are computed including SMA, RSI, MACD, Bollinger Bands, ATR, Stochastic Oscillator, OBV, ADX, and Aroon. 
Raw signal values were used for indicators such as SMA, RSI, MACD, ADX, ATR, and Aroon. In contrast, Bollinger Bands and the Stochastic Oscillator produced ternary signals: -1 (sell), 0 (hold), and 1 (buy). Although it would be preferable to maintain raw values for consistency, these two indicators inherently generate signals based on inequality conditions. For example, Bollinger Bands define an upper and lower bound around an n-day simple moving average, offset by the standard deviation of prices. A signal of 1 is triggered when the closing price falls below the lower band, -1 when it exceeds the upper band, and 0 otherwise. Since these signals depend on threshold comparisons rather than continuous output, the raw band values offer limited additional insight.
To capture past stock percent return trends in any given week, additional features include a four-week percent return split into two segments: a three-week percent return with a one-week lag from the present, and a one-week percent return with no lag from the present. The target variable for the machine learning algorithm is the two-week percent return on the stock price. Finally, all these features and variables for each stock the user wishes to include are then assembled into one large pandas dataframe, which acts as the main dataset the machine learning model trains, validates, and tests from. 
2.4 Machine Learning Model
An XGBoost classifier (XGBClassifier) is trained to predict a signal (Hold / Buy / Sell) based on the above features. Since sell signals are underrepresented in the validation set, class weights are tuned using compute_class_weight. The train/validate/test ratio for the features dataset is 70% / 15% / 15%. 
XGBoost is a suitable modeling choice for this project due to several key properties aligned with the demands of financial time series predictions. First, it effectively captures complex non-linear interactions among multiple technical indicators, a common feature in financial data. Second, its regularization techniques help mitigate overfitting. Additionally, XGBoost is well-known for its efficiency and scalability, making it well-suited for large datasets. The algorithm also supports sample weighting, enabling it to address class imbalance issues such as the underrepresentation of sell signals. Finally, XGBoost offers native multi-class classification capability, which aligns with the prediction structure (buy, hold, sell) employed in this work. 
2.4	Portfolio Optimization
Once the model generates recommendations, the project isolates stocks for which the predicted signal is “Buy.” Historical price data for the selected stocks is used to compute the baseline expected returns (mu) and sample covariance matrix (S) using the pypfopt library. The baseline expected returns are adjusted based on the probability estimated by the classifier by giving additional weighting to stocks with a higher buy probability. Portfolio allocation is computed by maximizing the Sharpe ratio using EfficientFrontier.
Once the mean variance portfolio is found, the model will output a Python dictionary of stock tickers mapped with their weights. Only tickers with nonzero weights will be included in this dictionary. 

5. Limitations and Future Improvements
First, the current model’s sell detection is not as robust as desired. Potential remedies include even stronger class weighting in the objective or enriching the feature set with indicators that more reliably and sharply delineate downturns. Addressing these issues should help raise recall and precision for sell signals. 
Second, the yfinance package is not a viable long-term source of data. First, when the API senses greater than usual traffic, it seems to rate-limit all users. Second, when I attempted to bulk download weekly data beginning before 2015, the API began offsetting the dates by one day for a few stocks. As dates act as indexes in the Pandas dataframes from yfinance, the downloaded data contained massive rows filled with NaNs because most other stocks had the correct dates. Finding a workaround within or without yfinance should happen to have access to more data to increase model accuracy.  Third, yfinance does not provide fundamental data, which should have been a large part of the feature set. Fundamental analysis metrics such as P/E ratio would have been massively useful, but they are all locked behind paid APIs and are not provided by yfinance. Perhaps in the future they will be included. 
Lastly, while the model can use the Markowitz model to recommend how much of a stock to buy, it cannot yet recommend how much of a stock to sell. More research is necessary to find other techniques or models to provide those recommendations. 

