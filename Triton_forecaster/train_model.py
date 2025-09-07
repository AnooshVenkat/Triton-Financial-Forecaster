import pandas as pd
import pandas_ta as ta
import yfinance as yf
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("--- Starting Model Training ---")

print("Fetching historical data for AAPL...")
stock_data = yf.download("AAPL", start="2015-01-01", end="2023-12-31")
stock_data.columns = stock_data.columns.get_level_values(0)
print("Engineering features using pandas-ta...")
stock_data.ta.rsi(length=14, append=True)
stock_data.ta.macd(fast=12, slow=26, signal=9, append=True)
stock_data.ta.bbands(length=20, append=True)
stock_data.ta.obv(append=True)
stock_data.ta.sma(length=50, append=True, col_prefix="SMA") 

stock_data['Weekly_Change'] = stock_data['Close'].shift(-5) - stock_data['Close']
stock_data['Target'] = (stock_data['Weekly_Change'] > 0).astype(int)

stock_data.dropna(inplace=True)
stock_data['sentiment_score'] = 0.0

feature_names = [
    'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'OBV', 'SMA_50', 'Volume',
    'sentiment_score'
]
print(f"Using {len(feature_names)} features for training in this exact order: {feature_names}")

X = stock_data[feature_names]
y = stock_data['Target']

print("Splitting data and training XGBoost Classifier...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

print("Evaluating model performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.4f}")

print("Saving model to 'model.json'...")
model.save_model("model.json")

print("--- Model Training Complete ---")
print("You can now move the 'model.json' file to 'model_repository/xgboost_predictor/1/'")