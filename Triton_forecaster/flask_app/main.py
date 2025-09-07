import os
import io
import base64
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from flask import Flask, jsonify, render_template, request
from newsapi import NewsApiClient
import tritonclient.http as httpclient
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")

STOCKS = {
    "Technology": {
        "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL", "Amazon": "AMZN",
        "NVIDIA": "NVDA", "Tesla": "TSLA", "Meta": "META"
    },
    "Finance": {
        "JPMorgan Chase": "JPM", "Bank of America": "BAC", "Wells Fargo": "WFC",
        "Goldman Sachs": "GS", "Visa": "V", "Mastercard": "MA"
    },
    "Healthcare": {
        "Johnson & Johnson": "JNJ", "UnitedHealth": "UNH", "Pfizer": "PFE",
        "Eli Lilly": "LLY", "Merck": "MRK"
    },
    "Consumer Goods": {
        "Procter & Gamble": "PG", "Coca-Cola": "KO", "PepsiCo": "PEP",
        "Walmart": "WMT", "Costco": "COST"
    },
    "Energy": { "ExxonMobil": "XOM", "Chevron": "CVX" }
}

def get_sentiment_from_triton(headlines):
    triton_client = httpclient.InferenceServerClient(url="192.168.200.10:8000", verbose=False)
    if not headlines: return 0.0
    headlines_np = np.array(headlines, dtype=object)
    input_tensor = httpclient.InferInput("TEXT_INPUT", [len(headlines)], "BYTES")
    input_tensor.set_data_from_numpy(headlines_np)
    response = triton_client.infer("vader_sentiment", [input_tensor])
    sentiment_scores = response.as_numpy("SENTIMENT_SCORE")
    mean_score = np.mean(sentiment_scores) if sentiment_scores is not None else 0.0
    return float(mean_score)

def get_prediction_from_triton(features):
    triton_client = httpclient.InferenceServerClient(url="192.168.200.10:8000", verbose=False)
    features_np = np.array([features], dtype=np.float32)
    input_tensor = httpclient.InferInput("input__0", features_np.shape, "FP32")
    input_tensor.set_data_from_numpy(features_np)
    response = triton_client.infer("xgboost_predictor", [input_tensor])
    prediction_proba = response.as_numpy("output__0")[0][0]
    proba_float = float(prediction_proba)
    return "UP" if proba_float > 0.5 else "DOWN"

def get_latest_features(ticker, sentiment_score):
    data = yf.download(ticker, period="100d", interval="1d")
    if data.empty: return None, None
    data.columns = data.columns.get_level_values(0)
    data.ta.rsi(length=14, append=True)
    data.ta.macd(fast=12, slow=26, signal=9, append=True)
    data.ta.bbands(length=20, append=True)
    data.ta.obv(append=True)
    data.ta.sma(length=50, append=True, col_prefix="SMA")
    latest_data = data.iloc[-1]
    feature_values = [
        latest_data.get('RSI_14', 0), latest_data.get('MACD_12_26_9', 0),
        latest_data.get('MACDh_12_26_9', 0), latest_data.get('MACDs_12_26_9', 0),
        latest_data.get('BBL_20_2.0', 0), latest_data.get('BBM_20_2.0', 0),
        latest_data.get('BBU_20_2.0', 0), latest_data.get('OBV', 0),
        latest_data.get('SMA_50', 0), latest_data.get('Volume', 0), sentiment_score
    ]
    return feature_values, data

def create_price_chart(data, ticker, prediction):
    """Creates a price chart and returns it as a Base64 string."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    
    line_color = '#3fb950' if prediction == 'UP' else '#f85149'
    high_color = '#3fb950'
    low_color = '#f85149'
    
    ax.plot(data.index, data['Close'], label=f'{ticker} Close Price', color=line_color, linewidth=2)
    
    high_price = data['Close'].max()
    low_price = data['Close'].min()
    high_date = data['Close'].idxmax()
    low_date = data['Close'].idxmin()
    
    y_buffer = (high_price - low_price) * 0.15 
    ax.set_ylim(low_price - y_buffer, high_price + y_buffer)

    ax.annotate(f'High: ${high_price:.2f}', xy=(high_date, high_price),
                textcoords="offset points", xytext=(0, 15),
                arrowprops=dict(arrowstyle="-|>", color=high_color, lw=1.5),
                color=high_color, ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="#0D1117", ec=high_color, lw=0.5, alpha=0.8))
    
    ax.annotate(f'Low: ${low_price:.2f}', xy=(low_date, low_price),
                textcoords="offset points", xytext=(0, -25),
                arrowprops=dict(arrowstyle="-|>", color=low_color, lw=1.5),
                color=low_color, ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="#0D1117", ec=low_color, lw=0.5, alpha=0.8))

    ax.set_title(f'{ticker} Price History (100 Days)', color='white', fontsize=16, loc='left') # Title aligned left
    ax.set_ylabel('Price (USD)', color='white', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#30363d')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1.12), frameon=False, fontsize=10)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True)
    plt.close(fig)
    buf.seek(0)
    
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

@app.route('/')
def index():
    return render_template('index.html', stocks_data=STOCKS)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = data.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker symbol is required"}), 400

    newsapi_client = NewsApiClient(api_key=NEWS_API_KEY)
    try:
        all_articles = newsapi_client.get_everything(q=ticker, language='en', sort_by='publishedAt', page_size=20)
        headlines = [article['title'] for article in all_articles['articles']]
    except Exception as e:
        print(f"Could not fetch news: {e}")
        headlines = []

    sentiment_score = get_sentiment_from_triton(headlines)
    features, price_data = get_latest_features(ticker, sentiment_score)
    
    if features is None:
        return jsonify({"error": f"Could not retrieve market data for {ticker}"}), 500
        
    prediction = get_prediction_from_triton(features)
    chart_image_url = create_price_chart(price_data, ticker, prediction)
    
    return jsonify({
        "ticker": ticker,
        "prediction": prediction,
        "sentiment_score": float(sentiment_score),
        "chart_url": chart_image_url
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)