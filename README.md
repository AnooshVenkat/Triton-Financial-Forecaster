# Triton Market Pulse 📈

Welcome to Triton Market Pulse! This is a full-stack web application designed to predict the weekly direction of stock prices by combining technical analysis with real-time sentiment analysis of financial news.

The entire machine learning pipeline is served using **NVIDIA's Triton Inference Server**, demonstrating a robust, high-performance, and scalable microservices architecture suitable for production environments.

---

## Features

* **AI-Powered Prediction:** Utilizes a sophisticated **XGBoost** model to forecast whether a stock will close higher or lower than the previous week.
* **Real-Time Sentiment Analysis:** Fetches the latest financial news for a given stock and uses a custom **Python model (VADER)** to analyze the sentiment of headlines, providing a score of the overall market mood.
* **High-Performance Inference:** All ML models are decoupled from the main application and served with **NVIDIA Triton Inference Server** for low-latency, high-throughput predictions.
* **Dynamic Visualizations:** Generates a clear and informative **Matplotlib** chart of the stock's 100-day price history, complete with high/low markers and a prediction-colored trend line.
* **Interactive Frontend:** A clean, modern UI allows users to select stocks from categorized dropdowns (by industry).
* **Containerized & Reproducible:** The entire application stack (Frontend, Backend, and Inference Server) is containerized with **Docker**, allowing for a simple, one-command setup.

---

## Architecture

Triton Market Pulse is built on a decoupled microservices architecture, which separates the web application logic from the machine learning inference. This is a standard production pattern that allows for independent scaling and updating of components.


1.  **Browser (Frontend):** The user interacts with the HTML, CSS, and JavaScript interface to select a stock and request a prediction.
2.  **Flask App (Backend Orchestrator):** A Python Flask server receives the request. It is responsible for:
    * Fetching financial news from **NewsAPI**.
    * Fetching historical market data from **yfinance**.
    * Orchestrating calls to the Triton Inference Server.
    * Generating the price chart image.
    * Serving the final results back to the frontend.
3.  **Triton Inference Server:** A dedicated, high-performance server that hosts our two machine learning models. It runs in its own Docker container and exposes endpoints for inference.
    * **VADER Sentiment Model:** A custom Python model that receives news headlines and returns an average sentiment score.
    * **XGBoost Predictor Model:** A Python model that receives the prepared market data (10 technical indicators + 1 sentiment score) and returns the weekly price direction forecast.

---

## Tech Stack

* **Inference Server:** NVIDIA Triton Inference Server
* **Backend:** Python, Flask
* **Machine Learning:** XGBoost, Scikit-learn, pandas-ta
* **Data & APIs:** yfinance (Yahoo Finance), NewsAPI
* **Frontend:** HTML, CSS, Vanilla JavaScript
* **Visualization:** Matplotlib
* **Containerization:** Docker, Docker Compose

---

## 🚀 Getting Started

Follow these steps to get the application running on your local machine.

### Prerequisites

* **Docker Desktop:** Ensure Docker is installed and running on your system.
* **Python 3.8+:** Required for running the one-time model training script.
* **A NewsAPI Key:** Sign up for a free API key at [NewsAPI.org](https://newsapi.org/).

### Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/AnooshVenkat/Triton-Financial-Forecaster
    cd Triton_Forecaster
    ```

2.  **Configure Your API Key (Choose One Method):**

    * **(Recommended) Using a `.env` file:**
        1.  In the root directory of the project, create a file named `.env`.
        2.  Add your API key to this file like so:
            ```
            NEWS_API_KEY=YOUR_ACTUAL_API_KEY_HERE
            ```
        Docker Compose will automatically pick up this variable.

    * **(Alternative) Using a Terminal Environment Variable:**
        Before running `docker-compose`, set the environment variable in your terminal.
        * **macOS / Linux:**
            ```bash
            export NEWS_API_KEY="YOUR_ACTUAL_API_KEY_HERE"
            ```
        * **Windows (PowerShell):**
            ```powershell
            $env:NEWS_API_KEY="YOUR_ACTUAL_API_KEY_HERE"
            ```

3.  **Train the Prediction Model:**
    Before launching the application, you need to train the XGBoost model. This is a one-time step.
    * Create and activate a Python virtual environment.
    * Install the required packages: `pip install -r flask_app/requirements.txt`.
    * Run the training script:
        ```bash
        python train_model.py
        ```
    * This will create a `model.json` file. Move this file into the correct directory for Triton to find it:
        `mv model.json model_repository/xgboost_predictor/1/`

### Running the Application

Once the model is trained and in place, you can launch the entire application stack with a single command from the project's root directory:

```bash
docker-compose up --build
```

### Accessing the App
Open your web browser and navigate to:
```bash
http://localhost:5000
```
You should now see the application running! Select an industry and a company, then click "Predict" to see the analysis.
