# 📈 Algorithmic Trading AI System

An end-to-end Machine Learning pipeline for algorithmic trading. This system fetches financial data, engineers technical indicators, trains an optimized Random Forest model, manages risk via backtesting, and serves predictions through an interactive Streamlit dashboard.

---

## 🧠 Architecture & The Machine Learning Journey

Building a trading AI is fundamentally different from traditional machine learning tasks. Financial data is notoriously noisy, non-stationary, and prone to extreme overfitting. Here is the analytical breakdown of the decisions made throughout the project:

### 1. Data Ingestion & Preprocessing
* **The Source:** We pull raw OHLCV (Open, High, Low, Close, Volume) data directly from `yfinance`.
* **Handling Missing Data:** Missing days (due to holidays or API drops) are strictly handled using **Forward Fill (ffill)** per individual ticker. We never interpolate or use future data, completely eliminating the risk of Data Leakage (peeking into the future).
* **Data Formatting:** The dataset is melted into a "Long Format". This ensures our pipeline can seamlessly scale from 1 stock to the entire index without structural changes.

### 2. Feature Engineering (The Variables)
Instead of feeding raw prices to the model (which leads to memorization), we engineered normalized momentum and volatility indicators:
* **Trend & Momentum:** `SMA_10`, `SMA_20`, `Momentum_10d`, `MACD`.
* **Mean Reversion:** `RSI_14` and `Bollinger Bands Width`.
* **Normalized Volume & Volatility:** We use `Volume_Surge` (current volume divided by 20-day average) instead of raw volume, and `Daily_Volatility`. This normalization allows the model to treat massive blue-chip stocks and small-cap stocks on the exact same mathematical scale.

### 3. Timeframes: The Lookback and The Horizon
* **The Lookback (60 Days):** We only feed the last 60 days of data to the daily predictor. This is the exact minimum required to cleanly calculate the 20-day moving averages and the 14-day RSI, keeping the production pipeline lightweight and fast.
* **The Target Horizon (5 Days):** We trained the model to predict exactly 5 days ahead (Swing Trading). Predicting 1 day ahead is too heavily influenced by random daily noise, while predicting 30 days ahead exposes the trade to unpredictable macroeconomic events. 5 days captures the "sweet spot" of technical momentum.

### 4. The Multi-Stock Universe
Instead of training a model on a single stock, we train on a basket of leading index stocks (TA-35). 
* **The Single-Stock Trap:** Training on one stock causes the model to memorize its specific historical timeline. 
* By feeding multiple stocks, the model is forced to learn universal market mechanics. It also allows us to calculate the `Market_Return_1d` feature, giving the model crucial context about the broader market's daily direction.

### 5. Model Selection: The Pivot to Random Forest

* **Our First Attempt (Gradient Boosting / LightGBM):** We initially started with a boosting algorithm. However, boosting builds trees sequentially, trying to constantly correct the errors of previous trees. In the stock market, most "errors" are just random noise. The boosting model learned the noise perfectly, resulting in massive overfitting and catastrophic backtest failures.
* **The Solution (Random Forest):** We pivoted to a Bagging approach. Random Forest builds hundreds of independent, parallel trees on random subsets of data. By averaging them out, it naturally smooths out the market's noise, providing a much more robust and stable prediction.

### 6. The Ultimate Metric: Why ROC-AUC?
In quantitative finance, standard metrics are often misleading:
* **Accuracy** is deceiving because market returns are imbalanced (a model can just predict "no movement" and achieve decent accuracy without making any trades).
* **Precision/Recall (F1)** requires an arbitrary probability cutoff (usually 0.5), which we don't use.
* **ROC-AUC (Area Under the Curve)** evaluates the model's pure *ranking ability* across all probabilities. It tells us: *If we pick one random profitable day and one random losing day, what is the chance the model ranked the profitable day higher?* A score above 0.52-0.54 is a genuine statistical edge, which we then exploit using a strict 0.55 probability threshold in production.

---

## 🌍 Switching Markets: From Israel (TA-35) to USA (S&P 500)

The system is designed to be market-agnostic. You can easily switch from Israeli stocks to US stocks by following these steps:

### 1. Update Configuration
Open `config/base.yaml` and modify the `tickers` list.
* **For Israeli Stocks:** Use the `.TA` suffix (e.g., `LUMI.TA`, `NICE.TA`).
* **For US Stocks:** Use standard symbols (e.g., `AAPL`, `TSLA`, `SPY`, `MSFT`).

### 2. The "Golden Rule": Do Not Mix Timezones
The model calculates a `Market_Return_1d` feature based on the average return of all tickers in the current session. 
* **Avoid mixing markets** with different trading days (e.g., Sunday-Thursday in Israel vs. Monday-Friday in the US). 
* Mixing markets will result in missing data (`NaNs`) on mismatched days, leading to inaccurate predictions.
* **Recommendation:** Create separate config files or clear the UI's active list before switching between global markets.

### 3. Retraining for New Markets
While the model is robust, market dynamics differ (volatility in NASDAQ is higher than in TA-35). For best results:
1. Update the `tickers` in `base.yaml`.
2. Run `uv run build_dataset.py` to fetch new historical data.
3. Run `uv run src/models/trainer.py` to allow the Random Forest to adapt to the new market's volatility and volume patterns.

---

## 🚀 Features

* **Machine Learning Pipeline:** Complete lifecycle from `yfinance` ingestion to `scikit-learn` modeling.
* **Hyperparameter Tuning:** Automated optimization using `Optuna`.
* **Experiment Tracking:** Integrated `MLflow` for logging parameters and ROC-AUC scores.
* **Realistic Risk Management:** Backtesting engine includes double-sided broker commissions (0.1%) and a strict Stop-Loss mechanism (5%), converting raw probabilities into real net-profit metrics.
* **Interactive UI:** A dynamic `Streamlit` dashboard with Session State management for live daily predictions.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Package Manager:** `uv`
* **Machine Learning:** Scikit-Learn, Optuna, MLflow
* **Data Processing:** Pandas, NumPy
* **Frontend:** Streamlit

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/trading_ai.git](https://github.com/YOUR_USERNAME/trading_ai.git)
   cd trading_ai

2. **Create a virtual environment and install dependencies (using uv):**
    ```bash
    uv venv
    uv pip install -r requirements.txt

3. **Build the initial dataset and train the model:**
    ```bash
    uv run build_dataset.py
    uv run src/models/trainer.py

4. **Launch the Dashboard:**
    ```bash
    uv run streamlit run ui/app.py

## ⚠️ Disclaimer
This project is for educational and portfolio purposes only. It does not constitute financial advice. Algorithmic trading involves significant risk, and the author is not responsible for any financial losses incurred from using this software.
