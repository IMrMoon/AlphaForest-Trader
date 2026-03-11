import pandas as pd
import joblib
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config_loader import load_config
from src.ingestion.data_loader import get_data_source, check_data_quality
from src.features.feature_builder import create_features

def get_daily_signals(tickers: list = None) -> pd.DataFrame:
    config = load_config()
    
    if tickers is None or len(tickers) == 0:
        tickers = config.data.tickers
        
    print(f"Initializing Daily Predictor for {len(tickers)} tickers...")
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    print(f"Fetching recent market data from {start_date} to {end_date}...")
    source = get_data_source(config.data.data_source)
    raw_data = source.fetch_data(tickers, start_date=start_date, end_date=end_date)
    
    if raw_data.empty:
        print("Error: No data fetched.")
        return pd.DataFrame()
        
    clean_data = check_data_quality(raw_data)
    
    print("Calculating technical features...")
    features_df = create_features(clean_data)
    
    latest_date = features_df['Date'].max()
    print(f"Generating signals based on market close of: {latest_date.date()}")
    
    latest_data = features_df[features_df['Date'] == latest_date].copy()
    features = ['Return_1d', 'SMA_10', 'SMA_20', 'Momentum_10d', 'RSI_14', 'MACD', 'BB_width', 'Market_Return_1d', 'Daily_Volatility', 'Volume_Surge']
    
    if latest_data.empty:
        print("Error: Not enough data to calculate features for the latest date.")
        return pd.DataFrame()
        
    X_live = latest_data[features]
    
    print("Loading trained Random Forest model...")
    model_path = "data/model.pkl"
    if not os.path.exists(model_path):
        print("Error: Trained model not found. Please run trainer.py first.")
        return pd.DataFrame()
        
    model = joblib.load(model_path)
    probs = model.predict_proba(X_live)[:, 1]
    latest_data['Buy_Probability'] = probs
    
    output_cols = ['Ticker', 'Close', 'Buy_Probability', 'RSI_14', 'Volume_Surge', 'Daily_Volatility']
    results_df = latest_data[output_cols].sort_values(by='Buy_Probability', ascending=False).reset_index(drop=True)
    
    return results_df

if __name__ == "__main__":
    print("Testing Backend Logic...")
    test_results = get_daily_signals()
    print("\n--- Final Output DataFrame ---")
    print(test_results)