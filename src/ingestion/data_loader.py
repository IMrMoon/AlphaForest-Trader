import yfinance as yf
import pandas as pd
from abc import ABC, abstractmethod
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config_loader import load_config

class DataSource(ABC):
    @abstractmethod
    def fetch_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        pass

class YFinanceSource(DataSource):
    def fetch_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        print(f"Downloading OHLCV data for {len(tickers)} tickers from yfinance...")
        df = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
        
        if isinstance(df.columns, pd.MultiIndex):
            df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index()
        else:
            df = df.reset_index()
            df['Ticker'] = tickers[0]
            
        required_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                df[col] = pd.NA
                
        return df[required_cols]

def get_data_source(source_name: str) -> DataSource:
    if source_name == "yfinance":
        return YFinanceSource()
    else:
        raise ValueError(f"Unknown data source: {source_name}")

def check_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    missing_values = df[numeric_cols].isnull().sum().sum()
    if missing_values > 0:
        print(f"Warning: Found {missing_values} missing values. Applying forward fill per ticker...")
        # מילוי חוסרים פר-מניה כדי למנוע זליגת נתונים ממניה אחת לאחרת
        df[numeric_cols] = df.groupby('Ticker')[numeric_cols].ffill()
        if df[numeric_cols].isnull().sum().sum() > 0:
            print("Dropping remaining NaNs...")
            df = df.dropna(subset=numeric_cols)
    else:
        print("Data quality check passed: No missing values.")
    return df