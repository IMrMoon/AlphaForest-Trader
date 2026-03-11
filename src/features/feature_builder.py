import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    features_list = []
    for ticker, group in df.groupby('Ticker'):
        group = group.sort_values('Date').copy()
        
        group['Return_1d'] = group['Close'].pct_change(1)
        group['SMA_10'] = group['Close'].rolling(window=10).mean()
        group['SMA_20'] = group['Close'].rolling(window=20).mean()
        group['Momentum_10d'] = group['Close'] / group['Close'].shift(10) - 1
        
        delta = group['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        group['RSI_14'] = 100 - (100 / (1 + rs))
        
        exp1 = group['Close'].ewm(span=12, adjust=False).mean()
        exp2 = group['Close'].ewm(span=26, adjust=False).mean()
        group['MACD'] = exp1 - exp2
        
        std_20 = group['Close'].rolling(window=20).std()
        group['BB_upper'] = group['SMA_20'] + (std_20 * 2)
        group['BB_lower'] = group['SMA_20'] - (std_20 * 2)
        group['BB_width'] = (group['BB_upper'] - group['BB_lower']) / group['SMA_20']
        
        group['Daily_Volatility'] = (group['High'] - group['Low']) / group['Close']
        avg_volume_20 = group['Volume'].rolling(window=20).mean()
        group['Volume_Surge'] = np.where(avg_volume_20 > 0, group['Volume'] / avg_volume_20, 1.0)
        
        features_list.append(group)
    
    all_features = pd.concat(features_list, ignore_index=True)
    all_features['Market_Return_1d'] = all_features.groupby('Date')['Return_1d'].transform('mean')
    
    return all_features.dropna()