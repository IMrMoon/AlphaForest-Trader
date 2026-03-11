import pandas as pd

def create_targets(df: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    targets_list = []
    for ticker, group in df.groupby('Ticker'):
        group = group.sort_values('Date').copy()
        future_returns = group['Close'].pct_change(periods=horizon).shift(-horizon)
        
        targets = (future_returns > threshold).astype(int)
        group['Target'] = targets
        group.loc[future_returns.isna(), 'Target'] = pd.NA
        
        targets_list.append(group[['Date', 'Ticker', 'Target']])
        
    all_targets = pd.concat(targets_list, ignore_index=True)
    return all_targets.dropna()