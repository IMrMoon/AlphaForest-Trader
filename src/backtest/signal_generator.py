import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config_loader import load_config

def run_backtest():
    config = load_config()
    print("Running realistic backtest with Stop Loss and Commissions...")
    preds_df = pd.read_csv("data/test_predictions.csv")
    full_df = pd.read_csv("data/training_data.csv")
    
    preds_df['Date'] = pd.to_datetime(preds_df['Date'])
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    full_df = full_df.sort_values('Date')
    
    entry_threshold = preds_df['Predict_Proba'].quantile(0.80)
    preds_df['Signal'] = (preds_df['Predict_Proba'] > entry_threshold).astype(int)
    trades = preds_df[preds_df['Signal'] == 1].copy()
    
    stop_loss = config.data.stop_loss_pct
    commission = config.data.commission_pct
    horizon = config.data.target_horizon_days
    actual_returns = []
    
    for _, trade in trades.iterrows():
        future_prices = full_df[(full_df['Ticker'] == trade['Ticker']) & (full_df['Date'] > trade['Date'])].head(horizon)
        if future_prices.empty:
            actual_returns.append(0)
            continue
        
        entry_price = trade['Close']
        final_trade_return = 0
        for _, future_day in future_prices.iterrows():
            current_return = (future_day['Close'] - entry_price) / entry_price
            if current_return <= stop_loss:
                final_trade_return = stop_loss
                break
            final_trade_return = current_return
            
        net_return = final_trade_return - (2 * commission)
        actual_returns.append(net_return)
        
    trades['Net_Return'] = actual_returns
    total_trades = len(trades)
    
    if total_trades > 0:
        win_rate = len(trades[trades['Net_Return'] > 0]) / total_trades
        strategy_return = trades['Net_Return'].sum()
        print(f"\nNet Backtest Results (Stop Loss: {stop_loss*100}%, Commission: {commission*100}% per side):")
        print(f"Total Trades Taken: {total_trades}")
        print(f"Win Rate (Net): {win_rate:.2%}")
        print(f"Strategy Cumulative Net Return: {strategy_return:.2%}")
    else:
        print("No trades generated.")

if __name__ == "__main__":
    run_backtest()