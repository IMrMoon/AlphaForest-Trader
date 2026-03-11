import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def objective(trial):
    df = pd.read_csv("data/training_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    features = ['Return_1d', 'SMA_10', 'SMA_20', 'Momentum_10d', 'RSI_14', 'MACD', 'BB_width', 'Market_Return_1d', 'Daily_Volatility', 'Volume_Surge']
    target = 'Target'
    
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    
    n_estimators = trial.suggest_int('n_estimators', 50, 300, step=50)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    
    return roc_auc_score(y_val, probs)

def run_optimization():
    print("Starting Optuna Hyperparameter Tuning...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    print("\n" + "="*50)
    print("🏆 Best Parameters Found 🏆")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")
    print(f"Best ROC-AUC: {study.best_value:.4f}")
    print("="*50)
    print("Please update these values in your config/base.yaml file.")

if __name__ == "__main__":
    run_optimization()