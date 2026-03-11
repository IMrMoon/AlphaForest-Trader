import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config_loader import load_config

def train_and_evaluate():
    config = load_config()
    print("Loading dataset for training...")
    df = pd.read_csv("data/training_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    features = ['Return_1d', 'SMA_10', 'SMA_20', 'Momentum_10d', 'RSI_14', 'MACD', 'BB_width', 'Market_Return_1d', 'Daily_Volatility', 'Volume_Surge']
    target = 'Target'
    
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size + val_size:].copy()
    
    X_train = pd.concat([train_df[features], df.iloc[train_size:train_size + val_size][features]])
    y_train = pd.concat([train_df[target], df.iloc[train_size:train_size + val_size][target]])
    X_test, y_test = test_df[features], test_df[target]
    
    mlflow.set_experiment("trading_ai_experiment")
    with mlflow.start_run():
        mlflow.log_param("model_type", config.model.model_type)
        mlflow.log_param("n_estimators", config.model.n_estimators)
        mlflow.log_param("target_threshold", config.data.target_threshold_pct)
        
        print("Training Random Forest with MLflow tracking...")
        model = RandomForestClassifier(n_estimators=config.model.n_estimators, max_depth=config.model.max_depth, class_weight='balanced', random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        joblib.dump(model, "data/model.pkl")
        
        probs = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, probs)
        mlflow.log_metric("roc_auc", auc_score)
        print(f"\nROC-AUC Score: {auc_score:.4f}")
        
        test_df['Predict_Proba'] = probs
        test_df.to_csv("data/test_predictions.csv", index=False)
        print("Predictions saved successfully.")

if __name__ == "__main__":
    train_and_evaluate()