import pandas as pd
from src.config_loader import load_config
from src.ingestion.data_loader import get_data_source, check_data_quality
from src.features.feature_builder import create_features
from src.labels.target_builder import create_targets

def main():
    config = load_config()
    source = get_data_source(config.data.data_source)
    raw_data = source.fetch_data(config.data.tickers, start_date="2020-01-01", end_date="2024-01-01")
    clean_data = check_data_quality(raw_data)
    
    print("Building features...")
    features_df = create_features(clean_data)
    
    print("Building targets...")
    targets_df = create_targets(clean_data, config.data.target_horizon_days, config.data.target_threshold_pct)
    
    print("Merging dataset...")
    dataset = pd.merge(features_df, targets_df, on=['Date', 'Ticker'], how='inner')
    dataset.to_csv("data/training_data.csv", index=False)
    
    print("Dataset built successfully!")
    print(dataset.head())
    print(f"Total rows: {len(dataset)}")

if __name__ == "__main__":
    main()