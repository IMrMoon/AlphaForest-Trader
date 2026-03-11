import yaml
from pydantic import BaseModel

class ProjectConfig(BaseModel):
    name: str
    environment: str

class DataConfig(BaseModel):
    universe: str
    target_horizon_days: int
    target_threshold_pct: float
    data_source: str
    tickers: list[str]
    stop_loss_pct: float
    commission_pct: float

class ModelConfig(BaseModel):
    model_type: str
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int

class AppConfig(BaseModel):
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig

def load_config(config_path: str = "config/base.yaml") -> AppConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    return AppConfig(**yaml_data)

if __name__ == "__main__":
    config = load_config()
    print("Config loaded successfully!")
    print(f"Universe: {config.data.universe}")
    print(f"Horizon: {config.data.target_horizon_days} days")
    print(f"Model: {config.model.model_type}")