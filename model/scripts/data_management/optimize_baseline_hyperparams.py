from pathlib import Path
import sys

import pandas as pd

repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from model.base_model import optimize_hyperparameters


if __name__ == "__main__":
    train_input = "predictions/ontology_model_filtered/train_split.csv"
    val_input = "predictions/ontology_model_filtered/val_split.csv"
    test_input = "predictions/ontology_model_filtered/test_split.csv"
    
    train_df = pd.read_csv(train_input, low_memory=False)
    val_df = pd.read_csv(val_input, low_memory=False)
    test_df = pd.read_csv(test_input, low_memory=False)
    
    print(f"Data loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    result = optimize_hyperparameters(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        n_trials=100,
        db_path="hyperopt/baseline_optuna3.db",
        batch_size=128,
        random_state=42
    )
