"""
Uses best hyperparameters from Optuna baseline optimization.
Tests lambdas from 0.1 to 2.0 in steps of 0.1 (20 values).
Compares only on VALIDATION set.
"""

from pathlib import Path
import json
import sys

import pandas as pd
import numpy as np

repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from model.chebi_model import train_model as train_multitask_model


def read_best_params(db_path="hyperopt/baseline_optuna.db"):
    """Read best hyperparameters from Optuna optimization."""
    best_params_file = Path(db_path).parent / "best_params.txt"
    
    if not best_params_file.exists():
        raise FileNotFoundError(f"Best params file not found: {best_params_file}")
    
    params = {}
    with open(best_params_file, "r") as f:
        lines = f.readlines()
    
    # Parse the file
    for line in lines:
        line = line.strip()
        if not line or ":" not in line:
            continue
        
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        
        # Skip non-parameter lines
        if key in ["Best Trial", "Best MAE (validation)", "Hyperparameters"]:
            continue
        
        # Try to parse value as int or float
        try:
            if key.startswith("n_"):
                params[key] = int(value)
            else:
                params[key] = float(value)
        except ValueError:
            params[key] = value
    
    return params


def build_hidden_dims(best_params):
    hidden_dims = []
    n_layers = best_params.get("n_layers", 3)
    
    for i in range(n_layers):
        key = f"n_units_l{i}"
        if key in best_params:
            hidden_dims.append(best_params[key])
    
    
    if not hidden_dims:
        hidden_dims = [1024, 256, 64]
    
    return tuple(hidden_dims)


def optimize_lambda():
    #test lambdas from 0.1 to 2.0 in steps of 0.1.
    
    # Read best hyperparameters from Optuna
    try:
        best_params = read_best_params()
        print(f"Best params loaded: {best_params}")
    except Exception as e:
        print(f"Error reading best params: {e}")
        print("Falling back to default architecture: (1024, 256, 64)")
        best_params = {
            "n_layers": 3,
            "n_units_l0": 1024,
            "n_units_l1": 256,
            "n_units_l2": 64,
            "learning_rate": 0.001,
            "epochs": 30,
            "dropout": 0.2,
        }
    
    9
    hidden_dims = build_hidden_dims(best_params)
    lr = best_params.get("learning_rate", 0.001)
    epochs = best_params.get("epochs", 30)
    dropout = best_params.get("dropout", 0.2)
    
    print(f"\nArchitecture: {hidden_dims}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Dropout: {dropout}")
    
    lambdas = np.arange(0.1, 2.1, 0.1)
    lambdas = np.round(lambdas, 1)  #redondeamos a un decimal para evitar problemas de precisión
    
    results = []
    
    print(f"\n{'='*80}")
    print(f"LAMBDA OPTIMIZATION (Validation MAE)")
    print(f"{'='*80}\n")
    
    for i, lam in enumerate(lambdas, 1):
        print(f"[{i}/20] Testing lambda={lam}...", end=" ", flush=True)
        
        output_dir = Path(f"predictions/lambda_sweep/lambda_{lam}")
        
        try:
            # Train multitask model
            summary = train_multitask_model(
                train_csv="predictions/ontology_model_filtered/train_split.csv",
                val_csv="predictions/ontology_model_filtered/val_split.csv",
                test_csv="predictions/ontology_model_filtered/test_split.csv",
                ontology_input="predictions/ontology_model_filtered/ontology_labels.csv",
                hidden_dims=hidden_dims,
                dropout_rate=dropout,
                epochs=epochs,
                batch_size=128,
                lr=lr,
                lambda_ontology=lam,
                output_dir=str(output_dir),
            )
            
            # Extract validation MAE
            val_metrics = summary.get("metrics", {}).get("val", {})
            val_mae = val_metrics.get("mae", np.inf)
            
            results.append({
                "lambda": lam,
                "val_mae": val_mae,
                "val_r2": val_metrics.get("r2"),
                "val_rmse": val_metrics.get("rmse"),
                "output_dir": str(output_dir),
            })
            
            print(f"VAL MAE = {val_mae:.4f}")
            
        except Exception as e:
            print(f"Error: {str(e)[:50]}")
            results.append({
                "lambda": lam,
                "val_mae": np.inf,
                "error": str(e),
            })
    
    # best lambda
    results_df = pd.DataFrame(results)
    best_idx = results_df["val_mae"].idxmin()
    best_lambda = results_df.loc[best_idx, "lambda"]
    best_mae = results_df.loc[best_idx, "val_mae"]
    



    print(f"\n{'='*80}")
    print(f"LAMBDA OPTIMIZATION RESULTS")
    print(f"{'='*80}\n")
    
    print(results_df.to_string(index=False))
    
    print(f"\n{'─'*80}")
    print(f"BEST LAMBDA: {best_lambda:.1f}")
    print(f"VALIDATION MAE: {best_mae:.4f}")
    print(f"{'─'*80}\n")
    
    # Save results
    output_dir = Path("predictions/lambda_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_dir / "lambda_sweep_results.csv", index=False)
    
    with open(output_dir / "best_lambda.txt", "w") as f:
        f.write(f"Best Lambda: {best_lambda:.1f}\n")
        f.write(f"Validation MAE: {best_mae:.4f}\n")
        f.write(f"\nArchitecture: {hidden_dims}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Dropout: {dropout}\n")
    
    print(f"Results saved to {output_dir}/lambda_sweep_results.csv")
    print(f"Best lambda info saved to {output_dir}/best_lambda.txt")
    
    return best_lambda, best_mae


if __name__ == "__main__":
    optimize_lambda()
