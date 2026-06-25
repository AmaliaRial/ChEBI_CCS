from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg") #para generar gráficos sin necesidad de una interfaz gráfica
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import median_absolute_error, r2_score

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from model.enconders.adduct_encoder import AdductOneHotEncoder

print(torch.cuda.is_available())  # Debe ser True
print(torch.cuda.get_device_name(0))  # Debe mostrar tu GPU

#siempre correr codigo con este formato --> conda run -n tfg_amalia python , xq sino no reconoce la GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CCSRegressor(nn.Module):

    #en el __init__ definimos las capas
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] | list[int] = (1024, 256, 64), dropout_rate: float = 0.2):
        super().__init__()

        layers = []
        in_features = input_dim
        
        for out_features in hidden_dims:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            layers.append(nn.Dropout(dropout_rate))
            in_features = out_features
        
        layers.append(nn.Linear(in_features, 1))
        self.model = nn.Sequential(*layers)

    #aqui unimos las capas
    def forward(self, x: torch.Tensor) -> torch.Tensor: #first stimation of CCS
        return self.model(x).squeeze(-1)


def get_fingerprint_columns(df: pd.DataFrame) -> list[str]:
    pattern = re.compile(r"^V\d+$")
    fp_cols = [col for col in df.columns if pattern.match(str(col))]
    if not fp_cols:
        raise ValueError("No se encontraron columnas de fingerprints con patrón V1..Vn.")

    fp_cols.sort(key=lambda name: int(name[1:])) #las ordenamos para que sigan tieniendo el mismo orden
    return fp_cols


def get_column_name(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No existe la columna {label} en el dataset. Se intentaron: {candidates}")


#contruimos la entrada del modelo (fingerprint, adduct y m/z)
def build_feature_matrix(df: pd.DataFrame, adduct_encoder: AdductOneHotEncoder | None = None, fit_encoder: bool = True,) -> tuple[np.ndarray, AdductOneHotEncoder, list[str]]:
    fp_cols = get_fingerprint_columns(df)
    fp_matrix = df[fp_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

    adduct_col = get_column_name(df, ("Adduct", "adduct"), "'Adduct'/'adduct'")
    mz_col = get_column_name(df, ("m/z", "mz"), "'m/z'/'mz'")

    if adduct_encoder is None:
        adduct_encoder = AdductOneHotEncoder()

    if fit_encoder:
        adduct_encoder.fit(df[adduct_col].to_numpy())
        adduct_ohe = adduct_encoder.transform(df[adduct_col].to_numpy())
    else:
        adduct_ohe = adduct_encoder.transform(df[adduct_col].to_numpy())

    categories = getattr(adduct_encoder, "categories_", None)
    if categories is None or len(categories) == 0:
        converter = getattr(adduct_encoder, "converter", {})
        categories = [k for k, _ in sorted(converter.items(), key=lambda item: item[1])]

    adduct_cols = [f"adduct__{cat}" for cat in categories]
    adduct_ohe_df = pd.DataFrame(adduct_ohe, columns=adduct_cols, index=df.index)

    mz = pd.to_numeric(df[mz_col], errors="coerce").fillna(0).astype(np.float32)
    mz_df = pd.DataFrame({"mz": mz})

    features_df = pd.concat([fp_matrix, adduct_ohe_df.astype(np.float32), mz_df], axis=1)
    return features_df.to_numpy(dtype=np.float32), adduct_encoder, fp_cols


def get_adduct_categories(adduct_encoder: AdductOneHotEncoder) -> list[str]:
    categories = getattr(adduct_encoder, "categories_", None)
    if categories is None or len(categories) == 0:
        converter = getattr(adduct_encoder, "converter", {})
        categories = [k for k, _ in sorted(converter.items(), key=lambda item: item[1])]
    return categories


def build_target(df: pd.DataFrame) -> np.ndarray:
    ccs_col = get_column_name(df, ("CCS", "ccs"), "'CCS'/'ccs'")
    return pd.to_numeric(df[ccs_col], errors="coerce").fillna(0).to_numpy(dtype=np.float32)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    medae = float(median_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": rmse,
        "mae": mae,
        "medae": medae,
        "r2": r2,
    }


def predict_array(model: nn.Module, features: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(features.to(device)).detach().cpu().numpy()


def train_model(train_csv: str, output_dir: str, val_csv: str | None = None, test_csv: str | None = None, test_size: float = 0.2, random_state: int = 42, epochs: int = 40, batch_size: int = 64, lr: float = 1e-3) -> None:
    
    #Se fijan semillas aletorias para garantizar el mismo resultado al entrenar varias veces el modelo
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # leemos el dataset nuevo: train/val/test ya vienen definidos
    train_df = pd.read_csv(train_csv, low_memory=False)
    val_df = pd.read_csv(val_csv, low_memory=False) if val_csv else None
    test_df = pd.read_csv(test_csv, low_memory=False) if test_csv else None

    if val_df is None or test_df is None:
        raise ValueError("Este entrenamiento requiere train, val y test explícitos.")

    #TRAIN
    x_train, adduct_encoder, fp_cols = build_feature_matrix(train_df, fit_encoder=True)
    y_train = build_target(train_df)

    #VALIDATION
    x_val, _, _ = build_feature_matrix(val_df, adduct_encoder=adduct_encoder, fit_encoder=False)
    y_val = build_target(val_df)

    #TEST
    # Para el test usamos el mismo encoder que se usó para el train.
    x_test, _, _ = build_feature_matrix(test_df, adduct_encoder=adduct_encoder, fit_encoder=False)
    y_test = build_target(test_df)

    #convertimos a tensores de PyTorch
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)


    #En vez de entrenar todo el dataset lo dividimos en mini-grupos para que sea mas eficiente.
    #Creamos un DataLoader que se encargará de manejar los batches y el shuffle de los datos.
    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    #Aqui definimos la "forma" de la res, pero aun no se han aprendido los pesos.
    model = CCSRegressor(input_dim=x_train.shape[1])
    model.to(DEVICE)

    loss_fn = nn.MSELoss()# medimos el error cuadrático medio entre las predicciones y los valores reales
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)# actualiza los pesos del modelo para minimizar la función de pérdida

    history: list[dict[str, float]] = []

    #bucle de entrenamiento, por cada batch de cada epoch
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader: #batch
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            #aqui se aprende el modelo, se hace un forward pass, se calcula la pérdida, se hace un backward pass y se actualizan los pesos
            optimizer.zero_grad() #--> forward pass

            pred = model(xb) #el modelo hace una predicción a partir de las características del batch
            batch_loss = loss_fn(pred, yb) #calculamos la pérdida
            batch_loss.backward() #--> backward pass: calcula como cambia cada peso
            optimizer.step()

        train_pred_epoch = predict_array(model, x_train_t, DEVICE)
        val_pred_epoch = predict_array(model, x_val_t, DEVICE)
        train_epoch_metrics = regression_metrics(y_train, train_pred_epoch)
        val_epoch_metrics = regression_metrics(y_val, val_pred_epoch)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean((y_train - train_pred_epoch) ** 2)),
                "val_loss": float(np.mean((y_val - val_pred_epoch) ** 2)),
                "train_rmse": train_epoch_metrics["rmse"],
                "val_rmse": val_epoch_metrics["rmse"],
                "train_mae": train_epoch_metrics["mae"],
                "val_mae": val_epoch_metrics["mae"],
            }
        )

    model.eval()
    pred_train = predict_array(model, x_train_t, DEVICE)
    pred_val = predict_array(model, x_val_t, DEVICE)
    pred_test = predict_array(model, x_test_t, DEVICE)

    train_metrics = regression_metrics(y_train, pred_train)
    val_metrics = regression_metrics(y_val, pred_val)
    test_metrics = regression_metrics(y_test, pred_test)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    curves_path = out / "training_curves.png"
    plot_training_curves(history, curves_path)


    metadata = {
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "random_state": int(random_state),
        "adduct_categories": get_adduct_categories(adduct_encoder),
        "history": history,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "architecture": {
            "layers_total": 5,
            "hidden_layers": 3,
            "activation": "LeakyReLU",
            "output_activation": "linear",
        },
    }
    with (out / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Prepare identifier columns from test_df if available
    pred_data = {
        "CCS_true": y_test,
        "CCS_pred": pred_test,
    }
    
    # Add identifier columns (row_id, name, smiles) if available
    if "row_id" in test_df.columns:
        pred_data["row_id"] = test_df["row_id"].values
    if "name" in test_df.columns:
        pred_data["name"] = test_df["name"].values
    if "smiles" in test_df.columns:
        pred_data["smiles"] = test_df["smiles"].values
    
    # Reorder columns to put identifiers first
    pred_df = pd.DataFrame(pred_data)
    column_order = []
    if "row_id" in pred_df.columns:
        column_order.append("row_id")
    if "name" in pred_df.columns:
        column_order.append("name")
    if "smiles" in pred_df.columns:
        column_order.append("smiles")
    column_order.extend(["CCS_true", "CCS_pred"])
    pred_df = pred_df[column_order]
    
    pred_df.to_csv(out / "test_predictions.csv", index=False)

    train_df.to_csv(out / "train_split.csv", index=False)
    val_df.to_csv(out / "val_split.csv", index=False)
    test_df.to_csv(out / "test_split.csv", index=False)



def plot_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [item["epoch"] for item in history]
    train_rmse = [item["train_rmse"] for item in history]
    val_rmse = [item["val_rmse"] for item in history]
    train_mae = [item["train_mae"] for item in history]
    val_mae = [item["val_mae"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)

    axes[0].plot(epochs, train_rmse, label="Train RMSE", linewidth=2)
    axes[0].plot(epochs, val_rmse, label="Val RMSE", linewidth=2)
    axes[0].set_title("RMSE per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_mae, label="Train MAE", linewidth=2)
    axes[1].plot(epochs, val_mae, label="Val MAE", linewidth=2)
    axes[1].set_title("MAE per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


#OPTIMIZACION DE PARAMETROS CON OPTUNA
def optimize_hyperparameters(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame | None = None, n_trials: int = 100, db_path: str = "hyperopt/baseline_optuna.db", batch_size: int = 128, random_state: int = 42) -> dict:

    import optuna
    from optuna.storages import RDBStorage
    from optuna.pruners import MedianPruner
    
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # Build feature matrices
    x_train, adduct_encoder, _ = build_feature_matrix(train_df, fit_encoder=True)
    y_train = build_target(train_df)
    
    x_val, _, _ = build_feature_matrix(val_df, adduct_encoder=adduct_encoder, fit_encoder=False)
    y_val = build_target(val_df)
    
    # Convert to tensors
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    
    print(f"Train shape: {x_train.shape}, Val shape: {x_val.shape}")
    
    def objective(trial):
        #we dirst set the search space (define our parameters)
        n_layers = trial.suggest_int("n_layers", 1, 5)
        params = {
            "n_layers": n_layers,
            "hidden_dims": [trial.suggest_int(f"n_units_l{i}", 32, 512, step=32) for i in range(n_layers)],
            "learning_rate": trial.suggest_float("learning_rate", 0.00001, 0.1, log=True),
            "epochs": trial.suggest_int("epochs", 10, 100),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5)
        }
        
        # Define our model
        model = CCSRegressor(x_train.shape[1], params["hidden_dims"], dropout_rate=params["dropout"])
        model.to(DEVICE) #xq sino corre en la CPU y como que no
        loss = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        
        # Prepare data loader
        train_loader = DataLoader(
            TensorDataset(x_train_t, y_train_t),
            batch_size=batch_size,
            shuffle=True
        )
        
        # Train
        for epoch in range(params["epochs"]):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss(pred, yb)
                loss.backward()
                optimizer.step()
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            y_pred_val = predict_array(model, x_val_t, DEVICE)
        
    
        mae_val = float(np.mean(np.abs(y_val - y_pred_val)))
        return mae_val
    
   
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    storage = RDBStorage(f"sqlite:///{db_path}")


    study = optuna.create_study(direction="minimize", storage=storage, study_name="baseline_ccs_optimization", load_if_exists=True, pruner=MedianPruner())
      
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_trial = study.best_trial
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"\nBest trial: {best_trial.number}")
    print(f"Best MAE (validation): {best_trial.value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    output_dir = db_path.parent
    with open(output_dir / "best_params_final.txt", "w") as f:
        f.write(f"Best Trial: {best_trial.number}\n")
        f.write(f"Best MAE (validation): {best_trial.value:.4f}\n\n")
        f.write("Hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"\nResults saved to {db_path}")
    print(f"Best params saved to {output_dir / 'best_params_final.txt'}")
    
    return {
        "best_trial_number": best_trial.number,
        "best_mae_val": best_trial.value,
        "best_params": best_trial.params,
        "best_Params": study.best_params,
        "db_path": str(db_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena modelo base para predecir CCS.")
    parser.add_argument("--train-input", default="data/model/train_ccs_fingerprints.csv", help="CSV de entrenamiento con columnas V1..Vn, adduct, mz y ccs." )
    parser.add_argument("--val-input", default="data/model/val_ccs_fingerprints.csv")
    parser.add_argument("--test-input", default="data/model/test_ccs_fingerprints.csv")
    parser.add_argument("--output-dir", default="predictions/base", help="Directorio de salida.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        train_csv=args.train_input,
        output_dir=args.output_dir,
        val_csv=args.val_input,
        test_csv=args.test_input,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
