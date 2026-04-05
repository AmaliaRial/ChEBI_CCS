from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model.enconders.adduct_encoder import AdductOneHotEncoder
from model.scripts.splitter import split_train_test


class CCSRegressor(nn.Module):

    #en el __init__ definimos las capas
    def __init__(self, input_dim: int, hidden_dims: tuple[int, int, int] = (1024, 256, 64)):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.output = nn.Linear(hidden_dims[2], 1)

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(0.2)

    #aqui unimos las capas
    def forward(self, x: torch.Tensor) -> torch.Tensor: #first stimation of CCS
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.output(x) 
        return x.squeeze(-1)


def get_fingerprint_columns(df: pd.DataFrame) -> list[str]:
    pattern = re.compile(r"^V\d+$")
    fp_cols = [col for col in df.columns if pattern.match(str(col))]
    if not fp_cols:
        raise ValueError("No se encontraron columnas de fingerprints con patrón V1..Vn.")

    fp_cols.sort(key=lambda name: int(name[1:])) #las ordenamos para que sigan tieniendo el mismo orden
    return fp_cols

#contruimos la entrada del modelo (fingerprint, adduct y m/z)
def build_feature_matrix(df: pd.DataFrame, adduct_encoder: AdductOneHotEncoder | None = None, fit_encoder: bool = True,) -> tuple[np.ndarray, AdductOneHotEncoder, list[str]]:
    fp_cols = get_fingerprint_columns(df)
    fp_matrix = df[fp_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

    if "Adduct" not in df.columns:
        raise ValueError("No existe la columna 'Adduct' en el dataset.")
    if "m/z" not in df.columns:
        raise ValueError("No existe la columna 'm/z' en el dataset.")

    if adduct_encoder is None:
        adduct_encoder = AdductOneHotEncoder()

    if fit_encoder:
        adduct_encoder.fit(df["Adduct"].to_numpy())
        adduct_ohe = adduct_encoder.transform(df["Adduct"].to_numpy())
    else:
        adduct_ohe = adduct_encoder.transform(df["Adduct"].to_numpy())

    categories = getattr(adduct_encoder, "categories_", None)
    if categories is None or len(categories) == 0:
        converter = getattr(adduct_encoder, "converter", {})
        categories = [k for k, _ in sorted(converter.items(), key=lambda item: item[1])]

    adduct_cols = [f"adduct__{cat}" for cat in categories]
    adduct_ohe_df = pd.DataFrame(adduct_ohe, columns=adduct_cols, index=df.index)

    mz = pd.to_numeric(df["m/z"], errors="coerce").fillna(0).astype(np.float32)
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
    if "CCS" not in df.columns:
        raise ValueError("No existe la columna 'CCS' en el dataset.")
    return pd.to_numeric(df["CCS"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)


def train_model(data_csv: str, output_dir: str, test_size: float = 0.2, random_state: int = 42, epochs: int = 40, batch_size: int = 64, lr: float = 1e-3,) -> None:
    
    #Se fijan semillas aletorias para garantizar el mismo resultado al entrenar varias veces el modelo
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    #leemos el dataset
    df = pd.read_csv(data_csv, low_memory=False)

    #hacemos el split train/test
    train_df, test_df = split_train_test(df, test_size=test_size, random_state=random_state)

    #TRAIN
    x_train, adduct_encoder, fp_cols = build_feature_matrix(train_df, fit_encoder=True)
    y_train = build_target(train_df)

    #TEST
    #Para el test usamos el mismo encoder que se usó para el train (al ser una variable categórica no queremos que cambie)
    x_test, _, _ = build_feature_matrix(test_df, adduct_encoder=adduct_encoder, fit_encoder=False)
    y_test = build_target(test_df)

    #convertimos a tensores de PyTorch
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)


    #En vez de entrenar todo el dataset lo dividimos en mini-grupos para que sea mas eficiente.
    #Creamos un DataLoader que se encargará de manejar los batches y el shuffle de los datos.
    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    #Aqui definimos la "forma" de la res, pero aun no se han aprendido los pesos.
    model = CCSRegressor(input_dim=x_train.shape[1])

    loss = nn.MSELoss()# medimos el error cuadrático medio entre las predicciones y los valores reales
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)# actualiza los pesos del modelo para minimizar la función de pérdida

    #bucle de entrenamiento, por cada batch de cada epoch
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader: #batch

            #aqui se aprende el modelo, se hace un forward pass, se calcula la pérdida, se hace un backward pass y se actualizan los pesos
            optimizer.zero_grad() #--> forward pass

            pred = model(xb) #el modelo hace una predicción a partir de las características del batch
            loss = loss(pred, yb) #calculamos la pérdida
            loss.backward() #--> backward pass: calcula como cambia cada peso
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_test = model(x_test_t)
        mse = nn.functional.mse_loss(pred_test, y_test_t).item()
        rmse = float(np.sqrt(mse))
        mae = nn.functional.l1_loss(pred_test, y_test_t).item()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)


    metadata = {
        "input_csv": str(data_csv),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "adduct_categories": get_adduct_categories(adduct_encoder),
        "metrics": {"rmse": rmse, "mae": mae},
        "architecture": {
            "layers_total": 5,
            "hidden_layers": 3,
            "activation": "LeakyReLU",
            "output_activation": "linear",
        },
    }
    with (out / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    pred_df = pd.DataFrame(
        {
            "CCS_true": y_test,
            "CCS_pred": pred_test,
        }
    )
    pred_df.to_csv(out / "test_predictions.csv", index=False)

    train_df.to_csv(out / "train_split.csv", index=False)
    test_df.to_csv(out / "test_split.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena modelo base para predecir CCS.")
    parser.add_argument(
        "--input",
        default="data/raw_datasets/fingerprints/AllCCS2_experimental_with_inchis_vectorfingerprintsVectorized.csv",
        help="CSV de entrada con columnas V1..Vn, Adduct, m/z y CCS.",
    )
    parser.add_argument("--output-dir", default="model/artifacts", help="Directorio de salida.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        data_csv=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
