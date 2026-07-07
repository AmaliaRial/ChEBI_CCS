from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    median_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    average_precision_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from model.enconders.adduct_encoder import AdductOneHotEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ONTOLOGY_PREFIXES = ("ont_", "ontology__")


class CCSRegressor(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (1024, 256, 64), num_ontology_labels: int = 1, dropout_rate: float = 0.2):
        super().__init__()
        #Usamos Sequence[int] para permitir cualquier secuencia de enteros (lista, tupple, etc.) como hidden_dims
        #Lista: [1024, 256, 64], Tupple: (1024, 256, 64), etc. De 3 capas

        """ antes de la optimization, el modelo tenía una arquitectura fija de 3 capas ocultas con dimensiones 1024, 256 y 64
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(0.2)

        self.ccs_head = nn.Linear(hidden_dims[2], 1)
        self.ontology_head = nn.Linear(hidden_dims[2], num_ontology_labels)"""

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims debe contener al menos una capa.")

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(dropout_rate)

        # tronco compartido (para las 2 tareas) de N capas
        self.hidden_layers = nn.ModuleList() #crea una lista especial vacía de PyTorch para guardar capas neuronales
        in_features = input_dim #la primera capa recibe como entrada todas las variables del dataset (es decir, si hay 2058 columnas, entonces in_features = 2058)
        for out_features in hidden_dims: #este for equivale a las self.fc1, self.fc2, self.fc3 de antes, pero ahora es dinámico según hidden_dims
            self.hidden_layers.append(nn.Linear(in_features, out_features)) #por cada tamaño de hidden_dims, creo una capa linear
            in_features = out_features #después actualizo in_features para que la siguiente capa reciba la salida de la anterior

        embedding_dim = hidden_dims[-1] #embedding, ultimo elemento de las capas ocultas
        self.ccs_head = nn.Linear(embedding_dim, 1) #para predecir CCS recibe el embedding y devuelve 1 numero
        self.ontology_head = nn.Linear(embedding_dim, num_ontology_labels) #igual que arriba pero en vez de predecir 1 numero, predice tantos como etiquetas ontológicas haya

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # dropout tras cada capa oculta salvo la última (la capa de embedding).
        # x es una matriz del nº de moleculas x nº de features (fingerprints + adduct + m/z)
        last = len(self.hidden_layers) - 1
        for index, layer in enumerate(self.hidden_layers):
            x = self.activation(layer(x))
            if index < last:
                x = self.dropout(x)
        
        embedding = x #x ahora es la representacion final de la molecula 

        ccs_pred = self.ccs_head(embedding).squeeze(-1) #usamos squeeze(-1) para eliminar la dimensión de salida de tamaño 1, dejando un vector de predicciones CCS seguidas y no separadas en vectores chiquititos de segunda dimension 1
        ontology_logits = self.ontology_head(embedding)
        return ccs_pred, ontology_logits, embedding


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


def get_ontology_label_columns(df: pd.DataFrame) -> list[str]:
    label_columns = [
        column
        for column in df.columns
        if any(str(column).startswith(prefix) for prefix in ONTOLOGY_PREFIXES)
    ]
    label_columns.sort()
    return label_columns


def read_ontology_dataset(ontology_csv: str, row_id_column: str) -> pd.DataFrame:
    header = pd.read_csv(ontology_csv, nrows=0)
    if row_id_column not in header.columns:
        raise ValueError(f"El dataset ontologico no contiene la columna {row_id_column}.")

    label_columns = get_ontology_label_columns(header)
    if not label_columns:
        raise ValueError("El dataset ontologico no contiene columnas ont_ ni ontology__.")

    label_dtypes = {column: np.float32 for column in label_columns}
    return pd.read_csv(
        ontology_csv,
        usecols=[row_id_column, *label_columns],
        dtype=label_dtypes,
        low_memory=False,
    )


def build_ontology_targets(df: pd.DataFrame, label_columns: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    if label_columns is None:
        label_columns = get_ontology_label_columns(df)
    if not label_columns:
        raise ValueError("No se encontraron columnas ontológicas con los prefijos ont_ o ontology__.")

    ontology_matrix = (
        df[label_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(np.float32)
        .to_numpy(dtype=np.float32)
    )
    return ontology_matrix, label_columns


def merge_ontology_labels(split_df: pd.DataFrame, ontology_df: pd.DataFrame | None, row_id_column: str = "row_id") -> pd.DataFrame:
    if get_ontology_label_columns(split_df):
        return split_df.copy()

    if ontology_df is None:
        raise ValueError(
            "Se necesita un dataset ontológico separado o columnas ont_ / ontology__ ya incluidas en los splits."
        )

    if row_id_column not in split_df.columns:
        raise ValueError(f"El split no contiene la columna de unión {row_id_column}.")
    if row_id_column not in ontology_df.columns:
        raise ValueError(f"El dataset ontológico no contiene la columna {row_id_column}.")

    label_columns = get_ontology_label_columns(ontology_df)
    if not label_columns:
        raise ValueError("El dataset ontológico no contiene columnas ont_ ni ontology__.")

    ontology_subset = ontology_df[[row_id_column] + label_columns].drop_duplicates(subset=[row_id_column])
    merged_df = split_df.merge(ontology_subset, on=row_id_column, how="left", validate="one_to_one")
    if merged_df[label_columns].isna().any().any():
        missing_ids = merged_df.loc[merged_df[label_columns].isna().any(axis=1), row_id_column].tolist()
        raise ValueError(f"Faltan etiquetas ontológicas para {len(missing_ids)} filas del split.")
    return merged_df


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


def predict_outputs(model: nn.Module, features: torch.Tensor, batch_size: int = 2048) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ccs_batches: list[np.ndarray] = []
    ontology_batches: list[np.ndarray] = []
    embedding_batches: list[np.ndarray] = []
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            feature_batch = features[start : start + batch_size].to(model_device)
            ccs_pred, ontology_logits, embedding = model(feature_batch)
            ccs_batches.append(ccs_pred.detach().cpu().numpy())
            ontology_batches.append(ontology_logits.detach().cpu().numpy())
            embedding_batches.append(embedding.detach().cpu().numpy())

    return (
        np.concatenate(ccs_batches, axis=0),
        np.concatenate(ontology_batches, axis=0),
        np.concatenate(embedding_batches, axis=0),
    )


def predict_array(model: nn.Module, features: torch.Tensor, batch_size: int = 2048) -> np.ndarray:
    ccs_pred, _, _ = predict_outputs(model, features, batch_size=batch_size)
    return ccs_pred


def ontology_metrics(y_true: np.ndarray, logits: np.ndarray, threshold: float = 0.5) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=np.float32)
    logits = np.asarray(logits, dtype=np.float32)
    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
    y_pred = (probabilities >= threshold).astype(np.int32)

    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics: dict[str, float | None] = {
        "threshold": float(threshold),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "average_precision_micro": None,
        "average_precision_macro": None,
        "roc_auc_micro": None,
        "roc_auc_macro": None,
    }

    def safe_score(score_function) -> float | None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value = float(score_function())
        except ValueError:
            return None
        return value if np.isfinite(value) else None

    metrics["average_precision_micro"] = safe_score(
        lambda: average_precision_score(y_true, probabilities, average="micro")
    )
    metrics["average_precision_macro"] = safe_score(
        lambda: average_precision_score(y_true, probabilities, average="macro")
    )
    metrics["roc_auc_micro"] = safe_score(
        lambda: roc_auc_score(y_true, probabilities, average="micro")
    )
    metrics["roc_auc_macro"] = safe_score(
        lambda: roc_auc_score(y_true, probabilities, average="macro")
    )

    return metrics


def readable_ontology_label(label: str) -> str:
    readable = str(label)
    for prefix in ("ontology_true__", "ontology_logit__", "ontology_prob__"):
        if readable.startswith(prefix):
            readable = readable[len(prefix) :]
            break
    for prefix in ONTOLOGY_PREFIXES:
        if readable.startswith(prefix):
            readable = readable[len(prefix) :]
            break
    return readable


def build_readable_predictions(row_ids: np.ndarray, y_true_ccs: np.ndarray, y_pred_ccs: np.ndarray, y_true_ontology: np.ndarray, ontology_probabilities: np.ndarray, ontology_label_columns: list[str], threshold: float, max_predicted_classes: int | None = None) -> pd.DataFrame:
    readable_labels = [readable_ontology_label(label) for label in ontology_label_columns]
    true_classes: list[str] = []
    predicted_classes: list[str] = []
    predicted_with_probabilities: list[str] = []

    for true_row, probability_row in zip(y_true_ontology, ontology_probabilities):
        true_indices = np.flatnonzero(true_row == 1)
        predicted_indices = np.flatnonzero(probability_row >= threshold)
        predicted_indices = predicted_indices[np.argsort(probability_row[predicted_indices])[::-1]]
        if max_predicted_classes is not None:
            predicted_indices = predicted_indices[:max_predicted_classes]

        true_classes.append(";".join(readable_labels[index] for index in true_indices))
        predicted_classes.append(";".join(readable_labels[index] for index in predicted_indices))
        predicted_with_probabilities.append(
            ";".join(
                f"{readable_labels[index]} ({probability_row[index]:.6f})"
                for index in predicted_indices
            )
        )

    return pd.DataFrame(
        {
            "row_id": row_ids,
            "CCS_true": y_true_ccs,
            "CCS_pred": y_pred_ccs,
            "CCS_error_abs": np.abs(y_true_ccs - y_pred_ccs),
            "true_classes": true_classes,
            "predicted_classes": predicted_classes,
            "predicted_classes_with_probabilities": predicted_with_probabilities,
        }
    )


def plot_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [item["epoch"] for item in history]
    train_total_loss = [item["train_total_loss"] for item in history]
    val_total_loss = [item["val_total_loss"] for item in history]
    train_rmse = [item["train_rmse"] for item in history]
    val_rmse = [item["val_rmse"] for item in history]
    train_ontology_f1 = [item["train_ontology_f1_micro"] for item in history]
    val_ontology_f1 = [item["val_ontology_f1_micro"] for item in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=160)

    axes[0].plot(epochs, train_total_loss, label="Train total loss", linewidth=2)
    axes[0].plot(epochs, val_total_loss, label="Val total loss", linewidth=2)
    axes[0].set_title("Total loss per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_rmse, label="Train RMSE", linewidth=2)
    axes[1].plot(epochs, val_rmse, label="Val RMSE", linewidth=2)
    axes[1].set_title("CCS RMSE per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    axes[2].plot(epochs, train_ontology_f1, label="Train ontology F1", linewidth=2)
    axes[2].plot(epochs, val_ontology_f1, label="Val ontology F1", linewidth=2)
    axes[2].set_title("Ontology micro F1 per epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def scalar_value(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def train_model(train_csv: str, output_dir: str, val_csv: str | None = None, test_csv: str | None = None, random_state: int = 42, epochs: int = 40, batch_size: int = 64, lr: float = 1e-3, lambda_ontology: float = 0.1, ontology_threshold: float = 0.5, ontology_csv: str | None = None, row_id_column: str = "row_id", device: str = "auto", hidden_dims: Sequence[int] | None = None, dropout_rate: float = 0.2) -> None:

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    resolved_device = resolve_device(device)
    hidden_dims = tuple(int(dim) for dim in hidden_dims) if hidden_dims else (1024, 256, 64)

    train_df = pd.read_csv(train_csv, low_memory=False)
    val_df = pd.read_csv(val_csv, low_memory=False) if val_csv else None
    test_df = pd.read_csv(test_csv, low_memory=False) if test_csv else None
    ontology_df = read_ontology_dataset(ontology_csv, row_id_column) if ontology_csv else None

    if val_df is None or test_df is None:
        raise ValueError("Este entrenamiento requiere train, val y test explícitos.")

    train_df = merge_ontology_labels(train_df, ontology_df, row_id_column=row_id_column)
    val_df = merge_ontology_labels(val_df, ontology_df, row_id_column=row_id_column)
    test_df = merge_ontology_labels(test_df, ontology_df, row_id_column=row_id_column)

    x_train, adduct_encoder, _ = build_feature_matrix(train_df, fit_encoder=True)
    y_train_ccs = build_target(train_df)
    y_train_ontology, ontology_label_columns = build_ontology_targets(train_df)

    x_val, _, _ = build_feature_matrix(val_df, adduct_encoder=adduct_encoder, fit_encoder=False)
    y_val_ccs = build_target(val_df)
    y_val_ontology, _ = build_ontology_targets(val_df, ontology_label_columns)

    x_test, _, _ = build_feature_matrix(test_df, adduct_encoder=adduct_encoder, fit_encoder=False)
    y_test_ccs = build_target(test_df)
    y_test_ontology, _ = build_ontology_targets(test_df, ontology_label_columns)

    print(f"Ontology label columns detected: {len(ontology_label_columns)}")
    print(f"First 10 ontology label columns: {ontology_label_columns[:10]}")
    print(f"Input feature dimension: {x_train.shape[1]}")
    print(
        "Train/val/test rows: "
        f"{len(train_df)} / {len(val_df)} / {len(test_df)}"
    )
    print(f"Training device: {resolved_device}")

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_ccs_t = torch.tensor(y_train_ccs, dtype=torch.float32)
    y_train_ontology_t = torch.tensor(y_train_ontology, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_ccs_t, y_train_ontology_t),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=resolved_device.type == "cuda",
    )

    model = CCSRegressor(input_dim=x_train.shape[1], hidden_dims=hidden_dims, num_ontology_labels=len(ontology_label_columns), dropout_rate=dropout_rate)
    model = model.to(resolved_device)

    ccs_loss_fn = nn.MSELoss()
    ontology_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_total_loss = 0.0
        n_samples = 0

        for xb, yb_ccs, yb_ontology in train_loader:
            xb = xb.to(resolved_device, non_blocking=True)
            yb_ccs = yb_ccs.to(resolved_device, non_blocking=True)
            yb_ontology = yb_ontology.to(resolved_device, non_blocking=True)
            optimizer.zero_grad()
            ccs_pred, ontology_logits, _ = model(xb)
            ccs_loss = ccs_loss_fn(ccs_pred, yb_ccs)
            ontology_loss = ontology_loss_fn(ontology_logits, yb_ontology)
            total_loss = ccs_loss + lambda_ontology * ontology_loss
            total_loss.backward()
            optimizer.step()

            batch_size_actual = xb.size(0)
            running_total_loss += float(total_loss.item()) * batch_size_actual
            n_samples += batch_size_actual

        train_ccs_pred_epoch, train_ontology_logits_epoch, _ = predict_outputs(
            model,
            x_train_t,
            batch_size=batch_size,
        )
        val_ccs_pred_epoch, val_ontology_logits_epoch, _ = predict_outputs(
            model,
            x_val_t,
            batch_size=batch_size,
        )

        train_ccs_metrics = regression_metrics(y_train_ccs, train_ccs_pred_epoch)
        val_ccs_metrics = regression_metrics(y_val_ccs, val_ccs_pred_epoch)
        train_ontology_metrics = ontology_metrics(y_train_ontology, train_ontology_logits_epoch, threshold=ontology_threshold)
        val_ontology_metrics = ontology_metrics(y_val_ontology, val_ontology_logits_epoch, threshold=ontology_threshold)
        train_ccs_loss_epoch = float(np.mean((y_train_ccs - train_ccs_pred_epoch) ** 2))
        val_ccs_loss_epoch = float(np.mean((y_val_ccs - val_ccs_pred_epoch) ** 2))

        train_ontology_logits_tensor = torch.as_tensor(
            np.asarray(train_ontology_logits_epoch, dtype=np.float32).copy(),
            dtype=torch.float32,
            device=DEVICE,
        )

        y_train_ontology_tensor = torch.as_tensor(
            np.asarray(y_train_ontology, dtype=np.float32).copy(),
            dtype=torch.float32,
            device=DEVICE,
        )

        train_ontology_loss_epoch = ontology_loss_fn(
            train_ontology_logits_tensor,
            y_train_ontology_tensor,
        )
        
        val_ontology_logits_tensor = torch.as_tensor(
            np.asarray(val_ontology_logits_epoch, dtype=np.float32).copy(),
            dtype=torch.float32,
            device=DEVICE,
        )

        y_val_ontology_tensor = torch.as_tensor(
            np.asarray(y_val_ontology, dtype=np.float32).copy(),
            dtype=torch.float32,
            device=DEVICE,
        )

        val_ontology_loss_epoch = ontology_loss_fn(
            val_ontology_logits_tensor,
            y_val_ontology_tensor,
        )


    

        history.append(
            {
                "epoch": float(epoch),

                "train_total_loss": scalar_value(running_total_loss / n_samples),
                "val_total_loss": scalar_value(val_ccs_loss_epoch + lambda_ontology * val_ontology_loss_epoch),

                "train_ccs_loss": scalar_value(train_ccs_loss_epoch),
                "val_ccs_loss": scalar_value(val_ccs_loss_epoch),

                "train_ontology_loss": scalar_value(train_ontology_loss_epoch),
                "val_ontology_loss": scalar_value(val_ontology_loss_epoch),

                "train_rmse": float(train_ccs_metrics["rmse"]),
                "val_rmse": float(val_ccs_metrics["rmse"]),

                "train_mae": float(train_ccs_metrics["mae"]),
                "val_mae": float(val_ccs_metrics["mae"]),

                "train_ontology_f1_micro": float(train_ontology_metrics["micro_f1"]),
                "val_ontology_f1_micro": float(val_ontology_metrics["micro_f1"]),
            }
        )

    pred_train, pred_train_logits, _ = predict_outputs(model, x_train_t, batch_size=batch_size)
    pred_val, pred_val_logits, _ = predict_outputs(model, x_val_t, batch_size=batch_size)
    pred_test, pred_test_logits, test_embeddings = predict_outputs(
        model,
        x_test_t,
        batch_size=batch_size,
    )

    train_metrics = regression_metrics(y_train_ccs, pred_train)
    val_metrics = regression_metrics(y_val_ccs, pred_val)
    test_metrics = regression_metrics(y_test_ccs, pred_test)

    train_ontology_metrics = ontology_metrics(y_train_ontology, pred_train_logits, threshold=ontology_threshold)
    val_ontology_metrics = ontology_metrics(y_val_ontology, pred_val_logits, threshold=ontology_threshold)
    test_ontology_metrics = ontology_metrics(y_test_ontology, pred_test_logits, threshold=ontology_threshold)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    curves_path = out / "training_curves.png"
    plot_training_curves(history, curves_path)

    metadata = {
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "ontology_csv": str(ontology_csv) if ontology_csv else None,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "random_state": int(random_state),
        "device": str(resolved_device),
        "lambda_ontology": float(lambda_ontology),
        "ontology_threshold": float(ontology_threshold),
        "adduct_categories": get_adduct_categories(adduct_encoder),
        "ontology_label_columns": ontology_label_columns,
        "history": history,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "ontology_train": train_ontology_metrics,
            "ontology_val": val_ontology_metrics,
            "ontology_test": test_ontology_metrics,
        },
        "architecture": {
            "input": "fingerprints + adduct one-hot + m/z",
            "shared_layers": list(hidden_dims),
            "dropout": float(dropout_rate),
            "heads": ["ccs_head", "ontology_head"],
            "activation": "LeakyReLU",
            "output_activation": "linear",
        },
    }
    with (out / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with (out / "ontology_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": float(ontology_threshold),
                "label_columns": ontology_label_columns,
                "train": train_ontology_metrics,
                "val": val_ontology_metrics,
                "test": test_ontology_metrics,
            },
            f,
            indent=2,
        )

    test_probabilities = torch.sigmoid(torch.from_numpy(pred_test_logits)).numpy()
    row_ids = test_df[row_id_column] if row_id_column in test_df.columns else pd.Series(np.arange(len(test_df)))

    # Prepare base prediction data with identifiers
    base_pred_data: dict[str, np.ndarray] = {}
    
    # Add identifier columns if available
    if "row_id" in test_df.columns:
        base_pred_data["row_id"] = test_df["row_id"].values
    if "name" in test_df.columns:
        base_pred_data["name"] = test_df["name"].values
    if "smiles" in test_df.columns:
        base_pred_data["smiles"] = test_df["smiles"].values
    
    base_pred_data["CCS_true"] = y_test_ccs
    base_pred_data["CCS_pred"] = pred_test

    test_pred_data_full: dict[str, np.ndarray] = base_pred_data.copy()
    test_pred_data_clean: dict[str, np.ndarray] = base_pred_data.copy()
    
    for index, label in enumerate(ontology_label_columns):
        test_pred_data_full[f"ontology_true__{label}"] = y_test_ontology[:, index]
        test_pred_data_full[f"ontology_logit__{label}"] = pred_test_logits[:, index]
        test_pred_data_full[f"ontology_prob__{label}"] = test_probabilities[:, index]
        test_pred_data_clean[f"ontology_true__{label}"] = y_test_ontology[:, index]
        test_pred_data_clean[f"ontology_prob__{label}"] = test_probabilities[:, index]

    pred_df_full = pd.DataFrame(test_pred_data_full)
    pred_df_full.to_csv(out / "test_predictions_full.csv", index=False)

    pred_df_clean = pd.DataFrame(test_pred_data_clean)
    pred_df_clean.to_csv(out / "test_predictions_clean.csv", index=False)

    pred_df_full.to_csv(out / "test_predictions.csv", index=False)

    readable_df = build_readable_predictions(
        row_ids=row_ids.to_numpy(),
        y_true_ccs=y_test_ccs,
        y_pred_ccs=pred_test,
        y_true_ontology=y_test_ontology,
        ontology_probabilities=test_probabilities,
        ontology_label_columns=ontology_label_columns,
        threshold=ontology_threshold,
    )
    readable_df.to_csv(out / "test_predictions_readable.csv", index=False)

    readable_top10_df = build_readable_predictions(
        row_ids=row_ids.to_numpy(),
        y_true_ccs=y_test_ccs,
        y_pred_ccs=pred_test,
        y_true_ontology=y_test_ontology,
        ontology_probabilities=test_probabilities,
        ontology_label_columns=ontology_label_columns,
        threshold=ontology_threshold,
        max_predicted_classes=10,
    )
    readable_top10_df.to_csv(out / "test_predictions_readable_top10.csv", index=False)

    embedding_columns = {f"embedding_{index}": test_embeddings[:, index] for index in range(test_embeddings.shape[1])}
    embedding_df = pd.DataFrame(
        {
            "row_id": row_ids.to_numpy(),
            "CCS_true": y_test_ccs,
            "CCS_pred": pred_test,
            **embedding_columns,
        }
    )
    embedding_df.to_csv(out / "embeddings_test.csv", index=False)

    train_df.to_csv(out / "train_split.csv", index=False)
    val_df.to_csv(out / "val_split.csv", index=False)
    test_df.to_csv(out / "test_split.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el modelo multitarea CCS + ChEBI.")
    parser.add_argument(
        "--train-input",
        default="data/model/train_ccs_fingerprints.csv",
        help="CSV de entrenamiento con columnas V1..Vn, adduct, mz y ccs.",
    )
    parser.add_argument("--val-input", default="data/model/val_ccs_fingerprints.csv")
    parser.add_argument("--test-input", default="data/model/test_ccs_fingerprints.csv")
    parser.add_argument(
        "--ontology-input",
        default="data/model/final_covered_ccs_fingerprints_multilabel_filtered.csv",
        help="CSV con row_id y columnas ont_ o ontology__ generadas por el preprocesado de ChEBI.",
    )
    parser.add_argument("--output-dir", default="predictions/ontology_model", help="Directorio de salida.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-ontology", type=float, default=0.1)
    parser.add_argument("--ontology-threshold", type=float, default=0.5)
    parser.add_argument("--row-id-column", default="row_id")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Dispositivo para el entrenamiento multitarea.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=None, help="Neuronas por capa oculta del tronco compartido (por defecto 1024 256 64).")
    parser.add_argument("--dropout", type=float, default=0.2, help="Tasa de dropout del tronco compartido.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        train_csv=args.train_input,
        output_dir=args.output_dir,
        val_csv=args.val_input,
        test_csv=args.test_input,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_ontology=args.lambda_ontology,
        ontology_threshold=args.ontology_threshold,
        ontology_csv=args.ontology_input,
        row_id_column=args.row_id_column,
        device=getattr(args, "device", "auto"),
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout,
    )
