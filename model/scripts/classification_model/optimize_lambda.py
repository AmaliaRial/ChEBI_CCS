"""Lambda sweep for the ontology multitask model.

Applies the best baseline hyperparameters found by Optuna to the multitask model and trains one
model per ontology-loss weight (lambda) in a configurable range (default 0.1 .. 2.0, 20 values).
The selected lambda is the one with the lowest *validation* MAE (never test).

Each lambda writes a full set of artifacts to its own folder, and a global summary is saved to the
sweep output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]  # Up to project root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.chebi_model import train_model


def parse_best_params_txt(path: Path) -> dict:
    """Parse a ``best_params.txt`` produced by the baseline Optuna optimization."""
    skip_keys = {"Best Trial", "Best MAE (validation)", "Hyperparameters"}
    params: dict[str, float | int] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key in skip_keys or not value:
            continue
        try:
            params[key] = int(value)
        except ValueError:
            params[key] = float(value)
    return params


def load_best_hyperparams(best_params_file: Path) -> dict:
    """Load the best hyperparameters from ``best_params.txt`` or an Optuna ``.db``.

    Raises a clear error if the file is missing -- never silently falls back to defaults.
    """
    if not best_params_file.exists():
        raise FileNotFoundError(
            f"No se encontró el fichero de mejores hiperparámetros: {best_params_file}\n"
            f"Se esperaba 'hyperopt/best_params.txt' (o un '.db' de Optuna). "
            f"Ejecuta primero la optimización del modelo base o indica la ruta con --best-params-file."
        )

    if best_params_file.suffix == ".db":
        import optuna

        study = optuna.load_study(
            study_name="baseline_ccs_optimization",
            storage=f"sqlite:///{best_params_file}",
        )
        params = dict(study.best_params)
    else:
        params = parse_best_params_txt(best_params_file)

    required = ("n_layers", "learning_rate", "epochs", "dropout")
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(
            f"El fichero {best_params_file} no contiene los hiperparámetros requeridos: {missing}. "
            f"Hiperparámetros encontrados: {sorted(params)}"
        )
    return params


def build_hidden_dims(params: dict) -> tuple[int, ...]:
    """Assemble the hidden-layer sizes from ``n_layers`` + ``n_units_l0..l{n-1}``."""
    n_layers = int(params["n_layers"])
    hidden_dims: list[int] = []
    for index in range(n_layers):
        key = f"n_units_l{index}"
        if key not in params:
            raise ValueError(
                f"Falta '{key}' para reconstruir la arquitectura (n_layers={n_layers}). "
                f"Hiperparámetros disponibles: {sorted(params)}"
            )
        hidden_dims.append(int(params[key]))
    if not hidden_dims:
        raise ValueError("No se pudo construir hidden_dims a partir de los mejores hiperparámetros.")
    return tuple(hidden_dims)


def make_lambda_grid(lambda_min: float, lambda_max: float, lambda_step: float) -> list[float]:
    grid = np.arange(lambda_min, lambda_max + lambda_step / 2.0, lambda_step)
    return [round(float(value), 1) for value in grid]


def read_split_metrics(summary_path: Path) -> dict[str, float]:
    """Read validation/test metrics written by ``train_model`` (it returns None)."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary.get("metrics", {})
    val = metrics.get("val", {})
    test = metrics.get("test", {})
    return {
        "val_mae": float(val.get("mae", np.inf)),
        "val_medae": float(val.get("medae", np.nan)),
        "val_rmse": float(val.get("rmse", np.nan)),
        "val_r2": float(val.get("r2", np.nan)),
        "test_mae": float(test.get("mae", np.nan)),
        "test_medae": float(test.get("medae", np.nan)),
        "test_rmse": float(test.get("rmse", np.nan)),
        "test_r2": float(test.get("r2", np.nan)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Barrido de lambda (peso de la pérdida ontológica) para el modelo multitarea." )
    parser.add_argument("--best-params-file", default=str(REPO_ROOT / "hyperopt" / "best_params.txt"), help="Fichero con los mejores hiperparámetros de Optuna (best_params.txt o un .db).")
    parser.add_argument("--train-input", default=str(REPO_ROOT / "predictions" / "base" / "train_split.csv"))
    parser.add_argument("--val-input", default=str(REPO_ROOT / "predictions" / "base" / "val_split.csv"))
    parser.add_argument("--test-input", default=str(REPO_ROOT / "predictions" / "base" / "test_split.csv"))
    parser.add_argument("--ontology-input", default=str(REPO_ROOT / "data" / "model" / "final_covered_ccs_fingerprints_multilabel_filtered.csv") )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "predictions" / "lambda_sweep"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lambda-min", type=float, default=0.1)
    parser.add_argument("--lambda-max", type=float, default=2.0)
    parser.add_argument("--lambda-step", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=None, help="Override opcional del número de épocas (solo para pruebas rápidas). " "Por defecto usa las épocas óptimas de Optuna.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    best_params = load_best_hyperparams(Path(args.best_params_file))
    hidden_dims = build_hidden_dims(best_params)
    lr = float(best_params["learning_rate"])
    dropout = float(best_params["dropout"])
    epochs = int(args.epochs) if args.epochs is not None else int(best_params["epochs"])

    print("Mejores hiperparámetros cargados desde:", args.best_params_file)
    print(f"  Arquitectura (hidden_dims): {hidden_dims}")
    print(f"  Learning rate: {lr}")
    print(f"  Épocas: {epochs}" + ("  (override)" if args.epochs is not None else ""))
    print(f"  Dropout: {dropout}")
    print(f"  Batch size: {args.batch_size}")

    lambdas = make_lambda_grid(args.lambda_min, args.lambda_max, args.lambda_step)
    n_lambdas = len(lambdas)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"BARRIDO DE LAMBDA — {n_lambdas} modelos (selección por MAE de VALIDACIÓN)")
    print(f"{'=' * 80}\n")

    results: list[dict] = []
    best_lambda: float | None = None
    best_val_mae = np.inf

    for position, lam in enumerate(lambdas, start=1):
        lambda_dir = output_dir / f"lambda_{lam}"
        print(f"[{position}/{n_lambdas}] lambda={lam} -> {lambda_dir}", flush=True)

        record: dict = {"lambda": lam, "output_dir": str(lambda_dir), "error": ""}
        try:
            train_model(
                train_csv=args.train_input,
                output_dir=str(lambda_dir),
                val_csv=args.val_input,
                test_csv=args.test_input,
                epochs=epochs,
                batch_size=args.batch_size,
                lr=lr,
                lambda_ontology=float(lam),
                ontology_csv=args.ontology_input,
                device=args.device,
                hidden_dims=hidden_dims,
                dropout_rate=dropout,
            )
            record.update(read_split_metrics(lambda_dir / "training_summary.json"))
            print(
                f"    VAL MAE = {record['val_mae']:.4f}  |  "
                f"VAL MedAE = {record['val_medae']:.4f}  |  "
                f"TEST MAE = {record['test_mae']:.4f}  |  "
                f"TEST MedAE = {record['test_medae']:.4f}"
            )
        except Exception as error:  # noqa: BLE001 - registramos el fallo y seguimos con el resto
            record["val_mae"] = float(np.inf)
            record["error"] = str(error)
            print(f"    ERROR: {error}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results.append(record)

        if record["val_mae"] < best_val_mae:
            best_val_mae = record["val_mae"]
            best_lambda = lam
        if best_lambda is not None:
            print(f"    Mejor lambda hasta ahora: {best_lambda} (VAL MAE = {best_val_mae:.4f})")

    columns = [
        "lambda",
        "val_mae",
        "val_medae",
        "val_rmse",
        "val_r2",
        "test_mae",
        "test_medae",
        "test_rmse",
        "test_r2",
        "output_dir",
        "error",
    ]
    results_df = pd.DataFrame(results).reindex(columns=columns)
    results_df.to_csv(output_dir / "lambda_sweep_results.csv", index=False)

    if best_lambda is None or not np.isfinite(best_val_mae):
        print("\nNingún entrenamiento terminó correctamente; revisa la columna 'error' del CSV.")
        print(f"Resultados guardados en {output_dir / 'lambda_sweep_results.csv'}")
        return

    hyperparameters_used = {
        "hidden_dims": list(hidden_dims),
        "n_layers": len(hidden_dims),
        "learning_rate": lr,
        "epochs": epochs,
        "dropout": dropout,
        "batch_size": args.batch_size,
    }
    best_summary = {
        "best_lambda": best_lambda,
        "best_val_mae": best_val_mae,
        "selection_criterion": "validation MAE",
        "hyperparameters": hyperparameters_used,
    }

    with (output_dir / "best_lambda.json").open("w", encoding="utf-8") as handle:
        json.dump(best_summary, handle, indent=2)

    with (output_dir / "best_lambda.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"Best Lambda: {best_lambda}\n")
        handle.write(f"Validation MAE: {best_val_mae:.4f}\n")
        handle.write("Selection criterion: validation MAE\n\n")
        handle.write("Optimized hyperparameters used:\n")
        handle.write(f"  Architecture (hidden_dims): {list(hidden_dims)}\n")
        handle.write(f"  Number of layers: {len(hidden_dims)}\n")
        handle.write(f"  Learning rate: {lr}\n")
        handle.write(f"  Epochs: {epochs}\n")
        handle.write(f"  Dropout: {dropout}\n")
        handle.write(f"  Batch size: {args.batch_size}\n")

    ranking = results_df.sort_values("val_mae", ascending=True, na_position="last")

    print(f"\n{'=' * 80}")
    print("RANKING DE LAMBDA (ordenado por MAE de VALIDACIÓN)")
    print(f"{'=' * 80}")
    print(ranking[["lambda", "val_mae", "val_medae", "val_rmse", "test_mae", "test_medae"]].to_string(index=False))
    print(f"\n{'-' * 80}")
    print(f"MEJOR LAMBDA: {best_lambda}  |  VALIDATION MAE: {best_val_mae:.4f}")
    print(f"{'-' * 80}\n")
    print(f"Resultados guardados en {output_dir / 'lambda_sweep_results.csv'}")
    print(f"Mejor lambda en {output_dir / 'best_lambda.txt'} y {output_dir / 'best_lambda.json'}")


if __name__ == "__main__":
    main()
