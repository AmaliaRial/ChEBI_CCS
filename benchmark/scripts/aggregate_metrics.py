import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _find_column(df: pd.DataFrame, name: str) -> str:
    lowered = {col.lower(): col for col in df.columns}
    if name.lower() not in lowered:
        raise ValueError("Missing required column '{}'".format(name))
    return lowered[name.lower()]


def _load_predictions(pred_path: Path, model_name: str) -> Tuple[pd.DataFrame, str]:
    pred_df = pd.read_csv(pred_path, low_memory=False)
    pred_col = _find_column(pred_df, "predicted_ccs")

    if "_row_id" in pred_df.columns:
        out = pred_df[["_row_id", pred_col]].copy()
        out = out.rename(columns={pred_col: "predicted_ccs_{}".format(model_name)})
        return out, "_row_id"

    smiles_col = _find_column(pred_df, "smiles")
    adduct_col = _find_column(pred_df, "adduct")

    out = pred_df[[smiles_col, adduct_col, pred_col]].copy()
    out = out.rename(
        columns={
            smiles_col: "smiles",
            adduct_col: "adduct",
            pred_col: "predicted_ccs_{}".format(model_name),
        }
    )

    out = out.drop_duplicates(subset=["smiles", "adduct"], keep="first")
    return out, "smiles_adduct"


def _compute_metrics(df: pd.DataFrame, pred_col: str) -> dict:
    valid = df[["ccs", pred_col]].dropna()
    valid = valid[(valid["ccs"] != 0) & (valid["ccs"].abs() > 1e-6)]
    if valid.empty:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "mpe": np.nan,
            "mape": np.nan,
            "std_abs_error": np.nan,
            "std_pct_error": np.nan,
            "outliers_gt_10pct": 0,
        }

    y_true = valid["ccs"].astype(float).to_numpy()
    y_pred = valid[pred_col].astype(float).to_numpy()

    abs_err = np.abs(y_pred - y_true)
    pct_err = 100.0 * (y_pred - y_true) / y_true
    abs_pct_err = np.abs(pct_err)

    return {
        "n": int(len(valid)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mpe": float(np.mean(pct_err)),
        "mape": float(np.mean(abs_pct_err)),
        "std_abs_error": float(np.std(abs_err, ddof=0)),
        "std_pct_error": float(np.std(pct_err, ddof=0)),
        "outliers_gt_10pct": int(np.sum(abs_pct_err > 10.0)),
    }


def _collect_prediction_files(predictions_dir: Path) -> List[Path]:
    return sorted(predictions_dir.glob("*/predictions.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate predictions and compute metrics.")
    parser.add_argument("--input", required=True, help="Input CSV with smiles/adduct/ccs.")
    parser.add_argument("--predictions-dir", default="predictions", help="Predictions folder.")
    parser.add_argument("--output-merged", default="reports/benchmark_predictions.csv", help="Merged CSV.")
    parser.add_argument("--output-metrics", default="reports/metrics.csv", help="Metrics CSV.")
    parser.add_argument("--by-dataset", action="store_true", help="Group by source_dataset.")
    args = parser.parse_args()

    base_df = pd.read_csv(Path(args.input), low_memory=False)
    smiles_col = _find_column(base_df, "smiles")
    adduct_col = _find_column(base_df, "adduct")
    ccs_col = _find_column(base_df, "ccs")

    base_df = base_df.copy()
    base_df = base_df.rename(columns={smiles_col: "smiles", adduct_col: "adduct", ccs_col: "ccs"})
    base_df["_row_id"] = np.arange(len(base_df), dtype=int)

    pred_files = _collect_prediction_files(Path(args.predictions_dir))
    if not pred_files:
        raise SystemExit("No prediction files found under {}".format(args.predictions_dir))

    merged_df = base_df
    model_names: List[str] = []

    for pred_file in pred_files:
        model_name = pred_file.parent.name
        pred_df, merge_mode = _load_predictions(pred_file, model_name)

        if merge_mode == "_row_id":
            merged_df = merged_df.merge(pred_df, on="_row_id", how="left")
        else:
            merged_df = merged_df.merge(pred_df, on=["smiles", "adduct"], how="left")

        model_names.append(model_name)

    output_merged = Path(args.output_merged)
    output_merged.parent.mkdir(parents=True, exist_ok=True)

    keep_cols = ["smiles", "adduct", "ccs"]
    if "source_dataset" in merged_df.columns:
        keep_cols.append("source_dataset")
    keep_cols.extend([col for col in merged_df.columns if col.startswith("predicted_ccs_")])
    merged_df[keep_cols].to_csv(output_merged, index=False)

    metrics_rows = []
    for model_name in model_names:
        pred_col = "predicted_ccs_{}".format(model_name)
        row = _compute_metrics(merged_df, pred_col)
        row.update({"model": model_name, "group": "overall"})
        metrics_rows.append(row)

        if args.by_dataset and "source_dataset" in merged_df.columns:
            for dataset_name, group_df in merged_df.groupby("source_dataset"):
                ds_row = _compute_metrics(group_df, pred_col)
                ds_row.update({"model": model_name, "group": str(dataset_name)})
                metrics_rows.append(ds_row)

    metrics_df = pd.DataFrame(metrics_rows)
    output_metrics = Path(args.output_metrics)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_metrics, index=False)


if __name__ == "__main__":
    main()
