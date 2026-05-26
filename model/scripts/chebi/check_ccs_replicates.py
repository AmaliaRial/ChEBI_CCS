#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("data/raw_datasets")
DEFAULT_OUTPUT_DIR = Path("data/clean_datasets/ccs_replicate_check")
DEFAULT_CV_THRESHOLD = 5.0
SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt"}

REPLICATE_PATTERNS = [
    re.compile(r"(?i)^ccs[\s_-]*\d+$"),
    re.compile(r"(?i)^ccs[\s_-]*rep(?:licate)?[\s_-]*\d+$"),
    re.compile(r"(?i)^rep(?:licate)?[\s_-]*ccs[\s_-]*\d+$"),
    re.compile(r"(?i)^collision[\s_-]*cross[\s_-]*section[\s_-]*\d+$"),
]


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first_line = handle.readline()
    if first_line.count("\t") > first_line.count(","):
        return "\t"
    return ","


def read_table(path: Path) -> pd.DataFrame:
    delimiter = detect_delimiter(path)
    return pd.read_csv(path, sep=delimiter, low_memory=False)


def find_replicate_columns(columns: list[str]) -> list[str]:
    replicate_columns: list[str] = []
    for column in columns:
        if any(pattern.match(column) for pattern in REPLICATE_PATTERNS):
            replicate_columns.append(column)
    return replicate_columns


def build_output_path(input_path: Path, input_root: Path, output_root: Path, suffix: str) -> Path:
    try:
        relative = input_path.relative_to(input_root)
    except ValueError:
        relative = input_path.name
        return output_root / f"{Path(relative).stem}{suffix}.csv"
    return output_root / relative.parent / f"{relative.stem}{suffix}.csv"


def compute_replicate_statistics(df: pd.DataFrame, replicate_columns: list[str], cv_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    replicate_values = df[replicate_columns].apply(pd.to_numeric, errors="coerce")
    replicate_count = replicate_values.notna().sum(axis=1)
    ccs_average = replicate_values.mean(axis=1, skipna=True)
    ccs_std = replicate_values.std(axis=1, skipna=True, ddof=0).fillna(0.0)

    average_values = ccs_average.to_numpy(dtype=np.float64)
    std_values = ccs_std.to_numpy(dtype=np.float64)
    count_values = replicate_count.to_numpy(dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ccs_cv_percent = np.where(
            np.isfinite(average_values) & (np.abs(average_values) > 0),
            std_values / np.abs(average_values) * 100.0,
            np.nan,
        )

    annotated_df = df.copy()
    annotated_df["ccs_average"] = ccs_average
    annotated_df["ccs_std"] = ccs_std
    annotated_df["ccs_cv_percent"] = ccs_cv_percent
    annotated_df["ccs_replicate_count"] = replicate_count.astype(int)
    annotated_df["ccs_replicate_columns"] = ";".join(replicate_columns)
    annotated_df["ccs"] = annotated_df["ccs_average"]

    accepted_mask = (replicate_count > 0) & np.isfinite(ccs_cv_percent) & (ccs_cv_percent <= cv_threshold)
    accepted_df = annotated_df.loc[accepted_mask].copy()
    discarded_df = annotated_df.loc[~accepted_mask].copy()

    report = {
        "input_rows": int(len(df)),
        "replicate_columns": replicate_columns,
        "accepted_rows": int(len(accepted_df)),
        "discarded_rows": int(len(discarded_df)),
        "cv_threshold": float(cv_threshold),
        "mean_replicate_count": float(replicate_count.mean()) if len(replicate_count) else 0.0,
        "max_replicate_count": int(replicate_count.max()) if len(replicate_count) else 0,
        "rows_with_missing_replicates": int((replicate_count == 0).sum()),
    }
    return accepted_df, discarded_df, report


def process_file(input_path: Path, input_root: Path, output_root: Path, cv_threshold: float) -> dict[str, Any]:
    df = read_table(input_path)
    replicate_columns = find_replicate_columns(list(df.columns))

    if not replicate_columns:
        return {
            "input_csv": str(input_path),
            "status": "skipped",
            "reason": "No replicate CCS columns found",
        }

    accepted_df, discarded_df, report = compute_replicate_statistics(df, replicate_columns, cv_threshold)

    accepted_path = build_output_path(input_path, input_root, output_root, "_accepted")
    discarded_path = build_output_path(input_path, input_root, output_root, "_discarded")
    report_path = build_output_path(input_path, input_root, output_root, "_report").with_suffix(".json")

    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    discarded_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    accepted_df.to_csv(accepted_path, index=False)
    discarded_df.to_csv(discarded_path, index=False)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_csv": str(input_path),
                "accepted_csv": str(accepted_path),
                "discarded_csv": str(discarded_path),
                **report,
            },
            handle,
            indent=2,
        )

    return {
        "input_csv": str(input_path),
        "status": "processed",
        "replicate_columns": replicate_columns,
        "accepted_csv": str(accepted_path),
        "discarded_csv": str(discarded_path),
        "report_json": str(report_path),
        **report,
    }


def collect_input_files(paths: list[Path], input_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend([candidate for candidate in sorted(path.rglob("*")) if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES])
        elif path.is_file():
            files.append(path)
    if not files and input_dir.exists():
        files.extend([candidate for candidate in sorted(input_dir.rglob("*")) if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES])
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detecta replicados CCS y filtra por CV.")
    parser.add_argument("--input", nargs="*", type=Path, default=[])
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cv-threshold", type=float, default=DEFAULT_CV_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_files = collect_input_files(args.input, args.input_dir)
    if not input_files:
        raise FileNotFoundError("No se encontraron archivos CSV/TSV para revisar.")

    summary: list[dict[str, Any]] = []
    for input_path in input_files:
        result = process_file(input_path, args.input_dir, args.output_dir, args.cv_threshold)
        summary.append(result)
        if result.get("status") == "processed":
            print(
                f"{input_path.name}: accepted={result['accepted_rows']} discarded={result['discarded_rows']} "
                f"replicate_cols={len(result['replicate_columns'])}"
            )
        else:
            print(f"{input_path.name}: skipped ({result.get('reason', 'no reason provided')})")

    summary_path = args.output_dir / "ccs_replicate_scan_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
