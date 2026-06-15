#!/usr/bin/env python3
"""Split the final fingerprints dataset into 80/10/10 train/val/test."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


repo_root = Path(__file__).resolve().parents[4]

INPUT_CSV = repo_root / "data" / "model" / "final_covered_ccs_fingerprints.csv"
VAL_CSV = repo_root / "data" / "model" / "val_ccs_fingerprints.csv"
TEST_CSV = repo_root / "data" / "model" / "test_ccs_fingerprints.csv"
TRAIN_CSV = repo_root / "data" / "model" / "train_ccs_fingerprints.csv"
MANIFEST_JSON = repo_root / "data" / "model" / "split_manifest.json"


def split_train_val_test(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split 80/10/10: train, val, test."""
    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        random_state=random_state,
        shuffle=True,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=random_state,
        shuffle=True,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_split_train_val_test(input_csv: str | Path, train_csv: str | Path, val_csv: str | Path, test_csv: str | Path, val_size: float = 0.1, test_size: float = 0.1, random_state: int = 42, manifest_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_csv, low_memory=False)
    train_df, val_df, test_df = split_train_val_test(
        df, val_size=val_size, test_size=test_size, random_state=random_state
    )

    Path(train_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(val_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(test_csv).parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    if manifest_path is not None:
        manifest = {
            "input_csv": str(input_csv),
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "test_csv": str(test_csv),
            "total_rows": int(len(df)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "val_size": float(val_size),
            "test_size": float(test_size),
            "random_state": int(random_state),
        }
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(manifest_path).open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    return train_df, val_df, test_df


def main() -> None:
    print(f"Input CSV: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        print(f"ERROR: Input file not found: {INPUT_CSV}")
        sys.exit(1)

    print("Performing 80/10/10 split...")
    train_df, val_df, test_df = save_split_train_val_test(
        input_csv=INPUT_CSV,
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        val_size=0.1,
        test_size=0.1,
        random_state=42,
        manifest_path=MANIFEST_JSON,
    )

    print("\n✓ Split complete!")
    print(f"  Train: {len(train_df)} rows → {TRAIN_CSV}")
    print(f"  Val:   {len(val_df)} rows → {VAL_CSV}")
    print(f"  Test:  {len(test_df)} rows → {TEST_CSV}")
    print(f"  Manifest: {MANIFEST_JSON}")


if __name__ == "__main__":
    main()
