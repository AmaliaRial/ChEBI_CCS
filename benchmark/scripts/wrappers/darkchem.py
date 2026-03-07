import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _find_column(df: pd.DataFrame, target: str) -> str:
    lowered = {col.lower(): col for col in df.columns}
    if target.lower() not in lowered:
        raise ValueError("Missing required column '{}'".format(target))
    return lowered[target.lower()]


def _is_network_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and (path / "arguments.txt").exists()


def _discover_networks(repo_path: Path, user_network_dir: Optional[str]) -> Dict[str, Path]:
    candidates = []

    if user_network_dir:
        network_base = Path(user_network_dir)
        if not network_base.is_absolute():
            network_base = repo_path / network_base
        candidates.append(network_base)
    else:
        candidates.extend(
            [
                repo_path / "darkchem-weights",
                repo_path / "Pretrained",
                repo_path / "saved_models",
                repo_path / "models",
            ]
        )

    for candidate in candidates:
        if _is_network_dir(candidate):
            return {"default": candidate}

        if candidate.exists() and candidate.is_dir():
            found = {}
            for child in candidate.iterdir():
                if child.is_dir() and _is_network_dir(child):
                    found[child.name.lower()] = child
            if found:
                return found

    return {}


def _normalize_group(adduct: str) -> str:
    if not isinstance(adduct, str):
        return "protonated"

    ad = adduct.strip().lower()

    if "na" in ad:
        return "sodiated"

    if ad.endswith("-") or "-]" in ad or "hcoo" in ad or "ch3coo" in ad or "cl" in ad:
        return "deprotonated"

    return "protonated"


def _predict_subset(
    df_subset: pd.DataFrame,
    smiles_col: str,
    network_dir: Path,
    property_index: int,
) -> pd.DataFrame:
    if df_subset.empty:
        return pd.DataFrame(columns=["_row_id", "predicted_ccs"])

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        temp_input = temp_dir_path / "darkchem_input.tsv"
        temp_output = temp_dir_path / "darkchem_input_darkchem.tsv"

        df_input = df_subset[["_row_id", smiles_col]].copy()
        df_input = df_input.rename(columns={smiles_col: "SMILES"})
        df_input.to_csv(temp_input, sep="\t", index=False)

        command = [
            sys.executable,
            "-c",
            "from darkchem.cli import main; main()",
            "predict",
            "prop",
            str(temp_input),
            str(network_dir),
        ]
        logging.info("Running DarkChem: %s", " ".join(command))
        subprocess.run(command, check=True, cwd=str(temp_dir_path))

        if not temp_output.exists():
            generated = sorted([p.name for p in temp_dir_path.glob("*darkchem*.tsv")])
            raise SystemExit(
                "DarkChem did not produce expected output: {}. Generated files: {}".format(
                    temp_output, generated
                )
            )

        pred_df = pd.read_csv(temp_output, sep="\t")

    prop_cols = sorted([col for col in pred_df.columns if col.startswith("prop_")])
    if not prop_cols:
        raise ValueError("DarkChem output has no property predictions (prop_* columns).")
    if "_row_id" not in pred_df.columns:
        raise ValueError("DarkChem output missing _row_id column.")

    if property_index < 0 or property_index >= len(prop_cols):
        raise ValueError(
            "property_index={} out of range. Available properties: {}".format(
                property_index, prop_cols
            )
        )

    ccs_col = prop_cols[property_index]
    logging.info("Using %s as predicted CCS", ccs_col)

    return pred_df[["_row_id", ccs_col]].rename(columns={ccs_col: "predicted_ccs"})


def main() -> None:
    parser = argparse.ArgumentParser(description="DarkChem wrapper.")
    parser.add_argument("--input", required=True, help="Input CSV with smiles/adduct/ccs.")
    parser.add_argument("--output", required=True, help="Output predictions CSV.")
    parser.add_argument("--repo", required=True, help="Path to DarkChem repo.")
    parser.add_argument(
        "--network-dir",
        default=None,
        help="DarkChem network folder, or parent folder containing adduct-specific subfolders.",
    )
    parser.add_argument(
        "--property-index",
        type=int,
        default=0,
        help="Index of prop_* column to use as CCS (default: 0).",
    )
    args = parser.parse_args()

    _setup_logging()

    repo_path = Path(args.repo)
    if not repo_path.exists():
        raise SystemExit("Repo not found: {}".format(repo_path))

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)
    smiles_col = _find_column(df, "smiles")
    adduct_col = _find_column(df, "adduct")

    df = df.copy()
    df["_row_id"] = df.index.astype(int)
    df["_darkchem_group"] = df[adduct_col].map(_normalize_group)

    networks = _discover_networks(repo_path, args.network_dir)
    if not networks:
        logging.warning(
            "DarkChem network not found. Expected arguments.txt in a model folder. Writing NaN predictions."
        )
        output_df = df[[smiles_col, adduct_col]].copy()
        output_df = output_df.rename(columns={smiles_col: "smiles", adduct_col: "adduct"})
        output_df["predicted_ccs"] = None
        output_df.to_csv(output_path, index=False)
        logging.info("Stub output saved to: %s", output_path)
        return

    logging.info("Discovered DarkChem networks: %s", ", ".join(sorted(networks.keys())))

    pred_parts = []
    for group_name, group_df in df.groupby("_darkchem_group"):
        network_dir = networks.get(group_name) or networks.get("default")
        if network_dir is None:
            logging.warning("No DarkChem network available for group '%s'. Rows will be NaN.", group_name)
            continue

        logging.info("Predicting group '%s' with network: %s (rows=%d)", group_name, network_dir, len(group_df))
        pred_part = _predict_subset(group_df, smiles_col, network_dir, args.property_index)
        pred_parts.append(pred_part)

    if pred_parts:
        preds = pd.concat(pred_parts, ignore_index=True)
    else:
        preds = pd.DataFrame(columns=["_row_id", "predicted_ccs"])

    output_df = df[[smiles_col, adduct_col, "_row_id"]].merge(preds, on="_row_id", how="left")
    output_df = output_df.rename(columns={smiles_col: "smiles", adduct_col: "adduct"})
    output_df = output_df[["_row_id", "smiles", "adduct", "predicted_ccs"]]
    output_df.to_csv(output_path, index=False)

    logging.info("Predictions saved to: %s", output_path)


if __name__ == "__main__":
    main()
