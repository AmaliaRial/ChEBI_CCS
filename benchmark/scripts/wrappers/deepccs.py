import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict

import pandas as pd


ADDUCT_MAP: Dict[str, str] = {
    "[M+H]+": "M+H",
    "[M+Na]+": "M+Na",
    "[M-H]-": "M-H",
    "[M-2H]2-": "M-2H",
}
SUPPORTED_ADDUCTS = {"M+H", "M+Na", "M-H", "M-2H"}


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _normalize_adduct(adduct: str) -> str:
    if not isinstance(adduct, str):
        return ""
    adduct = adduct.strip()
    return ADDUCT_MAP.get(adduct, adduct)


def _find_column(df: pd.DataFrame, target: str) -> str:
    lowered = {col.lower(): col for col in df.columns}
    if target.lower() not in lowered:
        raise ValueError("Missing column '{}'".format(target))
    return lowered[target.lower()]


def _load_smiles_tokens(smiles_encoder_path: Path) -> set:
    with smiles_encoder_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Encoder must be JSON object.")
    return set(data.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepCCS wrapper.")
    parser.add_argument("--input", required=True, help="Input CSV with smiles/adduct/ccs.")
    parser.add_argument("--output", required=True, help="Output predictions CSV.")
    parser.add_argument("--repo", required=True, help="Path to DeepCCS repo.")
    args = parser.parse_args()

    _setup_logging()

    repo_path = Path(args.repo)
    if not repo_path.exists():
        raise SystemExit("Repo not found: {}".format(repo_path))

    cli_path = repo_path / "interface" / "command_line_tool.py"
    model_dir = repo_path / "saved_models" / "default"
    if not cli_path.exists():
        raise SystemExit("CLI not found: {}".format(cli_path))
    if not model_dir.exists():
        raise SystemExit("Model not found: {}".format(model_dir))

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)
    smiles_col = _find_column(df, "smiles")
    adduct_col = _find_column(df, "adduct")

    df = df.copy()
    df["_row_id"] = df.index.astype(int)
    df["_deepccs_adduct"] = df[adduct_col].map(_normalize_adduct)
    supported_mask = df["_deepccs_adduct"].isin(SUPPORTED_ADDUCTS)
    unsupported_count = int((~supported_mask).sum())
    if unsupported_count:
        logging.warning("DeepCCS unsupported adducts: %d rows will be skipped.", unsupported_count)

    smiles_encoder_path = model_dir / "smiles_encoder.json"
    if not smiles_encoder_path.exists():
        raise SystemExit("Encoder not found: {}".format(smiles_encoder_path))

    try:
        from DeepCCS.model.splitter import SMILESsplitter
    except ImportError as exc:
        raise SystemExit("DeepCCS package not available in this environment.") from exc

    tokens = _load_smiles_tokens(smiles_encoder_path)
    splitter = SMILESsplitter()

    def _is_supported_smiles(value: str) -> bool:
        if not isinstance(value, str):
            return False
        return all(token in tokens for token in splitter.split(value))

    df_supported = df.loc[supported_mask, ["_row_id", smiles_col, "_deepccs_adduct"]].copy()
    df_supported["_smiles_supported"] = df_supported[smiles_col].map(_is_supported_smiles)
    unsupported_smiles_count = int((~df_supported["_smiles_supported"]).sum())
    if unsupported_smiles_count:
        logging.warning("DeepCCS unsupported SMILES tokens: %d rows will be skipped.", unsupported_smiles_count)

    df_supported = df_supported.loc[df_supported["_smiles_supported"], ["_row_id", smiles_col, "_deepccs_adduct"]]
    df_supported = df_supported.rename(columns={smiles_col: "SMILES", "_deepccs_adduct": "Adducts"})

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        temp_input = temp_dir_path / "deepccs_input.csv"
        temp_output = temp_dir_path / "deepccs_output.csv"

        if df_supported.empty:
            logging.warning("No rows eligible for DeepCCS prediction after filtering.")
            preds = pd.DataFrame(columns=["_row_id", "predicted_ccs"])
        else:
            df_supported.to_csv(temp_input, index=False)

            command = [
                sys.executable,
                str(cli_path),
                "predict",
                "-mp",
                str(model_dir),
                "-ap",
                str(model_dir),
                "-sp",
                str(model_dir),
                "-i",
                str(temp_input),
                "-o",
                str(temp_output),
            ]
            logging.info("Running DeepCCS: %s", " ".join(command))
            subprocess.run(command, check=True)

            if not temp_output.exists():
                raise SystemExit("DeepCCS did not produce an output file.")

            pred_df = pd.read_csv(temp_output)
            if "CCS_DeepCCS" not in pred_df.columns or "_row_id" not in pred_df.columns:
                raise ValueError("DeepCCS output missing required columns (_row_id, CCS_DeepCCS).")

            preds = pred_df[["_row_id", "CCS_DeepCCS"]].rename(columns={"CCS_DeepCCS": "predicted_ccs"})

    output_df = df[[smiles_col, adduct_col, "_row_id"]].merge(preds, on="_row_id", how="left")
    output_df = output_df.rename(columns={smiles_col: "smiles", adduct_col: "adduct"})
    output_df = output_df[["_row_id", "smiles", "adduct", "predicted_ccs"]]
    output_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

