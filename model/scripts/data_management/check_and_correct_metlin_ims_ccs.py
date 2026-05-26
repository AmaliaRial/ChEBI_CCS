#Para Metlin CCS que contiene varios CCS . Se mira la varianza de los CCS para cada compuesto. 
#Si la varianza es muy grande, descartamos ese compuesto. Aceptamos una varianza maxima de 5%.
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RAW_PATH = Path("data/raw_datasets/fingerprints/METLIN_IMS_vectorfingerprintsVectorized.tsv")
DEFAULT_COVERED_PATH = Path("data/model/final_covered_ccs.csv")
DEFAULT_OUTPUT_PATH = Path("data/model/final_covered_ccs_corrected.csv")
DEFAULT_REPORT_PATH = Path("data/model/reports/metlin_ims_ccs_check_report.csv")
DEFAULT_DISCARDED_PATH = Path("data/model/reports/metlin_ims_ccs_discarded_high_cv.csv")


def normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return re.sub(r"\s+", " ", text).lower()


def normalize_numeric(value: object, decimals: int = 4) -> str | None:
    if pd.isna(value):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return normalize_text(value)
    if np.isnan(numeric_value):
        return None
    return f"{numeric_value:.{decimals}f}"


def resolve_column(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {column.strip().lower(): column for column in columns}
    for candidate in candidates:
        found = lookup.get(candidate.strip().lower())
        if found is not None:
            return found
    return None


def annotate_raw_metrics(raw_df: pd.DataFrame, cv_threshold: float) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df.insert(0, "raw_row_id", np.arange(1, len(raw_df) + 1))

    replicate_columns = [
        column
        for column in raw_df.columns
        if re.match(r"(?i)^ccs[_\s-]?(\d+|rep\d+|replicate\d+)$", column)
    ]
    average_column = resolve_column(
        list(raw_df.columns),
        ["CCS_AVG", "CCS_AV", "CCS_AVERAGE", "CCS_MEAN", "AVERAGE_CCS", "MEAN_CCS"],
    )
    reported_cv_column = resolve_column(list(raw_df.columns), ["% CV", "CV", "ccs_cv_percent"])

    replicate_values = raw_df[replicate_columns].apply(pd.to_numeric, errors="coerce") if replicate_columns else pd.DataFrame(index=raw_df.index)
    replicate_count = replicate_values.notna().sum(axis=1) if replicate_columns else pd.Series(0, index=raw_df.index)
    computed_average = replicate_values.mean(axis=1) if replicate_columns else pd.Series(np.nan, index=raw_df.index)
    computed_std = replicate_values.std(axis=1, ddof=1) if replicate_columns else pd.Series(np.nan, index=raw_df.index)
    computed_cv = (computed_std.abs() / computed_average.abs()) * 100 if replicate_columns else pd.Series(np.nan, index=raw_df.index)
    computed_cv = computed_cv.where((replicate_count >= 2) & computed_average.notna() & (computed_average != 0))

    if average_column is not None:
        average_source = pd.to_numeric(raw_df[average_column], errors="coerce")
        raw_df["ccs_average_raw"] = average_source.fillna(computed_average)
        raw_df["ccs_average_source"] = np.where(average_source.notna(), average_column, "computed_from_replicates")
    else:
        raw_df["ccs_average_raw"] = computed_average
        raw_df["ccs_average_source"] = "computed_from_replicates"

    if reported_cv_column is not None:
        raw_df["ccs_cv_reported"] = pd.to_numeric(raw_df[reported_cv_column], errors="coerce")
        raw_df["ccs_cv_percent"] = computed_cv.fillna(raw_df["ccs_cv_reported"])
    else:
        raw_df["ccs_cv_reported"] = np.nan
        raw_df["ccs_cv_percent"] = computed_cv

    raw_df["ccs_std_raw"] = computed_std
    raw_df["ccs_replicate_count"] = replicate_count
    raw_df["ccs_accept"] = raw_df["ccs_cv_percent"].le(cv_threshold)
    return raw_df


def build_key(df: pd.DataFrame, column_names: list[str], mz_decimals: int = 4) -> pd.Series:
    parts = []
    for column_name in column_names:
        if column_name not in df.columns:
            parts.append(pd.Series([None] * len(df), index=df.index))
            continue
        if column_name.lower() in {"mz", "m/z"}:
            parts.append(df[column_name].apply(lambda value: normalize_numeric(value, decimals=mz_decimals)))
        elif column_name.lower() == "row_id":
            parts.append(df[column_name].apply(lambda value: normalize_numeric(value, decimals=0)))
        else:
            parts.append(df[column_name].apply(normalize_text))

    key = parts[0]
    for part in parts[1:]:
        key = key.fillna("") + "|" + part.fillna("")
    key = key.replace(r"^\|+|\|+$", "", regex=True)
    key = key.mask(key.str.contains(r"^\|*$", regex=True), None)
    return key


def prepare_matching_frame(df: pd.DataFrame, mz_decimals: int = 4) -> pd.DataFrame:
    prepared = df.copy()
    prepared["_row_id_key"] = build_key(prepared, ["row_id"], mz_decimals=mz_decimals) if "row_id" in prepared.columns else None
    prepared["_inchi_key"] = build_key(prepared, ["inchi", "adduct"], mz_decimals=mz_decimals) if {"inchi", "adduct"}.issubset(prepared.columns) else None
    prepared["_smiles_key"] = build_key(prepared, ["smiles", "adduct"], mz_decimals=mz_decimals) if {"smiles", "adduct"}.issubset(prepared.columns) else None
    prepared["_name_key"] = build_key(prepared, ["name", "adduct"], mz_decimals=mz_decimals) if {"name", "adduct"}.issubset(prepared.columns) else None
    prepared["_mz_key"] = build_key(prepared, ["mz", "adduct"], mz_decimals=mz_decimals) if {"mz", "adduct"}.issubset(prepared.columns) else None
    prepared["_inchikey_key"] = build_key(prepared, ["inchikey"], mz_decimals=mz_decimals) if "inchikey" in prepared.columns else None
    return prepared


def match_metlin_rows(covered_df: pd.DataFrame, raw_df: pd.DataFrame, mz_decimals: int = 4) -> pd.DataFrame:
    covered = covered_df.copy()
    raw = raw_df.copy()

    if "row_id" not in raw.columns:
        raw.insert(0, "row_id", raw["raw_row_id"])

    covered = prepare_matching_frame(covered, mz_decimals=mz_decimals)
    raw = prepare_matching_frame(raw, mz_decimals=mz_decimals)

    covered["match_strategy"] = pd.NA
    covered["match_raw_row_id"] = pd.NA
    covered["match_ccs_average_raw"] = np.nan
    covered["match_ccs_cv_percent"] = np.nan
    covered["match_ccs_std_raw"] = np.nan
    covered["match_ccs_accept"] = pd.NA
    covered["match_ccs_average_source"] = pd.NA
    covered["match_ccs_replicate_count"] = np.nan

    strategies = [
        ("row_id", "_row_id_key"),
        ("inchikey", "_inchikey_key"),
        ("inchi_plus_adduct", "_inchi_key"),
        ("smiles_plus_adduct", "_smiles_key"),
        ("name_plus_adduct", "_name_key"),
        ("mz_plus_adduct", "_mz_key"),
    ]

    for strategy_name, key_column in strategies:
        unmatched = covered[covered["match_raw_row_id"].isna()].copy()
        if unmatched.empty or key_column not in raw.columns:
            continue

        raw_subset = raw.loc[raw[key_column].notna(), [
            key_column,
            "raw_row_id",
            "ccs_average_raw",
            "ccs_cv_percent",
            "ccs_std_raw",
            "ccs_accept",
            "ccs_average_source",
            "ccs_replicate_count",
        ]].drop_duplicates(subset=[key_column], keep="first")

        if raw_subset.empty:
            continue

        merged = unmatched.merge(raw_subset, on=key_column, how="left")
        matched = merged[merged["raw_row_id"].notna()].copy()
        if matched.empty:
            continue

        matched = matched.set_index("row_id")
        matched_row_ids = matched.index
        covered.loc[covered["row_id"].isin(matched_row_ids), "match_strategy"] = strategy_name
        covered.loc[covered["row_id"].isin(matched_row_ids), "match_raw_row_id"] = covered.loc[covered["row_id"].isin(matched_row_ids), "row_id"].map(matched["raw_row_id"])
        covered.loc[covered["row_id"].isin(matched_row_ids), "match_ccs_average_raw"] = covered.loc[covered["row_id"].isin(matched_row_ids), "row_id"].map(matched["ccs_average_raw"])
        covered.loc[covered["row_id"].isin(matched_row_ids), "match_ccs_cv_percent"] = covered.loc[covered["row_id"].isin(matched_row_ids), "row_id"].map(matched["ccs_cv_percent"])
        covered.loc[covered["row_id"].isin(matched_row_ids), "match_ccs_std_raw"] = covered.loc[covered["row_id"].isin(matched_row_ids), "row_id"].map(matched["ccs_std_raw"])
        covered.loc[covered["row_id"].isin(matched_row_ids), "match_ccs_accept"] = covered.loc[covered["row_id"].isin(matched_row_ids), "row_id"].map(matched["ccs_accept"])
        covered.loc[covered["row_id"].isin(matched_row_ids), "match_ccs_average_source"] = covered.loc[covered["row_id"].isin(matched_row_ids), "row_id"].map(matched["ccs_average_source"])
        covered.loc[covered["row_id"].isin(matched_row_ids), "match_ccs_replicate_count"] = covered.loc[covered["row_id"].isin(matched_row_ids), "row_id"].map(matched["ccs_replicate_count"])

    return covered


def build_reports(covered_df: pd.DataFrame, raw_df: pd.DataFrame, cv_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metlin_mask = covered_df["source_dataset"].astype(str).str.upper().eq("METLIN_IMS")
    metlin_rows = covered_df.loc[metlin_mask].copy()

    raw_with_metrics = annotate_raw_metrics(raw_df, cv_threshold=cv_threshold)
    matched = match_metlin_rows(metlin_rows, raw_with_metrics)

    matched["ccs_original"] = pd.to_numeric(matched["ccs"], errors="coerce")
    matched["ccs_corrected"] = matched["ccs_original"]
    matched["ccs_delta"] = np.nan
    matched["ccs_match_status"] = "unmatched"
    matched["match_ccs_accept"] = matched["match_ccs_cv_percent"].le(cv_threshold)

    accepted_mask = matched["match_raw_row_id"].notna() & matched["match_ccs_accept"].fillna(False)
    discarded_mask = matched["match_raw_row_id"].notna() & ~matched["match_ccs_accept"].fillna(False)

    matched.loc[accepted_mask, "ccs_corrected"] = matched.loc[accepted_mask, "match_ccs_average_raw"]
    matched.loc[accepted_mask, "ccs_delta"] = matched.loc[accepted_mask, "ccs_corrected"] - matched.loc[accepted_mask, "ccs_original"]
    matched.loc[accepted_mask, "ccs_match_status"] = "corrected"

    matched.loc[discarded_mask, "ccs_delta"] = matched.loc[discarded_mask, "match_ccs_average_raw"] - matched.loc[discarded_mask, "ccs_original"]
    matched.loc[discarded_mask, "ccs_match_status"] = "discarded_high_cv"

    report = matched[
        [
            "row_id",
            "name",
            "adduct",
            "smiles",
            "inchi",
            "mz",
            "ccs_original",
            "ccs_corrected",
            "match_ccs_average_raw",
            "match_ccs_std_raw",
            "match_ccs_cv_percent",
            "match_ccs_replicate_count",
            "match_ccs_average_source",
            "match_strategy",
            "match_raw_row_id",
            "ccs_match_status",
            "ccs_delta",
        ]
    ].copy()
    report = report.rename(
        columns={
            "match_ccs_average_raw": "ccs_average_raw",
            "match_ccs_std_raw": "ccs_std_raw",
            "match_ccs_cv_percent": "ccs_cv_percent",
            "match_ccs_replicate_count": "ccs_replicate_count",
            "match_ccs_average_source": "ccs_average_source",
            "match_raw_row_id": "raw_row_id",
        }
    )

    discarded = report.loc[report["ccs_match_status"].eq("discarded_high_cv")].copy()
    corrected_subset = report.loc[~report["ccs_match_status"].eq("discarded_high_cv")].copy()
    return report, discarded, corrected_subset


def apply_corrections(covered_df: pd.DataFrame, report_df: pd.DataFrame) -> pd.DataFrame:
    corrected = covered_df.copy()
    report_indexed = report_df.set_index("row_id")

    corrected_row_ids = report_indexed.index[report_indexed["ccs_match_status"].eq("corrected")]
    if len(corrected_row_ids) > 0:
        corrected_values = report_indexed.loc[corrected_row_ids, "ccs_corrected"]
        mask = corrected["row_id"].isin(corrected_row_ids)
        corrected.loc[mask, "ccs"] = corrected.loc[mask, "row_id"].map(corrected_values)

    discarded_row_ids = report_indexed.index[report_indexed["ccs_match_status"].eq("discarded_high_cv")]
    if len(discarded_row_ids) > 0:
        corrected = corrected.loc[~corrected["row_id"].isin(discarded_row_ids)].copy()

    return corrected.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check and correct CCS values for METLIN IMS rows in the covered dataset.")
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH, help="Path to the raw METLIN IMS CSV/TSV file.")
    parser.add_argument("--covered-path", type=Path, default=DEFAULT_COVERED_PATH, help="Path to the covered CCS CSV.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path for the corrected covered CSV.")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH, help="Path for the row-level CCS report CSV.")
    parser.add_argument("--discarded-path", type=Path, default=DEFAULT_DISCARDED_PATH, help="Path for the discarded high-CV rows CSV.")
    parser.add_argument("--cv-threshold", type=float, default=5.0, help="Maximum accepted CCS coefficient of variation in percent.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_df = pd.read_csv(args.raw_path, encoding="utf-8-sig")
    covered_df = pd.read_csv(args.covered_path)

    report_df, discarded_df, corrected_subset = build_reports(covered_df, raw_df, cv_threshold=args.cv_threshold)
    corrected_df = apply_corrections(covered_df, corrected_subset)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.discarded_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    report_df.to_csv(args.report_path, index=False)
    discarded_df.to_csv(args.discarded_path, index=False)
    corrected_df.to_csv(args.output_path, index=False)

    metlin_total = len(report_df)
    matched_total = int(report_df["raw_row_id"].notna().sum())
    corrected_total = int(report_df["ccs_match_status"].eq("corrected").sum())
    discarded_total = int(report_df["ccs_match_status"].eq("discarded_high_cv").sum())
    unmatched_total = int(report_df["ccs_match_status"].eq("unmatched").sum())

    print(f"Loaded raw rows: {len(raw_df)}")
    print(f"Loaded covered rows: {len(covered_df)}")
    print(f"METLIN_IMS rows in covered dataset: {metlin_total}")
    print(f"Matched METLIN_IMS rows: {matched_total}")
    print(f"Corrected METLIN_IMS rows: {corrected_total}")
    print(f"Discarded METLIN_IMS rows with CV > {args.cv_threshold:.1f}%: {discarded_total}")
    print(f"Unmatched METLIN_IMS rows: {unmatched_total}")
    print(f"Wrote report: {args.report_path}")
    print(f"Wrote discarded rows: {args.discarded_path}")
    print(f"Wrote corrected dataset: {args.output_path}")


if __name__ == "__main__":
    main()