#!/usr/bin/env python3
"""Build the final covered CCS dataset using fingerprints only.

The input is the covered no-ontology CSV. The output keeps the covered compound
rows and appends only fingerprint vector columns (V1..Vn) from the matching raw
fingerprint tables. Descriptors are intentionally excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_CSV = Path("data/model/final_covered_ccs.csv")
DEFAULT_OUT_CSV = Path("data/model/final_covered_ccs_fingerprints.csv")
DEFAULT_MANIFEST_OUT = Path("data/model/final_covered_ccs_fingerprints_manifest.json")

DATASET_SPECS: dict[str, Path] = {
    "ccsbase_descriptors": Path("data/raw_datasets/fingerprints/ccsbase_vectorfingerprintsVectorized.csv"),
    "AllCCS2_experimental_with_inchis_descriptors": Path("data/raw_datasets/fingerprints/AllCCS2_experimental_with_inchis_vectorfingerprintsVectorized.csv"),
    "METLIN-CCS-Lipids": Path("data/raw_datasets/fingerprints/METLIN-CCS-Lipids_vectorfingerprintsVectorized.csv"),
    "METLIN_IMS": Path("data/raw_datasets/fingerprints/METLIN_IMS_vectorfingerprintsVectorized.tsv"),
}

CORE_COLUMNS = ["row_id", "smiles", "adduct", "ccs", "inchi", "name", "mz"]
CHEBI_COLUMNS = ["chebi_classes", "chebi_count", "chebi_name", "chebi_match_source"]
OUTPUT_EXCLUDE_COLUMNS = set(CORE_COLUMNS + ["source_dataset"] + CHEBI_COLUMNS)

SOURCE_KEY_CANDIDATES: dict[str, list[tuple[str, ...]]] = {
    "default": [("inchi",), ("smiles",), ("name",)],
    "ccsbase_descriptors": [("smiles",), ("inchi",), ("name",)],
    "AllCCS2_experimental_with_inchis_descriptors": [("inchi",), ("smiles",), ("name",)],
    "METLIN-CCS-Lipids": [("inchi",), ("smiles",), ("name",)],
    "METLIN_IMS": [("inchi",), ("smiles",), ("name",)],
}

RAW_KEY_CANDIDATES: dict[str, list[tuple[str, ...]]] = {
    "default": [("smi",), ("InChI",), ("smiles",), ("inchi",), ("name",), ("Name",), ("Molecule Name",)],
    "ccsbase_descriptors": [("smi",), ("name",)],
    "AllCCS2_experimental_with_inchis_descriptors": [("InChI",), ("Structure",), ("Name",)],
    "METLIN-CCS-Lipids": [("InChI",), ("Name",)],
    "METLIN_IMS": [("inchi",), ("smiles",), ("Molecule Name",), ("name",)],
}


def normalize(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def get_value_case_insensitive(row: dict[str, str], field: str) -> str:
    if field in row:
        return normalize(row[field])
    target = field.lower()
    for key, value in row.items():
        if key.lower() == target:
            return normalize(value)
    return ""


def read_table(path: Path) -> tuple[list[dict[str, str]], list[str], str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first_line = handle.readline()
        delimiter = "\t" if first_line.count("\t") > first_line.count(",") else ","
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames), delimiter


def fingerprint_columns(headers: list[str]) -> list[str]:
    pattern = re.compile(r"^V\d+$")
    columns = [column for column in headers if pattern.match(column)]
    columns.sort(key=lambda name: int(name[1:]))
    return columns


def build_key(row: dict[str, str], fields: tuple[str, ...]) -> tuple[str, ...] | None:
    values: list[str] = []
    for field in fields:
        value = get_value_case_insensitive(row, field)
        if not value:
            return None
        values.append(value)
    return tuple(values)


def source_candidates(row: dict[str, str], dataset_name: str) -> list[tuple[str, ...]]:
    candidates = SOURCE_KEY_CANDIDATES.get(dataset_name, SOURCE_KEY_CANDIDATES["default"])
    fallback = SOURCE_KEY_CANDIDATES["default"]
    keys: list[tuple[str, ...]] = []
    for fields in candidates + [item for item in fallback if item not in candidates]:
        key = build_key(row, fields)
        if key is not None and key not in keys:
            keys.append(key)
    return keys


def raw_candidates(row: dict[str, str], dataset_name: str) -> list[tuple[str, ...]]:
    candidates = RAW_KEY_CANDIDATES.get(dataset_name, RAW_KEY_CANDIDATES["default"])
    fallback = RAW_KEY_CANDIDATES["default"]
    keys: list[tuple[str, ...]] = []
    for fields in candidates + [item for item in fallback if item not in candidates]:
        key = build_key(row, fields)
        if key is not None and key not in keys:
            keys.append(key)
    return keys


def load_source_rows(path: Path) -> list[dict[str, str]]:
    rows, _, _ = read_table(path)
    return rows


def collect_needed_keys(source_rows: list[dict[str, str]]) -> dict[str, set[tuple[str, ...]]]:
    needed: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for row in source_rows:
        dataset_name = get_value_case_insensitive(row, "source_dataset") or "default"
        for key in source_candidates(row, dataset_name):
            needed[dataset_name].add(key)
        for key in source_candidates(row, "default"):
            needed["default"].add(key)
    return needed


def build_fingerprint_index(
    dataset_name: str,
    path: Path,
    needed_keys: set[tuple[str, ...]],
) -> tuple[dict[tuple[str, ...], dict[str, str]], dict[str, Any], list[str]]:
    rows, headers, _ = read_table(path)
    fp_cols = fingerprint_columns(headers)
    index: dict[tuple[str, ...], dict[str, str]] = {}
    matched_rows = 0

    print(f"Loading fingerprints: {dataset_name}", flush=True)
    print(f"  rows={len(rows)} fingerprint_columns={len(fp_cols)}", flush=True)

    for row in rows:
        keys = raw_candidates(row, dataset_name)
        chosen_key = next((key for key in keys if key in needed_keys), None)
        if chosen_key is None or chosen_key in index:
            continue

        index[chosen_key] = {column: normalize(row.get(column)) for column in fp_cols}
        matched_rows += 1

    manifest = {
        "path": str(path),
        "rows": len(rows),
        "matched_rows": matched_rows,
        "fingerprint_columns": len(fp_cols),
        "matched_keys": len(index),
    }
    return index, manifest, fp_cols


def merge_rows(
    source_rows: list[dict[str, str]],
    dataset_indexes: dict[str, dict[tuple[str, ...], dict[str, str]]],
    fingerprint_columns_union: list[str],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    output_rows: list[dict[str, str]] = []
    stats = {
        "matched": 0,
        "unmatched": 0,
        "matched_by_dataset": defaultdict(int),
        "unmatched_row_ids": [],
    }

    dataset_order = list(DATASET_SPECS.keys())

    for index, row in enumerate(source_rows, start=1):
        if index == 1 or index % 2500 == 0:
            print(f"Merging source row {index}/{len(source_rows)}", flush=True)

        dataset_name = get_value_case_insensitive(row, "source_dataset")
        candidate_datasets = [dataset_name] if dataset_name in dataset_indexes else []
        candidate_datasets.extend([name for name in dataset_order if name != dataset_name])

        matched_dataset = None
        matched_row = None
        for current_dataset in candidate_datasets:
            indexes = dataset_indexes.get(current_dataset)
            if not indexes:
                continue
            for key in source_candidates(row, current_dataset):
                matched_row = indexes.get(key)
                if matched_row is not None:
                    matched_dataset = current_dataset
                    break
            if matched_row is not None:
                break

        if matched_row is None:
            stats["unmatched"] += 1
            stats["unmatched_row_ids"].append(get_value_case_insensitive(row, "row_id"))
            continue

        stats["matched"] += 1
        stats["matched_by_dataset"][matched_dataset or dataset_name or "default"] += 1

        output_row = {field: normalize(row.get(field)) for field in CORE_COLUMNS}
        for field, value in row.items():
            if field in OUTPUT_EXCLUDE_COLUMNS:
                continue
            output_row[field] = normalize(value)

        for column in fingerprint_columns_union:
            output_row[column] = matched_row.get(column, "")

        output_rows.append(output_row)

    stats["matched_by_dataset"] = dict(sorted(stats["matched_by_dataset"].items()))
    stats["total_rows"] = len(source_rows)
    return output_rows, stats


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final CCS dataset using fingerprints only.")
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_OUT)
    args = parser.parse_args()

    if not args.source_csv.exists():
        fallback_paths = [
            Path("data/model/final_covered_ccs_chebi.csv"),
            Path("data/model/final_covered_ccs_no_ontology.csv"),
            Path("data/model/final_covered_ccs.csv"),
        ]
        args.source_csv = next((path for path in fallback_paths if path.exists()), args.source_csv)

    if not args.source_csv.exists():
        raise FileNotFoundError(
            "Source CSV not found. Rebuild the covered CSV first with build_final_covered_dataset.py."
        )

    source_rows = load_source_rows(args.source_csv)
    needed_keys = collect_needed_keys(source_rows)

    dataset_indexes: dict[str, dict[tuple[str, ...], dict[str, str]]] = {}
    dataset_manifests: dict[str, Any] = {}
    fingerprint_columns_union: list[str] = []
    seen_fp_columns: set[str] = set()

    for dataset_name, path in DATASET_SPECS.items():
        dataset_needed = needed_keys.get(dataset_name, set()) | needed_keys.get("default", set())
        index, manifest, fp_cols = build_fingerprint_index(dataset_name, path, dataset_needed)
        dataset_indexes[dataset_name] = index
        dataset_manifests[dataset_name] = manifest

        for column in fp_cols:
            if column not in seen_fp_columns:
                fingerprint_columns_union.append(column)
                seen_fp_columns.add(column)

    output_rows, stats = merge_rows(source_rows, dataset_indexes, fingerprint_columns_union)

    fieldnames = list(CORE_COLUMNS)
    for row in source_rows:
        for field in row.keys():
            if field in OUTPUT_EXCLUDE_COLUMNS or field in fieldnames:
                continue
            fieldnames.append(field)
    for column in fingerprint_columns_union:
        if column not in fieldnames:
            fieldnames.append(column)

    write_csv(args.out_csv, output_rows, fieldnames)

    manifest = {
        "source_csv": str(args.source_csv),
        "out_csv": str(args.out_csv),
        "total_rows": stats["total_rows"],
        "matched_rows": stats["matched"],
        "unmatched_rows": stats["unmatched"],
        "matched_by_dataset": stats["matched_by_dataset"],
        "unmatched_row_ids": stats["unmatched_row_ids"],
        "fingerprint_columns": fingerprint_columns_union,
        "fingerprint_column_count": len(fingerprint_columns_union),
        "dataset_manifests": dataset_manifests,
    }

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_out.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print("Fingerprint dataset built")
    print(f"Source rows: {stats['total_rows']}")
    print(f"Matched rows: {stats['matched']}")
    print(f"Unmatched rows: {stats['unmatched']}")
    print(f"Fingerprint columns: {len(fingerprint_columns_union)}")
    print(f"Output CSV: {args.out_csv}")
    print(f"Manifest: {args.manifest_out}")


if __name__ == "__main__":
    main()