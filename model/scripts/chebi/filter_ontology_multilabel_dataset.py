#!/usr/bin/env python3
"""Filter the complete all-ancestors ontology multilabel dataset.

The input is the master multilabel dataset with all available is_a ancestors.
This script keeps all non-ontology columns unchanged and filters only the
ontology target columns (those starting with ``ont_``).

Filtering is configurable through command-line arguments and supports:
- minimum class count
- maximum class frequency ratio
- generic/blacklist exclusion
- optional blacklist file
- optional top-N selection after filtering
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_CSV = REPO_ROOT / "data" / "model" / "final_covered_ccs_fingerprints_multilabel_all_ancestors.csv"
DEFAULT_INPUT_MANIFEST = REPO_ROOT / "data" / "model" / "ontology_label_manifest_all_ancestors.json"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data" / "model" / "final_covered_ccs_fingerprints_multilabel_filtered.csv"
DEFAULT_OUTPUT_MANIFEST = REPO_ROOT / "data" / "model" / "ontology_label_manifest_filtered.json"

DEFAULT_GENERIC_BLACKLIST = {
    "chemical entity",
    "molecular entity",
    "organic molecular entity",
    "organic molecule",
    "molecule",
    "polyatomic entity",
    "main group molecular entity",
    "p-block molecular entity",
    "s-block molecular entity",
    "carbon group molecular entity",
    "oxygen molecular entity",
    "nitrogen molecular entity",
    "hydrogen molecular entity",
    "phosphorus molecular entity",
    "sulfur molecular entity",
    "chlorine molecular entity",
    "fluorine molecular entity",
    "chalcogen molecular entity",
    "pnictogen molecular entity",
    "heteroatomic molecular entity",
    "heteroorganic entity",
}


@dataclass
class LabelRecord:
    chebi_id: str
    name: str
    count: int
    column: str
    depths: list[int]
    min_depth: int | None
    max_depth: int | None
    depth_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter the complete ontology multilabel dataset.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--min-class-count", type=int, default=50)
    parser.add_argument("--max-class-frequency-ratio", type=float, default=0.80)
    parser.add_argument(
        "--exclude-generic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude generic ontology labels using the default blacklist.",
    )
    parser.add_argument(
        "--blacklist-file",
        type=Path,
        default=None,
        help="Optional TXT or JSON file with ontology column names, ChEBI IDs, or names to exclude.",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=None,
        help="If set, keep only the top N labels after filtering.",
    )
    return parser.parse_args()


def load_manifest(input_manifest: Path) -> dict[str, Any]:
    if not input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_manifest}")
    with input_manifest.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_labels_from_manifest(manifest: dict[str, Any]) -> list[LabelRecord]:
    labels: list[LabelRecord] = []
    for item in manifest.get("ontology_labels", []):
        depths = item.get("depths") or []
        labels.append(
            LabelRecord(
                chebi_id=str(item.get("chebi_id", "")),
                name=str(item.get("name", "")),
                count=int(item.get("count", 0)),
                column=str(item.get("column", "")),
                depths=[int(depth) for depth in depths],
                min_depth=item.get("min_depth"),
                max_depth=item.get("max_depth"),
                depth_count=int(item.get("depth_count", len(depths))),
            )
        )
    labels.sort(key=lambda item: (-item.count, item.chebi_id))
    return labels


def normalize_text(value: str) -> str:
    text = value.strip().lower()
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def normalize_chebi_id(value: str) -> str:
    return value.strip().upper().replace(" ", "")


def load_blacklist_values(blacklist_file: Path | None) -> set[str]:
    if blacklist_file is None:
        return set()
    if not blacklist_file.exists():
        raise FileNotFoundError(f"Blacklist file not found: {blacklist_file}")

    values: set[str] = set()
    suffix = blacklist_file.suffix.lower()

    if suffix == ".json":
        with blacklist_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        def collect(obj: Any) -> None:
            if isinstance(obj, str):
                stripped = obj.strip()
                if stripped:
                    values.add(stripped)
            elif isinstance(obj, dict):
                for nested in obj.values():
                    collect(nested)
            elif isinstance(obj, list):
                for nested in obj:
                    collect(nested)

        collect(payload)
    else:
        with blacklist_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    values.add(stripped)

    return values


def label_matches_blacklist(label: LabelRecord, blacklist_values: set[str]) -> bool:
    if not blacklist_values:
        return False

    candidates = {
        label.column,
        label.chebi_id,
        label.name,
        normalize_text(label.column),
        normalize_text(label.chebi_id),
        normalize_text(label.name),
        normalize_chebi_id(label.chebi_id),
    }
    return any(candidate in blacklist_values for candidate in candidates)


def load_input_dataset(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    return pd.read_csv(input_csv, low_memory=False)


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    ontology_columns = [column for column in df.columns if column.startswith("ont_")]
    base_columns = [column for column in df.columns if not column.startswith("ont_")]
    return base_columns, ontology_columns


def filter_labels(
    labels: list[LabelRecord],
    total_rows: int,
    min_class_count: int,
    max_class_frequency_ratio: float,
    exclude_generic: bool,
    blacklist_values: set[str],
    max_labels: int | None,
) -> tuple[list[LabelRecord], list[dict[str, Any]], dict[str, int]]:
    removed: list[dict[str, Any]] = []
    selected: list[LabelRecord] = []
    summary = {
        "removed_count_below_min_class_count": 0,
        "removed_frequency_above_max_ratio": 0,
        "removed_generic_or_blacklisted": 0,
        "removed_by_max_labels": 0,
    }

    generic_blacklist = {normalize_text(value) for value in DEFAULT_GENERIC_BLACKLIST}
    frequency_threshold = float(max_class_frequency_ratio)

    for label in labels:
        reason = None
        reasons: list[str] = []

        if label.count < min_class_count:
            reason = "count_below_min_class_count"
            reasons.append(reason)
            summary["removed_count_below_min_class_count"] += 1
        elif total_rows > 0 and (label.count / total_rows) > frequency_threshold:
            reason = "frequency_above_max_class_frequency_ratio"
            reasons.append(reason)
            summary["removed_frequency_above_max_ratio"] += 1
        else:
            if exclude_generic:
                normalized_name = normalize_text(label.name)
                normalized_column = normalize_text(label.column)
                if normalized_name in generic_blacklist or normalized_column in generic_blacklist:
                    reason = "generic_blacklisted"
                    reasons.append(reason)
                    summary["removed_generic_or_blacklisted"] += 1
            if reason is None and blacklist_values and label_matches_blacklist(label, blacklist_values):
                reason = "blacklisted"
                reasons.append(reason)
                summary["removed_generic_or_blacklisted"] += 1

        if reason is None:
            selected.append(label)
        else:
            removed.append(
                {
                    "chebi_id": label.chebi_id,
                    "name": label.name,
                    "count": label.count,
                    "column": label.column,
                    "depths": label.depths,
                    "reason": reason,
                    "reasons": reasons,
                }
            )

    if max_labels is not None and len(selected) > max_labels:
        overflow = selected[max_labels:]
        selected = selected[:max_labels]
        for label in overflow:
            removed.append(
                {
                    "chebi_id": label.chebi_id,
                    "name": label.name,
                    "count": label.count,
                    "column": label.column,
                    "depths": label.depths,
                    "reason": "max_labels_limit",
                    "reasons": ["max_labels_limit"],
                }
            )
        summary["removed_by_max_labels"] = len(overflow)

    removed.sort(key=lambda item: (-item["count"], item["chebi_id"]))
    return selected, removed, summary


def build_filtered_dataset(
    df: pd.DataFrame,
    selected_labels: list[LabelRecord],
) -> pd.DataFrame:
    base_columns, ontology_columns = split_columns(df)
    selected_columns = [label.column for label in selected_labels if label.column in ontology_columns]
    keep_columns = base_columns + selected_columns
    return df.loc[:, keep_columns].copy()


def save_manifest(
    output_manifest: Path,
    input_manifest_path: Path,
    input_manifest: dict[str, Any],
    selected_labels: list[LabelRecord],
    removed_labels: list[dict[str, Any]],
    filtering_parameters: dict[str, Any],
    total_rows: int,
) -> None:
    manifest = {
        "input_manifest": str(input_manifest_path),
        "filtering_parameters": filtering_parameters,
        "number_of_rows": int(total_rows),
        "number_of_selected_labels": int(len(selected_labels)),
        "selected_labels": [asdict(label) for label in selected_labels],
        "removed_labels": removed_labels,
        "original_ontology_label_count": int(input_manifest.get("ontology_label_count", len(input_manifest.get("ontology_labels", [])))),
    }

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)


def print_summary(
    labels: list[LabelRecord],
    selected_labels: list[LabelRecord],
    summary: dict[str, int],
    output_csv: Path,
    output_manifest: Path,
) -> None:
    print(f"Number of original ontology labels: {len(labels)}")
    print(f"Number removed because count < min_class_count: {summary['removed_count_below_min_class_count']}")
    print(
        "Number removed because frequency ratio > max_class_frequency_ratio: "
        f"{summary['removed_frequency_above_max_ratio']}"
    )
    print(f"Number removed because generic/blacklisted: {summary['removed_generic_or_blacklisted']}")
    if summary.get("removed_by_max_labels", 0):
        print(f"Number removed because max_labels limit: {summary['removed_by_max_labels']}")
    print(f"Number of final selected ontology labels: {len(selected_labels)}")
    print("Top 50 selected labels by count:")
    for index, label in enumerate(selected_labels[:50], start=1):
        depths = ",".join(str(depth) for depth in label.depths)
        print(f"  {index:02d}. {label.column} | {label.chebi_id} | {label.name} | {label.count} | depths={depths}")
    print(f"Output CSV: {output_csv}")
    print(f"Output manifest: {output_manifest}")


def main() -> None:
    args = parse_args()

    input_manifest = load_manifest(args.input_manifest)
    labels = load_labels_from_manifest(input_manifest)
    input_df = load_input_dataset(args.input_csv)
    total_rows = len(input_df)

    blacklist_values = load_blacklist_values(args.blacklist_file)
    selected_labels, removed_labels, summary = filter_labels(
        labels=labels,
        total_rows=total_rows,
        min_class_count=args.min_class_count,
        max_class_frequency_ratio=args.max_class_frequency_ratio,
        exclude_generic=args.exclude_generic,
        blacklist_values=blacklist_values,
        max_labels=args.max_labels,
    )

    filtered_df = build_filtered_dataset(input_df, selected_labels)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(args.output_csv, index=False)

    save_manifest(
        output_manifest=args.output_manifest,
        input_manifest_path=args.input_manifest,
        input_manifest=input_manifest,
        selected_labels=selected_labels,
        removed_labels=removed_labels,
        filtering_parameters={
            "input_csv": str(args.input_csv),
            "input_manifest": str(args.input_manifest),
            "min_class_count": args.min_class_count,
            "max_class_frequency_ratio": args.max_class_frequency_ratio,
            "exclude_generic": bool(args.exclude_generic),
            "blacklist_file": str(args.blacklist_file) if args.blacklist_file else None,
            "max_labels": args.max_labels,
            "selected_ontology_columns": [label.column for label in selected_labels],
        },
        total_rows=total_rows,
    )

    print_summary(labels, selected_labels, summary, args.output_csv, args.output_manifest)


if __name__ == "__main__":
    main()