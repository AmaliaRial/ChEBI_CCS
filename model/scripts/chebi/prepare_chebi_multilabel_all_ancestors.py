#!/usr/bin/env python3
"""Build the complete ontology multilabel dataset from ChEBI chunk outputs.

This version keeps all available is_a ancestors, without any blacklist,
minimum count threshold, or depth cutoff.

The matched ChEBI ids come from the chunk outputs, and the ancestor labels
are resolved from the local chebi.obo hierarchy. The output columns are safe
for CSV and Python usage and the manifest stores the ChEBI id, name, count,
and the depth values where each label appears.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CHUNKS_DIR = Path("predictions/chebi/chunks")
DEFAULT_ONTOLOGY_OBO = Path("data/ontology/chebi.obo")
DEFAULT_FINGERPRINT_CSV = Path("data/model/final_covered_ccs_fingerprints.csv")
DEFAULT_OUTPUT_CSV = Path("data/model/final_covered_ccs_fingerprints_multilabel_all_ancestors.csv")
DEFAULT_MANIFEST_JSON = Path("data/model/ontology_label_manifest_all_ancestors.json")
REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the complete ontology multilabel dataset from ChEBI chunks."
    )
    parser.add_argument("--chunks-dir", type=Path, default=REPO_ROOT / DEFAULT_CHUNKS_DIR)
    parser.add_argument("--ontology-obo", type=Path, default=REPO_ROOT / DEFAULT_ONTOLOGY_OBO)
    parser.add_argument("--fingerprint-csv", type=Path, default=REPO_ROOT / DEFAULT_FINGERPRINT_CSV)
    parser.add_argument("--output-csv", type=Path, default=REPO_ROOT / DEFAULT_OUTPUT_CSV)
    parser.add_argument("--manifest-json", type=Path, default=REPO_ROOT / DEFAULT_MANIFEST_JSON)
    return parser.parse_args()


def load_chunk_files(chunks_dir: Path) -> list[Path]:
    chunk_files = sorted(chunks_dir.glob("results_pablo_hybrid_chunk*.json"))
    if not chunk_files:
        raise FileNotFoundError(f"No ChEBI chunk JSON files found in {chunks_dir}")
    return chunk_files


def load_chunk_matches(chunks_dir: Path) -> dict[int, str]:
    row_to_chebi_id: dict[int, str] = {}

    for chunk_file in load_chunk_files(chunks_dir):
        with chunk_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        for item in payload.get("results", []):
            row_id = item.get("row_id")
            match = item.get("match") or {}
            chebi_id = match.get("chebi_id")
            if row_id is None or not chebi_id:
                continue

            row_to_chebi_id[int(row_id)] = str(chebi_id)

    return row_to_chebi_id


def parse_obo_terms(obo_path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    if not obo_path.exists():
        raise FileNotFoundError(f"Ontology OBO file not found: {obo_path}")

    term_to_parents: dict[str, list[str]] = defaultdict(list)
    term_names: dict[str, str] = {}

    current_id: str | None = None
    current_name: str | None = None
    current_parents: list[str] = []
    in_term = False

    def flush_term() -> None:
        nonlocal current_id, current_name, current_parents, in_term
        if current_id is not None:
            term_names[current_id] = current_name or ""
            term_to_parents[current_id] = list(dict.fromkeys(current_parents))
        current_id = None
        current_name = None
        current_parents = []
        in_term = False

    with obo_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            if line == "[Term]":
                flush_term()
                in_term = True
                continue

            if not line:
                if in_term:
                    flush_term()
                continue

            if not in_term:
                continue

            if line.startswith("id: "):
                current_id = line[4:].strip()
                continue

            if line.startswith("name: "):
                current_name = line[6:].strip()
                continue

            if line.startswith("is_a: "):
                parent_id = line[6:].split("!", 1)[0].strip().split()[0]
                if parent_id:
                    current_parents.append(parent_id)

    flush_term()
    return term_to_parents, term_names


def load_fingerprint_dataset(fingerprint_csv: Path) -> pd.DataFrame:
    if not fingerprint_csv.exists():
        raise FileNotFoundError(f"Fingerprint CSV not found: {fingerprint_csv}")

    df = pd.read_csv(fingerprint_csv, low_memory=False)
    if "row_id" not in df.columns:
        raise ValueError("The fingerprint dataset does not contain a row_id column.")

    df = df.copy()
    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    if df["row_id"].duplicated().any():
        duplicate_ids = df.loc[df["row_id"].duplicated(), "row_id"].head(10).tolist()
        raise ValueError(f"Fingerprint dataset contains duplicated row_id values: {duplicate_ids}")

    return df


def sanitize_name(name: str) -> str:
    cleaned = name.lower().strip()
    cleaned = cleaned.replace("'", "")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def sanitize_chebi_id(chebi_id: str) -> str:
    return chebi_id.replace(":", "_")


def build_ancestor_assignments(
    row_to_chebi_id: dict[int, str],
    term_to_parents: dict[str, list[str]],
    term_names: dict[str, str],
) -> tuple[dict[int, set[str]], dict[str, str], dict[str, set[int]], Counter[str]]:
    """Collect all is_a ancestors for each matched molecule.

    The returned distances are the shortest distances from the matched term
    to each ancestor, measured in number of is_a hops.
    """
    row_to_classes: dict[int, set[str]] = defaultdict(set)
    class_names: dict[str, str] = {}
    class_depths: dict[str, set[int]] = defaultdict(set)

    def collect_ancestors(start_chebi_id: str) -> dict[str, int]:
        distances: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque()

        for parent_id in term_to_parents.get(start_chebi_id, []):
            queue.append((parent_id, 1))

        while queue:
            chebi_id, depth = queue.popleft()
            previous_depth = distances.get(chebi_id)
            if previous_depth is not None and previous_depth <= depth:
                continue

            distances[chebi_id] = depth
            for parent_id in term_to_parents.get(chebi_id, []):
                queue.append((parent_id, depth + 1))

        return distances

    for row_id, chebi_id in row_to_chebi_id.items():
        ancestor_distances = collect_ancestors(chebi_id)
        if not ancestor_distances:
            continue

        for ancestor_id, depth in ancestor_distances.items():
            row_to_classes[row_id].add(ancestor_id)
            class_depths[ancestor_id].add(depth)
            if ancestor_id not in class_names:
                class_names[ancestor_id] = term_names.get(ancestor_id, "")

    class_counts: Counter[str] = Counter()
    for classes in row_to_classes.values():
        class_counts.update(classes)

    return row_to_classes, class_names, class_depths, class_counts


def build_column_names(
    class_counts: Counter[str],
    class_names: dict[str, str],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Create safe and unique column names for each ChEBI class."""
    base_name_to_ids: dict[str, list[str]] = defaultdict(list)
    for chebi_id in class_counts:
        base_name = sanitize_name(class_names.get(chebi_id, ""))
        if not base_name:
            base_name = "chebi"
        base_name_to_ids[base_name].append(chebi_id)

    chebi_to_column: dict[str, str] = {}
    manifest_rows: list[dict[str, Any]] = []

    for base_name, chebi_ids in sorted(base_name_to_ids.items()):
        chebi_ids_sorted = sorted(chebi_ids)
        if len(chebi_ids_sorted) == 1 and base_name != "chebi":
            chebi_id = chebi_ids_sorted[0]
            column_name = f"ont_{base_name}"
            chebi_to_column[chebi_id] = column_name
        else:
            for chebi_id in chebi_ids_sorted:
                column_name = f"ont_{base_name}_chebi_{sanitize_chebi_id(chebi_id)}"
                chebi_to_column[chebi_id] = column_name

    for chebi_id, count in class_counts.items():
        column_name = chebi_to_column[chebi_id]
        manifest_rows.append(
            {
                "chebi_id": chebi_id,
                "name": class_names.get(chebi_id, ""),
                "count": int(count),
                "column": column_name,
            }
        )

    manifest_rows.sort(key=lambda item: (-item["count"], item["chebi_id"]))
    return chebi_to_column, manifest_rows


def build_ontology_matrix(
    fingerprint_df: pd.DataFrame,
    row_to_classes: dict[int, set[str]],
    chebi_to_column: dict[str, str],
) -> tuple[pd.DataFrame, int]:
    selected_columns = [column for _, column in sorted(chebi_to_column.items(), key=lambda item: item[1])]
    ontology_matrix = pd.DataFrame(0, index=fingerprint_df.index, columns=selected_columns, dtype="uint8")

    row_id_to_index = {int(row_id): index for index, row_id in zip(fingerprint_df.index, fingerprint_df["row_id"].tolist())}

    for row_id, classes in row_to_classes.items():
        index = row_id_to_index.get(row_id)
        if index is None:
            continue

        for chebi_id in classes:
            column_name = chebi_to_column.get(chebi_id)
            if column_name is not None:
                ontology_matrix.at[index, column_name] = 1

    rows_with_labels = int((ontology_matrix.sum(axis=1) > 0).sum())
    return ontology_matrix, rows_with_labels


def build_manifest(
    manifest_json: Path,
    fingerprint_csv: Path,
    chunks_dir: Path,
    ontology_obo: Path,
    output_csv: Path,
    fingerprint_rows: int,
    rows_with_ontology_labels: int,
    label_rows: list[dict[str, Any]],
    class_depths: dict[str, set[int]],
) -> None:
    manifest_labels: list[dict[str, Any]] = []
    for item in label_rows:
        chebi_id = item["chebi_id"]
        depths = sorted(class_depths.get(chebi_id, set()))
        manifest_labels.append(
            {
                "chebi_id": chebi_id,
                "name": item["name"],
                "count": item["count"],
                "column": item["column"],
                "depths": depths,
                "min_depth": depths[0] if depths else None,
                "max_depth": depths[-1] if depths else None,
                "depth_count": len(depths),
            }
        )

    manifest = {
        "fingerprint_csv": str(fingerprint_csv),
        "chunks_dir": str(chunks_dir),
        "ontology_obo": str(ontology_obo),
        "output_csv": str(output_csv),
        "fingerprint_rows": int(fingerprint_rows),
        "rows_with_ontology_labels": int(rows_with_ontology_labels),
        "ontology_label_count": int(len(manifest_labels)),
        "ontology_labels": manifest_labels,
        "top_50_ontology_labels": manifest_labels[:50],
        "notes": "All available is_a ancestors were kept. No blacklist, cutoff, or minimum class count was applied.",
    }

    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    with manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)


def main() -> None:
    args = parse_args()

    fingerprint_df = load_fingerprint_dataset(args.fingerprint_csv)
    row_to_chebi_id = load_chunk_matches(args.chunks_dir)
    term_to_parents, term_names = parse_obo_terms(args.ontology_obo)
    row_to_classes, class_names, class_depths, class_counts = build_ancestor_assignments(
        row_to_chebi_id=row_to_chebi_id,
        term_to_parents=term_to_parents,
        term_names=term_names,
    )

    if not class_counts:
        raise ValueError("No ontology classes were collected from the chunk data.")

    chebi_to_column, label_rows = build_column_names(class_counts, class_names)
    ontology_matrix, rows_with_labels = build_ontology_matrix(
        fingerprint_df=fingerprint_df,
        row_to_classes=row_to_classes,
        chebi_to_column=chebi_to_column,
    )

    output_df = pd.concat([fingerprint_df.reset_index(drop=True), ontology_matrix.reset_index(drop=True)], axis=1)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)

    build_manifest(
        manifest_json=args.manifest_json,
        fingerprint_csv=args.fingerprint_csv,
        chunks_dir=args.chunks_dir,
        ontology_obo=args.ontology_obo,
        output_csv=args.output_csv,
        fingerprint_rows=len(fingerprint_df),
        rows_with_ontology_labels=rows_with_labels,
        label_rows=label_rows,
        class_depths=class_depths,
    )

    top_50 = label_rows[:50]
    print(f"Number of rows: {len(fingerprint_df)}")
    print(f"Number of ontology labels created: {len(label_rows)}")
    print("Top 50 ontology labels by frequency:")
    for index, item in enumerate(top_50, start=1):
        name = item["name"] or "<no_name>"
        depths = ",".join(str(depth) for depth in sorted(class_depths.get(item["chebi_id"], set())))
        print(f"  {index:02d}. {item['column']} | {item['chebi_id']} | {name} | {item['count']} | depths={depths}")

    print(f"Output CSV: {args.output_csv}")
    print(f"Manifest JSON: {args.manifest_json}")


if __name__ == "__main__":
    main()