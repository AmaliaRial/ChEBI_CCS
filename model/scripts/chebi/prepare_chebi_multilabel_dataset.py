#!/usr/bin/env python3
"""Build the ontology multilabel dataset from ChEBI chunk outputs.

This script uses the chunk matches as a bridge to get matched CHEBI ids,
then resolves the full ancestor hierarchy from the local chebi.obo file
by traversing is_a relationships up to depth 4, excluding generic terms.

Each matched molecule gets labeled with all its meaningful ancestors,
capturing terms like "lipid", "steroid", "phospholipid" etc.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CHUNKS_DIR = Path("predictions/chebi/chunks")
DEFAULT_ONTOLOGY_OBO = Path("data/ontology/chebi.obo")
DEFAULT_FINGERPRINT_CSV = Path("data/model/final_covered_ccs_fingerprints.csv")
DEFAULT_OUTPUT_CSV = Path("data/model/final_covered_ccs_fingerprints_multilabel.csv")
DEFAULT_MANIFEST_JSON = Path("data/model/ontology_label_columns.json")
DEFAULT_MIN_CLASS_COUNT = 30
REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the ontology multilabel dataset from ChEBI chunks.")
    parser.add_argument("--chunks-dir", type=Path, default=REPO_ROOT / DEFAULT_CHUNKS_DIR)
    parser.add_argument("--ontology-obo", type=Path, default=REPO_ROOT / DEFAULT_ONTOLOGY_OBO)
    parser.add_argument("--fingerprint-csv", type=Path, default=REPO_ROOT / DEFAULT_FINGERPRINT_CSV)
    parser.add_argument("--output-csv", type=Path, default=REPO_ROOT / DEFAULT_OUTPUT_CSV)
    parser.add_argument("--manifest-json", type=Path, default=REPO_ROOT / DEFAULT_MANIFEST_JSON)
    parser.add_argument("--min-class-count", type=int, default=DEFAULT_MIN_CLASS_COUNT)
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


def sanitize_chebi_id(chebi_id: str) -> str:
    return chebi_id.replace(":", "_")


def build_ontology_assignments(
    row_to_chebi_id: dict[int, str],
    term_to_parents: dict[str, list[str]],
    term_names: dict[str, str],
) -> tuple[dict[int, set[str]], dict[str, str], Counter[str]]:
    """Build ontology assignments by traversing ancestor hierarchy up to depth 4.
    
    For each matched molecule, we traverse upward through is_a relationships
    to collect meaningful ancestors, excluding ultra-generic terms.
    """
    # Terms too generic to be useful as labels
    GENERIC_BLACKLIST = {
        "CHEBI:23367",  # molecular entity
        "CHEBI:24431",  # chemical entity
        "CHEBI:50860",  # organic molecular entity
        "CHEBI:36357",  # polyatomic entity
        "CHEBI:36359",  # phosphorus oxoacid derivative
    }
    
    row_to_classes: dict[int, set[str]] = defaultdict(set)
    class_names: dict[str, str] = {}

    def collect_ancestors(chebi_id: str, visited: set[str] | None = None, depth: int = 0, max_depth: int = 4) -> set[str]:
        """Recursively collect ancestors up to max_depth, excluding blacklisted terms."""
        if visited is None:
            visited = set()
        
        if chebi_id in visited or depth > max_depth:
            return set()
        
        visited.add(chebi_id)
        ancestors = set()
        
        # Add this term if not blacklisted
        if chebi_id not in GENERIC_BLACKLIST:
            ancestors.add(chebi_id)
        
        # Traverse to parent terms
        parents = term_to_parents.get(chebi_id, [])
        for parent_id in parents:
            if parent_id not in visited:
                ancestors.update(collect_ancestors(parent_id, visited, depth + 1, max_depth))
        
        return ancestors

    for row_id, chebi_id in row_to_chebi_id.items():
        ancestors = collect_ancestors(chebi_id)
        
        for ancestor_id in ancestors:
            row_to_classes[row_id].add(ancestor_id)
            if ancestor_id not in class_names:
                class_names[ancestor_id] = term_names.get(ancestor_id, "")

    class_counts: Counter[str] = Counter()
    for classes in row_to_classes.values():
        class_counts.update(classes)

    return row_to_classes, class_names, class_counts


def build_selected_labels(
    class_counts: Counter[str],
    class_names: dict[str, str],
    min_class_count: int,
) -> list[dict[str, Any]]:
    selected = []
    for chebi_id, count in class_counts.items():
        if count >= min_class_count:
            selected.append(
                {
                    "chebi_id": chebi_id,
                    "name": class_names.get(chebi_id, ""),
                    "count": int(count),
                    "column": f"ont_{sanitize_chebi_id(chebi_id)}",
                }
            )

    selected.sort(key=lambda item: (-item["count"], item["chebi_id"]))
    return selected


def build_top_class_summary(
    class_counts: Counter[str],
    class_names: dict[str, str],
    top_n: int = 20,
) -> list[dict[str, Any]]:
    summary = []
    for chebi_id, count in sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]:
        summary.append(
            {
                "chebi_id": chebi_id,
                "name": class_names.get(chebi_id, ""),
                "count": int(count),
            }
        )
    return summary


def build_ontology_matrix(
    fingerprint_df: pd.DataFrame,
    row_to_classes: dict[int, set[str]],
    selected_labels: list[dict[str, Any]],
) -> tuple[pd.DataFrame, int]:
    selected_columns = [item["column"] for item in selected_labels]
    ontology_matrix = pd.DataFrame(0, index=fingerprint_df.index, columns=selected_columns, dtype="uint8")

    row_id_to_index = {int(row_id): index for index, row_id in zip(fingerprint_df.index, fingerprint_df["row_id"].tolist())}
    selected_by_chebi_id = {item["chebi_id"]: item["column"] for item in selected_labels}

    for row_id, classes in row_to_classes.items():
        index = row_id_to_index.get(row_id)
        if index is None:
            continue

        selected_for_row = [selected_by_chebi_id[chebi_id] for chebi_id in classes if chebi_id in selected_by_chebi_id]
        if not selected_for_row:
            continue

        ontology_matrix.loc[index, selected_for_row] = 1

    rows_with_labels = int((ontology_matrix.sum(axis=1) > 0).sum())
    return ontology_matrix, rows_with_labels


def save_manifest(
    manifest_json: Path,
    fingerprint_csv: Path,
    chunks_dir: Path,
    ontology_obo: Path,
    output_csv: Path,
    min_class_count: int,
    fingerprint_rows: int,
    rows_with_ontology_labels: int,
    selected_labels: list[dict[str, Any]],
    top_classes: list[dict[str, Any]],
) -> None:
    manifest = {
        "fingerprint_csv": str(fingerprint_csv),
        "chunks_dir": str(chunks_dir),
        "ontology_obo": str(ontology_obo),
        "output_csv": str(output_csv),
        "min_class_count": int(min_class_count),
        "fingerprint_rows": int(fingerprint_rows),
        "rows_with_ontology_labels": int(rows_with_ontology_labels),
        "selected_ontology_class_count": int(len(selected_labels)),
        "selected_ontology_labels": selected_labels,
        "top_20_ontology_classes": top_classes,
    }

    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    with manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)


def main() -> None:
    args = parse_args()

    fingerprint_df = load_fingerprint_dataset(args.fingerprint_csv)
    row_to_chebi_id = load_chunk_matches(args.chunks_dir)
    term_to_parents, term_names = parse_obo_terms(args.ontology_obo)
    row_to_classes, class_names, class_counts = build_ontology_assignments(
        row_to_chebi_id,
        term_to_parents,
        term_names,
    )

    selected_labels = build_selected_labels(class_counts, class_names, args.min_class_count)
    if not selected_labels:
        raise ValueError(
            f"No ontology classes meet the minimum count threshold of {args.min_class_count}."
        )

    ontology_matrix, rows_with_labels = build_ontology_matrix(fingerprint_df, row_to_classes, selected_labels)
    output_df = pd.concat([fingerprint_df.reset_index(drop=True), ontology_matrix.reset_index(drop=True)], axis=1)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)

    top_classes = build_top_class_summary(class_counts, class_names, top_n=20)
    save_manifest(
        manifest_json=args.manifest_json,
        fingerprint_csv=args.fingerprint_csv,
        chunks_dir=args.chunks_dir,
        ontology_obo=args.ontology_obo,
        output_csv=args.output_csv,
        min_class_count=args.min_class_count,
        fingerprint_rows=len(fingerprint_df),
        rows_with_ontology_labels=rows_with_labels,
        selected_labels=selected_labels,
        top_classes=top_classes,
    )

    print(f"Number of rows in fingerprint dataset: {len(fingerprint_df)}")
    print(f"Number of rows with ontology labels: {rows_with_labels}")
    print(f"Number of selected ontology classes: {len(selected_labels)}")
    print("Top 20 ontology classes by frequency:")
    for index, item in enumerate(top_classes, start=1):
        name = item["name"] or "<no_name>"
        print(f"  {index:02d}. {item['chebi_id']} | {name} | {item['count']}")
    print(f"Output file path: {args.output_csv}")
    print(f"Manifest file path: {args.manifest_json}")


if __name__ == "__main__":
    main()