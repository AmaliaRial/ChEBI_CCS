#!/usr/bin/env python3
"""Rebuild the covered CCS dataset without ontology columns.

This script merges the six hybrid ChEBI chunk outputs back into the canonical
source CSV and keeps only rows that have a ChEBI match.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_CSV = Path("data/unified/unified_ccs.csv")
DEFAULT_RESULTS_DIR = Path("predictions/chebi/chunks")
DEFAULT_OUT = Path("data/model/final_covered_ccs.csv")
DEFAULT_MANIFEST_OUT = Path("data/model/final_covered_manifest.json")


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def load_matches(results_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    matches: dict[int, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []

    result_files = sorted(results_dir.glob("results_pablo_hybrid_chunk*.json"))
    if not result_files:
        raise FileNotFoundError(f"No hybrid JSON outputs found in {results_dir}")

    for result_file in result_files:
        with result_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        chunk_summary = payload.get("summary", {})
        summaries.append(
            {
                "file": result_file.name,
                "total_inputs": chunk_summary.get("total_inputs", 0),
                "matched": chunk_summary.get("matched", 0),
                "coverage_pct": chunk_summary.get("coverage_pct", 0),
                "matched_by_source": chunk_summary.get("matched_by_source", {}),
            }
        )

        for item in payload.get("results", []):
            match = item.get("match") or {}
            row_id = item.get("row_id")
            chebi_id = match.get("chebi_id")
            if row_id is None or not chebi_id:
                continue

            row_id_int = int(row_id)
            if row_id_int in matches:
                continue

            matches[row_id_int] = {
                "match": match,
                "classifications": item.get("classifications", []),
            }

    return matches, {"chunk_summaries": summaries}


def build_rows(source_rows: list[dict[str, str]], matches: dict[int, dict[str, Any]]) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []

    for row in source_rows:
        row_id = int(row["row_id"])
        matched = matches.get(row_id)
        if not matched:
            continue

        classifications = matched.get("classifications", [])
        chebi_classes = json.dumps([cls.get("id") for cls in classifications if cls.get("id")], ensure_ascii=True)
        match = matched.get("match", {})

        enriched = dict(row)
        enriched["chebi_classes"] = chebi_classes
        enriched["chebi_count"] = str(len([cls for cls in classifications if cls.get("id")]))
        enriched["chebi_name"] = str(match.get("name", "") or "")
        enriched["chebi_match_source"] = str(match.get("match_source", "") or "")
        output_rows.append(enriched)

    return output_rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the covered no-ontology CCS dataset.")
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_OUT)
    args = parser.parse_args()

    if not args.source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {args.source_csv}")
    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    source_rows, source_headers = read_csv(args.source_csv)
    matches, chunk_info = load_matches(args.results_dir)
    covered_rows = build_rows(source_rows, matches)

    fieldnames = list(source_headers)
    for column in ["chebi_classes", "chebi_count", "chebi_name", "chebi_match_source"]:
        if column not in fieldnames:
            fieldnames.append(column)

    write_csv(args.out, covered_rows, fieldnames)

    manifest = {
        "source_csv": str(args.source_csv),
        "results_dir": str(args.results_dir),
        "total_source_rows": len(source_rows),
        "covered_rows": len(covered_rows),
        "coverage_pct": round((len(covered_rows) / len(source_rows) * 100.0) if source_rows else 0.0, 2),
        "out": str(args.out),
        **chunk_info,
    }

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_out.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print("Rebuilt covered dataset")
    print(f"Source rows: {len(source_rows)}")
    print(f"Covered rows: {len(covered_rows)}")
    print(f"Coverage: {manifest['coverage_pct']}%")
    print(f"Output CSV: {args.out}")
    print(f"Manifest: {args.manifest_out}")


if __name__ == "__main__":
    main()