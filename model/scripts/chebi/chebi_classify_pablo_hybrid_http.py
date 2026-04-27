#!/usr/bin/env python3
"""Hybrid ChEBI classifier over the public HTTP API.

This script keeps the Pablo HTTP workflow, but expands the input handling so it
can classify records that provide SMILES, InChI, or InChIKey. When SMILES are
present, it also tries a canonical SMILES and an InChI derived from that SMILES
before giving up.

The script is intended for smaller validation runs first because the public API
is slow and rate-limited in practice. Use --sample-size or --limit to test a
reduced subset before launching a full 68k run.

Input formats:
- JSONL records with row_id, smiles, inchi, and/or inchikey fields
- plain text files with one value per line
- a single value passed directly on the command line

Output:
- JSON payload with summary and per-row results
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import requests
from rdkit import Chem
from rdkit import RDLogger


DEFAULT_OBO = "data/ontology/chebi.obo"
DEFAULT_OUT = "predictions/chebi/results_pablo_hybrid.json"
DEFAULT_TIMEOUT = 30

INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")

RDLogger.DisableLog("rdApp.*")


def canonical_smiles(smiles: str | None) -> str | None:
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def inchi_from_smiles(smiles: str | None) -> str | None:
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        return Chem.MolToInchi(mol).strip()
    except Exception:
        return None


def infer_input_type(value: str) -> str:
    text = value.lstrip("\ufeff").strip()
    if text.startswith("InChI="):
        return "inchi"
    if INCHIKEY_RE.match(text):
        return "inchikey"
    return "smiles"


def parse_obo(obo_path: str) -> tuple[
    dict[str, list[str]],
    dict[str, str],
    dict[str, list[str]],
    dict[str, list[str]],
]:
    parents: dict[str, list[str]] = {}
    names: dict[str, str] = {}
    smiles_map: dict[str, list[str]] = defaultdict(list)
    inchi_map: dict[str, list[str]] = defaultdict(list)
    current_id: str | None = None
    current_smiles: str | None = None
    current_inchi: str | None = None

    def commit_current() -> None:
        if not current_id:
            return
        if current_smiles:
            smiles_map[current_smiles].append(current_id)
            canonical = canonical_smiles(current_smiles)
            if canonical:
                smiles_map[canonical].append(current_id)
        if current_inchi:
            inchi_map[current_inchi].append(current_id)

    with open(obo_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line == "[Term]":
                commit_current()
                current_id = None
                current_smiles = None
                current_inchi = None
                continue
            if not line:
                continue
            if line.startswith("id: CHEBI:"):
                current_id = line.split("id: ", 1)[1].strip()
                parents.setdefault(current_id, [])
                continue
            if current_id is None:
                continue
            if line.startswith("name: "):
                names[current_id] = line.split("name: ", 1)[1].strip()
                continue
            if line.startswith("is_a: CHEBI:"):
                parent = line.split("is_a: ", 1)[1].split(" ! ", 1)[0].strip()
                parents[current_id].append(parent)
                continue

            if line.startswith('property_value: "http://purl.obolibrary.org/obo/chebi/SMILES"'):
                parts = line.split('"')
                if len(parts) >= 4:
                    current_smiles = parts[3]
                continue

            if line.startswith('property_value: chemrof:smiles_string '):
                parts = line.split('"')
                if len(parts) >= 2:
                    current_smiles = parts[1]
                continue

            if line.startswith('property_value: "http://purl.obolibrary.org/obo/chebi/InChI"'):
                parts = line.split('"')
                if len(parts) >= 4:
                    current_inchi = parts[3]
                continue

            if line.startswith('property_value: chemrof:inchi_string '):
                parts = line.split('"')
                if len(parts) >= 2:
                    current_inchi = parts[1]
                continue

    commit_current()
    return parents, names, dict(smiles_map), dict(inchi_map)


def collect_ancestors(start_id: str, parents: dict[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    queue = deque(parents.get(start_id, []))
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        for parent in parents.get(current, []):
            if parent not in seen:
                queue.append(parent)
    return seen


def write_output(path: str, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    matched = sum(1 for item in results if item.get("match", {}).get("chebi_id"))
    unmatched = total - matched
    coverage_pct = (matched / total * 100.0) if total else 0.0

    matched_by_source: dict[str, int] = {}
    for item in results:
        source = item.get("match", {}).get("match_source")
        if source:
            matched_by_source[source] = matched_by_source.get(source, 0) + 1

    return {
        "total_inputs": total,
        "matched": matched,
        "unmatched": unmatched,
        "coverage_pct": round(coverage_pct, 2),
        "matched_by_source": dict(sorted(matched_by_source.items())),
    }


def fetch_chebi_match(session: requests.Session, value: str, value_type: str) -> dict[str, Any] | None:
    url = "https://www.ebi.ac.uk/chebi/backend/api/public/advanced_search/"
    body = {
        "text_search_specification": {
            "and_specification": [
                {
                    "text": value,
                    "category": value_type,
                }
            ]
        }
    }

    response = session.post(url, json=body, timeout=DEFAULT_TIMEOUT)
    if response.status_code != 200:
        raise RuntimeError(f"ChEBI API error: HTTP {response.status_code} - {response.text[:200]}")

    data = response.json()
    results = data.get("results", [])
    if not results:
        return None

    best = results[0].get("_source", {})
    chebi_accession = best.get("chebi_accession")
    if not chebi_accession:
        return None

    return {
        "chebi_accession": chebi_accession,
        "name": best.get("name"),
        "inchi": best.get("inchi"),
        "inchikey": best.get("inchikey"),
        "smiles": best.get("smiles"),
    }


def load_inputs(input_arg: str) -> list[dict[str, Any]]:
    path = Path(input_arg)
    if not path.exists():
        value_type = infer_input_type(input_arg)
        record: dict[str, Any] = {"row_id": 1, "smiles": None, "inchi": None, "inchikey": None}
        record[value_type] = input_arg
        return [record]

    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for row_number, raw_line in enumerate(handle, start=1):
            line = raw_line.lstrip("\ufeff").strip()
            if not line:
                continue
            if line.startswith("{"):
                row = json.loads(line)
                records.append(
                    {
                        "row_id": row.get("row_id", row_number),
                        "smiles": row.get("smiles"),
                        "inchi": row.get("inchi"),
                        "inchikey": row.get("inchikey"),
                    }
                )
            else:
                value_type = infer_input_type(line)
                record = {"row_id": row_number, "smiles": None, "inchi": None, "inchikey": None}
                record[value_type] = line
                records.append(record)

    return records


def select_records(records: list[dict[str, Any]], limit: int | None, sample_size: int | None, seed: int) -> list[dict[str, Any]]:
    selected = list(records)
    if sample_size is not None and sample_size > 0 and sample_size < len(selected):
        rng = random.Random(seed)
        selected = rng.sample(selected, sample_size)
        selected.sort(key=lambda item: int(item.get("row_id", 0)))

    if limit is not None and limit >= 0:
        selected = selected[:limit]

    return selected


def build_candidates(row: dict[str, Any]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add_candidate(value_type: str, value: str | None, source: str) -> None:
        if not isinstance(value, str):
            return
        text = value.strip()
        if not text:
            return
        key = (value_type, text)
        if key in seen:
            return
        seen.add(key)
        candidates.append({"value_type": value_type, "value": text, "source": source})

    add_candidate("inchi", row.get("inchi"), "input_inchi")

    smiles = row.get("smiles")
    if isinstance(smiles, str) and smiles.strip():
        smiles_text = smiles.strip()
        add_candidate("smiles", smiles_text, "input_smiles")

        canonical = canonical_smiles(smiles_text)
        if canonical:
            add_candidate("smiles", canonical, "canonical_smiles")

        derived_inchi = inchi_from_smiles(smiles_text)
        if derived_inchi:
            add_candidate("inchi", derived_inchi, "inchi_from_smiles")

    return candidates


def local_lookup_candidate(
    candidate: dict[str, str],
    smiles_map: dict[str, list[str]],
    inchi_map: dict[str, list[str]],
) -> str | None:
    value_type = candidate["value_type"]
    value = candidate["value"]

    if value_type == "inchi":
        ids = inchi_map.get(value, [])
        return ids[0] if ids else None

    if value_type == "smiles":
        ids = smiles_map.get(value, [])
        return ids[0] if ids else None

    return None


def classify_one(
    row: dict[str, Any],
    session: requests.Session,
    parents: dict[str, list[str]],
    names: dict[str, str],
    smiles_map: dict[str, list[str]],
    inchi_map: dict[str, list[str]],
) -> dict[str, Any]:
    row_id = row.get("row_id")
    smiles = row.get("smiles")
    inchi = row.get("inchi")
    inchikey = row.get("inchikey")

    normalized_row = {
        "smiles": smiles.strip() if isinstance(smiles, str) and smiles.strip() else None,
        "inchi": inchi.strip() if isinstance(inchi, str) and inchi.strip() else None,
        "inchikey": inchikey.strip() if isinstance(inchikey, str) and inchikey.strip() else None,
    }
    candidates = build_candidates(normalized_row)
    attempts: list[dict[str, Any]] = []
    # First pass: local OBO matching to avoid unnecessary API calls.
    for candidate in candidates:
        chebi_id = local_lookup_candidate(candidate, smiles_map, inchi_map)
        if not chebi_id:
            attempts.append(
                {
                    "value_type": candidate["value_type"],
                    "value": candidate["value"],
                    "source": candidate["source"],
                    "backend": "local_obo",
                    "match": False,
                }
            )
            continue

        ancestors = collect_ancestors(chebi_id, parents)
        classifications = [{"id": cid, "name": names.get(cid)} for cid in sorted(ancestors)]
        attempts.append(
            {
                "value_type": candidate["value_type"],
                "value": candidate["value"],
                "source": candidate["source"],
                "backend": "local_obo",
                "match": True,
            }
        )

        return {
            "row_id": row_id,
            "input": normalized_row,
            "match": {
                "chebi_id": chebi_id,
                "name": names.get(chebi_id),
                "smiles": normalized_row.get("smiles"),
                "inchi": normalized_row.get("inchi"),
                "inchikey": normalized_row.get("inchikey"),
                "match_source": candidate["source"],
                "query_type": candidate["value_type"],
                "query_value": candidate["value"],
                "match_backend": "local_obo",
            },
            "classifications": classifications,
            "attempts": attempts,
        }

    # Second pass: web API only for rows not matched locally.
    for candidate in candidates:
        try:
            match = fetch_chebi_match(session, candidate["value"], candidate["value_type"])
        except Exception as exc:
            attempts.append(
                {
                    "value_type": candidate["value_type"],
                    "value": candidate["value"],
                    "source": candidate["source"],
                    "backend": "web_api",
                    "error": str(exc),
                }
            )
            continue

        if not match:
            attempts.append(
                {
                    "value_type": candidate["value_type"],
                    "value": candidate["value"],
                    "source": candidate["source"],
                    "backend": "web_api",
                    "match": False,
                }
            )
            continue

        chebi_id = match["chebi_accession"]
        ancestors = collect_ancestors(chebi_id, parents)
        classifications = [{"id": cid, "name": names.get(cid)} for cid in sorted(ancestors)]

        return {
            "row_id": row_id,
            "input": normalized_row,
            "match": {
                "chebi_id": chebi_id,
                "name": match.get("name"),
                "smiles": match.get("smiles"),
                "inchi": match.get("inchi"),
                "inchikey": match.get("inchikey"),
                "match_source": candidate["source"],
                "query_type": candidate["value_type"],
                "query_value": candidate["value"],
                "match_backend": "web_api",
            },
            "classifications": classifications,
            "attempts": attempts,
        }

    return {
        "row_id": row_id,
        "input": normalized_row,
        "error": "No ChEBI match found",
        "attempts": attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid ChEBI classification over the public HTTP API."
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "JSONL with smiles/inchi/inchikey fields, a plain text file with one "
            "value per line, or a single value passed directly."
        ),
    )
    parser.add_argument("--obo", default=DEFAULT_OBO, help="Path to chebi.obo")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSON file")
    parser.add_argument(
        "--limit",
        type=int,
        help="Keep only the first N records after optional sampling.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Randomly sample N records before classification.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --sample-size.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.obo):
        payload = {
            "error": (
                "chebi.obo not found. Download 'chebi full' from "
                "https://www.ebi.ac.uk/chebi/downloads"
            )
        }
        write_output(args.out, payload)
        print(payload["error"])
        sys.exit(1)

    rows = load_inputs(args.input)
    rows = select_records(rows, args.limit, args.sample_size, args.seed)

    parents, names, smiles_map, inchi_map = parse_obo(args.obo)
    results: list[dict[str, Any]] = []

    session = requests.Session()
    total_rows = len(rows)

    try:
        for index, row in enumerate(rows, start=1):
            if index == 1 or index % 250 == 0:
                print(f"Progress: {index}/{total_rows}", flush=True)
            results.append(classify_one(row, session, parents, names, smiles_map, inchi_map))
    finally:
        session.close()

    summary = build_summary(results)
    payload = {
        "summary": summary,
        "results": results,
        "input_count": total_rows,
    }
    write_output(args.out, payload)

    print("=== Hybrid ChEBI Classification Summary ===")
    print(f"Total inputs: {summary['total_inputs']}")
    print(f"Matched: {summary['matched']}")
    print(f"Unmatched: {summary['unmatched']}")
    print(f"Coverage: {summary['coverage_pct']}%")
    print(f"Matched by source: {json.dumps(summary['matched_by_source'], sort_keys=True)}")
    print(f"Output JSON: {args.out}")

    if len(rows) == 1:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
