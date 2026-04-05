#!/usr/bin/env python3
""" 1. pipeline prepara datos (compounds.jsonl)
    2. pipeline llama a local
    3. local clasifica con chebi.obo local (no como el script de pablo que hace una llamada ala web por cada linea y hay 68k lineas :/)
    4. pipeline mergea resultados y guarda unified_ccs_chebi.csv
    
    conda run -n tfg_amalia python model/scripts/chebi_classify_pipeline.py"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def ensure_row_id(df: pd.DataFrame) -> pd.DataFrame:
    if "row_id" in df.columns:
        return df
    df = df.copy()
    df["row_id"] = pd.Series(df.index, index=df.index) + 1
    return df


def write_compounds_jsonl(df: pd.DataFrame, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in df.itertuples(index=False):
            payload = {
                "row_id": int(row.row_id),
                "smiles": row.smiles if pd.notna(row.smiles) else None,
                "inchi": row.inchi if hasattr(row, "inchi") and pd.notna(row.inchi) else None,
            }
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")


def run_local_classifier(input_jsonl: str, obo_file: str, out_json: str) -> None:
    script = Path(__file__).parent / "chebi_classify.py"
    cmd = [
        sys.executable,
        str(script),
        "--input",
        input_jsonl,
        "--obo",
        obo_file,
        "--out",
        out_json,
    ]
    subprocess.run(cmd, check=True)


def parse_results(result_json: str) -> tuple[dict[int, list[str]], dict[int, dict]]:
    with open(result_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    row_to_classes: dict[int, list[str]] = {}
    row_to_match: dict[int, dict] = {}

    for item in data.get("results", []):
        if "match" not in item:
            continue
        row_id = item.get("row_id")
        if row_id is None:
            continue

        match = item["match"]
        chebi_id = match.get("chebi_id")
        if not chebi_id:
            continue

        classes = [cls.get("id") for cls in item.get("classifications", []) if cls.get("id")]
        classes.append(chebi_id)
        row_to_classes[int(row_id)] = sorted(set(classes))
        row_to_match[int(row_id)] = match

    return row_to_classes, row_to_match


def enrich_dataframe(df: pd.DataFrame, row_to_classes: dict[int, list[str]], row_to_match: dict[int, dict]) -> pd.DataFrame:
    out = df.copy()
    out["chebi_classes"] = out["row_id"].apply(lambda rid: json.dumps(row_to_classes.get(int(rid), [])))
    out["chebi_count"] = out["row_id"].apply(lambda rid: len(row_to_classes.get(int(rid), [])))
    out["chebi_name"] = out["row_id"].apply(lambda rid: row_to_match.get(int(rid), {}).get("name", ""))
    out["chebi_match_source"] = out["row_id"].apply(lambda rid: row_to_match.get(int(rid), {}).get("match_source", ""))
    return out


def main() -> None:
    input_csv = Path("data/unified/unified_ccs.csv")
    obo_file = Path("data/ontology/chebi.obo")
    compounds_jsonl = Path("data/ontology/compounds.jsonl")
    result_json = Path("predictions/chebi/result.json")
    output_csv = Path("data/unified/unified_ccs_chebi.csv")

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not obo_file.exists():
        raise FileNotFoundError(f"OBO file not found: {obo_file}")

    print(f"Loading: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    df = ensure_row_id(df)

    print(f"Writing compounds file: {compounds_jsonl}")
    write_compounds_jsonl(df, str(compounds_jsonl))

    print("Running local classifier...")
    run_local_classifier(str(compounds_jsonl), str(obo_file), str(result_json))

    print(f"Parsing results: {result_json}")
    row_to_classes, row_to_match = parse_results(str(result_json))

    print("Enriching dataset...")
    enriched = enrich_dataframe(df, row_to_classes, row_to_match)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_csv, index=False)

    covered = int((enriched["chebi_count"] > 0).sum())
    total = len(enriched)
    coverage = (covered / total * 100.0) if total else 0.0

    print("Pipeline Summary: ")
    print(f"Rows: {total}")
    print(f"Covered: {covered}")
    print(f"Coverage: {coverage:.2f}%")
    print(f"Output CSV: {output_csv}")
    print(f"Output JSON: {result_json}")


if __name__ == "__main__":
    main()
