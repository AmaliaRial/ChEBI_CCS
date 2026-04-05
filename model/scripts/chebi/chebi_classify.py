#!/usr/bin/env python3
"""Simple local ChEBI classifier. (Run in chebi_classify_pipeline.py)

Input:
- JSONL with {row_id, smiles, inchi} per line, or
- plain text with one SMILES per line, or
- single SMILES string.


Output:
- JSON with {summary, results}
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*") #apagar warnings de rdkit sobre smiles invalidos, etc.


def canonical_smiles(smiles: str | None) -> str | None: #normalizar smiles para mejorar matching (el script de pablo no lo hace y hay muchos casos de no match por eso)
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def inchi_from_smiles(smiles: str | None) -> str | None: #convierte SMILES a inchi usando rdkit por si falla con smiles que mire con el inchi
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        return Chem.MolToInchi(mol).strip()
    except Exception:
        return None

#lee el .obo termino a termino y construye 4 estructuras de padres, nombres, smiles e inchi
def parse_obo(obo_path: str) -> tuple[dict[str, list[str]], dict[str, str], dict[str, list[str]], dict[str, list[str]]]:
    parents: dict[str, list[str]] = defaultdict(list) # (is_a relationships)
    names: dict[str, str] = {} #term
    smiles_map: dict[str, list[str]] = defaultdict(list)
    inchi_map: dict[str, list[str]] = defaultdict(list)

    current_id: str | None = None
    current_smiles: str | None = None
    current_inchi: str | None = None

    #cuando se encuentra un nuevo [Term] o al final del archivo, se llama a esta funcion para guardar la info del termino anterior en las estructuras
    #Parsea prefijos específicos como Término y las cosas que salen al principio y tal
    def commit_current() -> None: 
        if not current_id:
            return
        if current_smiles:
            smiles_map[current_smiles].append(current_id)
            canon = canonical_smiles(current_smiles)
            if canon:
                smiles_map[canon].append(current_id)
        if current_inchi:
            inchi_map[current_inchi].append(current_id)

    with open(obo_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")

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

            if not current_id:
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
    return dict(parents), names, dict(smiles_map), dict(inchi_map)

#dado un chebi_id, devuelve el set de todos sus ancestros (is_a) usando la estructura de padres construida antes (de forma hierárquica)
def ancestors_of(chebi_id: str, parents: dict[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    queue = deque(parents.get(chebi_id, [])) #empieza la cola con lso padres directos del chebi_id
    while queue:
        item = queue.popleft() # saca al primero, marca como visto y mete sus padres al final de la cola
        if item in seen: #continuar hasta que acabe la cola
            continue
        seen.add(item)
        queue.extend(parents.get(item, []))
    return seen


def load_inputs(input_arg: str) -> list[dict[str, Any]]:
    p = Path(input_arg) #si metemos el jsonl con la info de los compuestos, lo parsea y devuelve una lista de dicts con row_id, smiles e inchi. Si no existe el archivo, asume que el input es un único SMILES y lo devuelve como tal con row_id 1
    if not p.exists():
        return [{"row_id": 1, "smiles": input_arg, "inchi": None}]

    records: list[dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                row = json.loads(line)
                records.append(
                    {
                        "row_id": row.get("row_id", i),
                        "smiles": row.get("smiles"),
                        "inchi": row.get("inchi"),
                    }
                )
            else:
                records.append({"row_id": i, "smiles": line, "inchi": None})
    return records


#empezamos a mapear una fila a un Termino
#primero intenta con inchi exacto, luego con smiles exacto o canonical, y si no con el inchi derivado del smiles. Si encuentra un match, devuelve el chebi_id, el nombre, la fuente del match (smiles o inchi) y el valor que matcheó. Luego con el chebi_id busca sus ancestros y devuelve la lista de clasificaciones (ancestros + el propio chebi_id)
def classify_one( 
    row: dict[str, Any],
    parents: dict[str, list[str]],
    names: dict[str, str],
    smiles_map: dict[str, list[str]],
    inchi_map: dict[str, list[str]],
) -> dict[str, Any]:
    row_id = row.get("row_id")
    smiles = row.get("smiles")
    inchi = row.get("inchi")

    smiles = smiles.strip() if isinstance(smiles, str) and smiles.strip() else None
    inchi = inchi.strip() if isinstance(inchi, str) and inchi.strip() else None

    chebi_id = None
    match_source = None
    matched_value = None

    if inchi and inchi in inchi_map and inchi_map[inchi]:
        chebi_id = inchi_map[inchi][0]
        match_source = "inchi"
        matched_value = inchi
    else:
        if smiles:
            candidates = [smiles, canonical_smiles(smiles)]
            for candidate in candidates:
                if candidate and candidate in smiles_map and smiles_map[candidate]:
                    chebi_id = smiles_map[candidate][0]
                    match_source = "smiles"
                    matched_value = candidate
                    break

            if not chebi_id:
                derived = inchi_from_smiles(smiles)
                if derived and derived in inchi_map and inchi_map[derived]:
                    chebi_id = inchi_map[derived][0]
                    match_source = "inchi"
                    matched_value = derived

    if not chebi_id:
        return {
            "row_id": row_id,
            "input": {"smiles": smiles, "inchi": inchi},
            "error": "No ChEBI match found",
        }

    anc = ancestors_of(chebi_id, parents)
    classes = [{"id": cid, "name": names.get(cid, "")} for cid in sorted(anc)]
    return {
        "row_id": row_id,
        "input": {"smiles": smiles, "inchi": inchi},
        "match": {
            "chebi_id": chebi_id,
            "name": names.get(chebi_id, ""),
            "match_source": match_source,
            match_source: matched_value,
        },
        "classifications": classes,
    }


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    matched = sum(1 for r in results if "match" in r)
    unmatched = total - matched
    coverage = (matched / total * 100.0) if total else 0.0
    return {
        "total_inputs": total,
        "matched": matched,
        "unmatched": unmatched,
        "coverage_pct": round(coverage, 2),
    }

#Por si lo queremos ejecutar por su cuenta
def main() -> None:
    parser = argparse.ArgumentParser(description="Simple local ChEBI classification from OBO")
    parser.add_argument("--input", required=True)
    parser.add_argument("--obo", default="data/ontology/chebi.obo")
    parser.add_argument("--out", default="predictions/chebi/result.json")
    args = parser.parse_args()

    if not Path(args.obo).exists():
        raise FileNotFoundError(f"OBO file not found: {args.obo}")

    rows = load_inputs(args.input)
    parents, names, smiles_map, inchi_map = parse_obo(args.obo)

    results = []
    for i, row in enumerate(rows, start=1):
        if i == 1 or i % 1000 == 0:
            print(f"Progress: {i}/{len(rows)}", flush=True)
        results.append(classify_one(row, parents, names, smiles_map, inchi_map))

    summary = build_summary(results)
    payload = {"summary": summary, "results": results}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("Local ChEBI Summary: ")
    print(f"Total: {summary['total_inputs']}")
    print(f"Matched: {summary['matched']}")
    print(f"Unmatched: {summary['unmatched']}")
    print(f"Coverage: {summary['coverage_pct']}%")
    print(f"Output: {args.out}\n")


if __name__ == "__main__":
    main()
