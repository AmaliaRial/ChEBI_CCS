#!/usr/bin/env python3
"""
Produce an overview of the ChEBI ontology hierarchy for matched CHEBI ids.

Outputs a JSON with:
- ancestor_distance_counts: how many unique ancestors appear at each distance
- ancestor_frequency_by_distance: top ancestors per distance with counts across matched terms
- matched_term_counts: number of matched terms processed
- suggestions: simple heuristics for depth cutoff

Usage:
  python ontology_hierarchy_overview.py --unified-json <path> --obo <path> --max-depth 6 --output <path>
"""

from pathlib import Path
import json
import argparse
from collections import defaultdict, Counter, deque


def parse_obo(obo_path: Path):
    term_to_parents = defaultdict(list)
    term_names = {}

    with obo_path.open("r", encoding="utf-8") as fh:
        in_term = False
        cur_id = None
        cur_name = None
        cur_parents = []

        def flush_term():
            nonlocal cur_id, cur_name, cur_parents, in_term
            if cur_id:
                term_names[cur_id] = cur_name or ""
                term_to_parents[cur_id] = list(dict.fromkeys(cur_parents))
            cur_id = None
            cur_name = None
            cur_parents = []
            in_term = False

        for raw in fh:
            line = raw.rstrip("\n")
            if line == "[Term]":
                flush_term()
                in_term = True
                continue
            if not in_term:
                continue
            if not line:
                if in_term:
                    flush_term()
                continue
            if line.startswith("id: "):
                cur_id = line[4:].strip()
                continue
            if line.startswith("name: "):
                cur_name = line[6:].strip()
                continue
            if line.startswith("is_a: "):
                parent_id = line[6:].split("!",1)[0].strip().split()[0]
                if parent_id:
                    cur_parents.append(parent_id)
        flush_term()
    return term_to_parents, term_names


def load_matched_chebi_ids(unified_json: Path):
    with unified_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    chebi_ids = []
    for item in data.get("results", []):
        match = item.get("match") or {}
        cid = match.get("chebi_id")
        if cid:
            chebi_ids.append(cid)
    return chebi_ids


def build_ancestor_distances(chebi_ids, term_to_parents, max_depth=6):
    # For each matched term, BFS upward to collect ancestors and their minimal distance
    ancestor_stats = defaultdict(Counter)  # distance -> Counter(ancestor_id -> count)
    unique_ancestors_by_distance = defaultdict(set)

    processed = 0
    for cid in chebi_ids:
        processed += 1
        # BFS queue of (node, distance)
        q = deque()
        q.append((cid, 0))
        seen = {cid: 0}
        while q:
            node, dist = q.popleft()
            # record ancestor (including self at dist 0)
            if dist <= max_depth:
                ancestor_stats[dist][node] += 1
                unique_ancestors_by_distance[dist].add(node)
            if dist == max_depth:
                continue
            parents = term_to_parents.get(node, [])
            for p in parents:
                if p not in seen or seen[p] > dist + 1:
                    seen[p] = dist + 1
                    q.append((p, dist + 1))
    # Convert unique sets to counts
    ancestor_distance_counts = {str(d): len(s) for d, s in unique_ancestors_by_distance.items()}
    # For each distance, get top ancestors
    ancestor_frequency_by_distance = {}
    for dist, counter in ancestor_stats.items():
        top = counter.most_common(50)
        ancestor_frequency_by_distance[str(dist)] = top
    return {
        "processed_matched_terms": processed,
        "ancestor_distance_counts": ancestor_distance_counts,
        "ancestor_frequency_by_distance": ancestor_frequency_by_distance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-json", type=Path, required=True)
    parser.add_argument("--obo", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("data/ontology/ontology_hierarchy_overview.json"))
    args = parser.parse_args()

    print("Parsing OBO (this may take a while)...")
    term_to_parents, term_names = parse_obo(args.obo)
    print(f"Terms parsed: {len(term_to_parents)}")

    print("Loading matched CHEBI ids...")
    chebi_ids = load_matched_chebi_ids(args.unified_json)
    print(f"Matched CHEBI ids: {len(chebi_ids)}")

    print(f"Computing ancestor distances up to depth {args.max_depth}...")
    overview = build_ancestor_distances(chebi_ids, term_to_parents, max_depth=args.max_depth)

    # Enrich top ancestors with names
    enriched = {}
    for dist, top_list in overview["ancestor_frequency_by_distance"].items():
        enriched[dist] = [ {"chebi_id": anc, "count": cnt, "name": term_names.get(anc,"")} for anc, cnt in top_list ]

    out = {
        "processed_matched_terms": overview["processed_matched_terms"],
        "ancestor_distance_counts": overview["ancestor_distance_counts"],
        "ancestor_frequency_by_distance": enriched,
        "max_depth": args.max_depth,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print(f"Overview written to {args.output}")

if __name__ == "__main__":
    main()
