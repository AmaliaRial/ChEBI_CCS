#!/usr/bin/env python3
"""
Merge all ChEBI chunk results and extract only covered (matched) molecules.
"""

import json
from pathlib import Path


def merge_covered_chunks(chunks_dir, output_json):
    """
    Read all chunk JSON files, filter covered molecules, and write unified output.
    
    Args:
        chunks_dir (str): Directory containing chunk JSON files
        output_json (str): Output path for unified JSON
    """
    chunks_dir = Path(chunks_dir)
    output_json = Path(output_json)
    
    # Find all chunk files
    chunk_files = sorted(chunks_dir.glob("results_pablo_hybrid_chunk*.json"))
    print(f"Found {len(chunk_files)} chunk files")
    
    all_covered_results = []
    total_input_count = 0
    chunk_info = {}
    
    for chunk_file in chunk_files:
        print(f"\nProcessing {chunk_file.name}...")
        with open(chunk_file, "r") as f:
            chunk_data = json.load(f)
        
        chunk_name = chunk_file.stem
        input_count = chunk_data.get("input_count", 0)
        total_input_count += input_count
        
        # Filter results with match (covered molecules)
        covered_results = [
            result for result in chunk_data.get("results", [])
            if result.get("match") is not None
        ]
        
        chunk_info[chunk_name] = {
            "input_count": input_count,
            "total_results": len(chunk_data.get("results", [])),
            "covered_count": len(covered_results)
        }
        
        print(f"  Input count: {input_count}")
        print(f"  Total results: {len(chunk_data.get('results', []))}")
        print(f"  Covered results: {len(covered_results)}")
        
        all_covered_results.extend(covered_results)
    
    # Create unified output
    unified_data = {
        "chunk_metadata": chunk_info,
        "total_input_count": total_input_count,
        "total_covered_count": len(all_covered_results),
        "results": all_covered_results
    }
    
    # Write output
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(unified_data, f, indent=2)
    
    print(f"\n✓ Unified JSON written to {output_json}")
    print(f"  Total inputs across all chunks: {total_input_count}")
    print(f"  Total covered molecules: {len(all_covered_results)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Merge covered molecules from all ChEBI chunks"
    )
    
    # Get repo root
    repo_root = Path(__file__).parent.parent.parent.parent
    
    parser.add_argument(
        "--chunks-dir",
        default=str(repo_root / "predictions" / "chebi" / "chunks"),
        help="Directory containing chunk JSON files"
    )
    parser.add_argument(
        "--output-json",
        default=str(repo_root / "data" / "ontology" / "compounds_covered_unified.json"),
        help="Output JSON path"
    )
    
    args = parser.parse_args()
    merge_covered_chunks(args.chunks_dir, args.output_json)
