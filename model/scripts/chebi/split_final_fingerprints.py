#!/usr/bin/env python3
"""
Split the final fingerprints dataset into 80/10/10 train/val/test.

Input: data/model/final_covered_ccs_fingerprints.csv (16,892 rows × 2,221 columns)

Output:
  - data/model/train_ccs_fingerprints.csv (80%)
  - data/model/val_ccs_fingerprints.csv (10%)
  - data/model/test_ccs_fingerprints.csv (10%)
  - data/model/split_manifest.json (metadata)
"""

from pathlib import Path
import sys

# Add parent dirs to path for imports
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root / "model" / "scripts"))

from splitter import save_split_train_val_test

INPUT_CSV = repo_root / "data" / "model" / "final_covered_ccs_fingerprints.csv"
TRAIN_CSV = repo_root / "data" / "model" / "train_ccs_fingerprints.csv"
VAL_CSV = repo_root / "data" / "model" / "val_ccs_fingerprints.csv"
TEST_CSV = repo_root / "data" / "model" / "test_ccs_fingerprints.csv"
MANIFEST_JSON = repo_root / "data" / "model" / "split_manifest.json"


def main():
	print(f"Input CSV: {INPUT_CSV}")
	if not INPUT_CSV.exists():
		print(f"ERROR: Input file not found: {INPUT_CSV}")
		sys.exit(1)

	print(f"Performing 80/10/10 split...")
	train_df, val_df, test_df = save_split_train_val_test(
		input_csv=INPUT_CSV,
		train_csv=TRAIN_CSV,
		val_csv=VAL_CSV,
		test_csv=TEST_CSV,
		val_size=0.1,
		test_size=0.1,
		random_state=42,
		manifest_path=MANIFEST_JSON,
	)

	print(f"\n✓ Split complete!")
	print(f"  Train: {len(train_df)} rows → {TRAIN_CSV}")
	print(f"  Val:   {len(val_df)} rows → {VAL_CSV}")
	print(f"  Test:  {len(test_df)} rows → {TEST_CSV}")
	print(f"  Manifest: {MANIFEST_JSON}")
	print(f"\n✓ All files written successfully")


if __name__ == "__main__":
	main()
