# ChEBI_CCS Repository Technical Map

**Last Updated:** May 13, 2026  
**Project:** Biomedical Engineering Thesis - CCS Prediction with ChEBI Ontology Integration

---

## 1. Project Overview

This repository implements a machine learning pipeline for Collision Cross Section (CCS) prediction that integrates chemical ontology information from ChEBI. The project has four main components:

1. **CCS Data Preparation**: Dataset cleaning, unification, ChEBI matching, fingerprint generation, and train/val/test splitting
2. **ChEBI Classification**: Local classification over ChEBI ontology with optional HTTP variants
3. **Base Model Training**: Baseline CCS regression without ontology features
4. **Ontology-Aware Model Training**: Multitask neural network (CCS regression + multilabel classification)
5. **External Model Benchmarking**: Wrapper execution for DeepCCS and DarkChem

---

## 2. Repository Structure (Updated)

```
model/scripts/
├── data_management/    ← Data processing & splitting
│   ├── splitter.py
│   ├── build_final_covered_dataset.py
│   ├── build_final_fingerprint_dataset.py
│   └── check_and_correct_metlin_ims_ccs.py
└── chebi/             ← ChEBI classification & validation
    ├── chebi_classify.py
    ├── chebi_classify_pipeline.py
    ├── check_ccs_replicates.py
    ├── prepare_chebi_multilabel_dataset.py
    ├── chebi_classify_pablo_http.py
    └── chebi_classify_pablo_hybrid_http.py
```

---

## 3. Core Active Dataset Files

| File | Purpose | Rows | Columns | Notes |
|------|---------|------|---------|-------|
| `data/unified/unified_ccs.csv` | Base unified dataset from all sources | ~68k | row_id, smiles, adduct, ccs, inchi, name, mz, source_dataset | Input to ChEBI pipeline |
| `data/model/final_covered_ccs.csv` | After ChEBI matching; only matched rows | ~50-55k | + chebi_classes, chebi_count, chebi_name, chebi_match_source | ChEBI metadata in JSON |
| `data/model/final_covered_ccs_fingerprints.csv` | With fingerprint vectors only (no descriptors) | ~50-55k | rows + V1..V2211 (fingerprint columns) | Final dataset for model training |
| `data/model/train_ccs_fingerprints.csv` | Training split (80%) | ~41-44k | Same as above | Deterministic split, random_state=42 |
| `data/model/val_ccs_fingerprints.csv` | Validation split (10%) | ~5-5.5k | Same as above | Deterministic split, random_state=42 |
| `data/model/test_ccs_fingerprints.csv` | Test split (10%) | ~5-5.5k | Same as above | Deterministic split, random_state=42 |
| `data/model/chebi_ontology_labels.csv` | Binary multilabel ontology columns | Same as final_covered_ccs | row_id, source_dataset, chebi_* columns, ontology__*labels | Uses min_class_count filtering |

---

## 4. Python Scripts - Detailed Inventory

### A. Core Data Processing Scripts (in `model/scripts/data_management/`)

#### `model/scripts/data_management/build_final_covered_dataset.py`
- **Purpose**: Rebuild the covered CCS dataset by merging ChEBI classification results
- **Input**: 
  - `data/unified/unified_ccs.csv` (source)
  - `predictions/chebi/chunks/results_pablo_hybrid_chunk*.json` (ChEBI matches)
- **Output**: 
  - `data/model/final_covered_ccs.csv`
  - `data/model/final_covered_manifest.json` (metadata)
- **Key Functions**:
  - `read_csv()`: Load CSV with fieldnames
  - `load_matches()`: Parse ChEBI classification JSON outputs
  - `build_rows()`: Enrich rows with ChEBI metadata (chebi_classes as JSON, chebi_count, chebi_name, chebi_match_source)
  - `write_csv()`: Write enriched rows
- **Status**: **ACTIVE** - Part of recommended workflow
- **Dependencies**: Requires prior ChEBI classification pipeline execution

---

#### `model/scripts/data_management/build_final_fingerprint_dataset.py`
- **Purpose**: Append fingerprint vectors (V1..Vn) to covered dataset; exclude descriptor columns
- **Input**: 
  - `data/model/final_covered_ccs.csv` (source with row_id keys)
  - Raw fingerprint tables: `data/raw_datasets/fingerprints/*.csv` or `*.tsv`
- **Output**: 
  - `data/model/final_covered_ccs_fingerprints.csv` (with V1..Vn columns)
  - `data/model/final_covered_ccs_fingerprints_manifest.json` (metadata)
- **Key Functions**:
  - `normalize()`: Text normalization for key matching
  - `read_table()`: Auto-detect delimiter (CSV/TSV)
  - `fingerprint_columns()`: Extract V\d+ pattern columns
  - `build_fingerprint_index()`: Match source rows to raw fingerprints by SMILES/InChI/Name
  - `source_candidates()` / `raw_candidates()`: Generate candidate keys for matching
- **Config Datasets**:
  - `ccsbase_descriptors`
  - `AllCCS2_experimental_with_inchis_descriptors`
  - `METLIN-CCS-Lipids`
  - `METLIN_IMS`
- **Status**: **ACTIVE** - Part of recommended workflow
- **Notes**: Intentionally excludes descriptors to keep fingerprints only

---

#### `model/scripts/data_management/splitter.py`
- **Purpose**: Split final fingerprints dataset into 80/10/10 train/val/test
- **Input**: `data/model/final_covered_ccs_fingerprints.csv` (16,892 rows × 2,221 columns)
- **Output**:
  - `data/model/train_ccs_fingerprints.csv` (80%)
  - `data/model/val_ccs_fingerprints.csv` (10%)
  - `data/model/test_ccs_fingerprints.csv` (10%)
  - `data/model/split_manifest.json` (metadata)
- **Key Functions**: `split_train_val_test()`, `save_split_train_val_test()`, `main()`
- **Parameters**: `val_size=0.1`, `test_size=0.1`, `random_state=42`
- **Status**: **ACTIVE** - Part of recommended workflow
- **Notes**: Deterministic splits for reproducibility

---

#### `model/scripts/data_management/splitter.py`
- **Purpose**: Reusable utility functions for train/test/val splitting
- **Functions**:
  - `split_train_test(df, test_size=0.2, random_state=42)` → (train_df, test_df)
  - `split_train_val_test(df, val_size=0.1, test_size=0.1, random_state=42)` → (train_df, val_df, test_df)
  - `save_split()` → saves 80/20 split CSVs
  - `save_split_train_val_test()` → saves 80/10/10 split CSVs with optional manifest
- **Status**: **DEPRECATED** - Functionality consolidated into `splitter.py`
- **Notes**: Not meant to be run directly; imported by other scripts

---

#### `model/scripts/data_management/check_and_correct_metlin_ims_ccs.py`
- **Purpose**: METLIN_IMS-specific CCS processing; handles replicates, CV filtering, matching logic
- **Input**: `data/raw_datasets/fingerprints/METLIN_IMS_vectorfingerprintsVectorized.tsv` (raw)
- **Output**: `data/model/final_covered_ccs_corrected.csv` + reports
- **Key Functions**:
  - `normalize_text()` / `normalize_numeric()`: Normalize matching keys
  - `annotate_raw_metrics()`: Compute replicate statistics (average, std, CV%)
  - `prepare_matching_frame()`: Build candidate keys (row_id, inchi, smiles, name, mz)
  - `match_metlin_rows()`: Multi-strategy matching (row_id → inchikey → inchi+adduct → smiles+adduct → name+adduct → mz+adduct)
- **Status**: **PROBABLY OBSOLETE** - Superseded by check_ccs_replicates.py (more general)
- **Notes**: May be kept for reference or METLIN_IMS-specific edge cases; moved to data_management

---

### B. ChEBI Classification & Validation Scripts (in `model/scripts/chebi/`)

#### `model/scripts/data_management/prepare_chebi_multilabel_dataset.py`
- **Purpose**: Convert ChEBI classes (stored as JSON strings) into binary multilabel columns
- **Input**: `data/model/final_covered_ccs.csv` (with chebi_classes column as JSON)
- **Output**: 
  - `data/model/chebi_ontology_labels.csv` (binary multilabel table)
  - `data/model/chebi_ontology_manifest.json` (metadata)
- **Key Functions**: (in chebi/ subfolder)
  - `parse_label_values()`: Parse ChEBI classes from multiple formats (JSON, list, CSV)
  - `select_label_column()`: Auto-detect label column
  - `build_label_table()`: Create binary columns (ontology__CHEBI:XXXXX format)
  - Filters ontology labels by `min_class_count` (default 25)
- **Parameters**: `--min-class-count 25` (configurable)
- **Output Columns**: 
  - row_id
  - ontology__CHEBI:XXXXX (binary: 0 or 1)
  - ontology_label_count (total labels per row)
  - ontology_has_labels (boolean)
- **Status**: **ACTIVE** - Part of recommended workflow for ontology model training
- **Notes**: Creates compact multilabel representation; filters sparse classes

---

### B. ChEBI Classification & Validation Scripts (in `model/scripts/chebi/`)

#### `model/scripts/data_management/check_and_correct_metlin_ims_ccs.py`
- **Purpose**: METLIN_IMS-specific CCS processing; handles replicates, CV filtering, matching logic
- **Input**: `data/raw_datasets/fingerprints/METLIN_IMS_vectorfingerprintsVectorized.tsv` (raw)
- **Output**: `data/model/final_covered_ccs_corrected.csv` + reports
- **Key Functions**:
  - `normalize_text()` / `normalize_numeric()`: Normalize matching keys
  - `annotate_raw_metrics()`: Compute replicate statistics (average, std, CV%)
  - `prepare_matching_frame()`: Build candidate keys (row_id, inchi, smiles, name, mz)
  - `match_metlin_rows()`: Multi-strategy matching (row_id → inchikey → inchi+adduct → smiles+adduct → name+adduct → mz+adduct)
- **Status**: **PROBABLY OBSOLETE** - Superseded by check_ccs_replicates.py (more general)
- **Notes**: May be kept for reference or METLIN_IMS-specific edge cases

---

### B. ChEBI Classification Scripts

#### `model/scripts/chebi/chebi_classify.py`
- **Purpose**: Local ChEBI classifier; loads OBO file and matches molecules by SMILES/InChI
- **Input**: 
  - JSONL file (one {row_id, smiles, inchi} per line) OR
  - Plain text (one SMILES per line) OR
  - Single SMILES string
  - `data/ontology/chebi.obo` (ChEBI ontology)
- **Output**: JSON with `{summary, results}` containing matched ChEBI IDs and classifications
- **Key Functions**:
  - `canonical_smiles()`: Normalize SMILES using RDKit
  - `inchi_from_smiles()`: Convert SMILES to InChI
  - `parse_obo()`: Parse OBO file into (parents, names, smiles_map, inchi_map)
  - `ancestors_of()`: Compute all ancestors (is_a) for a ChEBI ID
  - `load_inputs()`: Parse JSONL or SMILES input
  - `classify_compound()`: Match molecule and return ancestors
- **Status**: **ACTIVE** - Used by `chebi_classify_pipeline.py`
- **Notes**: Local execution (no HTTP); handles RDKit canonicalization for robustness

---

#### `model/scripts/chebi/chebi_classify_pipeline.py`
- **Purpose**: Orchestrate full local ChEBI classification pipeline
- **Workflow**:
  1. Load `data/unified/unified_ccs.csv`
  2. Write compounds to JSONL (`data/ontology/compounds.jsonl`)
  3. Call `chebi_classify.py` (subprocess)
  4. Parse results JSON
  5. Enrich dataframe with ChEBI metadata
  6. Save `data/unified/unified_ccs_chebi.csv`
- **Input**: `data/unified/unified_ccs.csv`
- **Output**: 
  - `data/unified/unified_ccs_chebi.csv` (enriched with chebi_classes, chebi_count, chebi_name, chebi_match_source)
  - `predictions/chebi/result.json` (classification results)
  - `data/ontology/compounds.jsonl` (intermediate)
- **Key Functions**:
  - `ensure_row_id()`: Add row_id if missing
  - `write_compounds_jsonl()`: Export molecules for classification
  - `run_local_classifier()`: Execute chebi_classify.py via subprocess
  - `parse_results()`: Extract classifications from JSON
  - `enrich_dataframe()`: Add ChEBI columns to dataframe
- **Status**: **ACTIVE** - Higher-level wrapper for full classification
- **Command**: `conda run -n chebi_ccs python model/scripts/chebi/chebi_classify_pipeline.py`
- **Notes**: More efficient than HTTP variants (local OBO file, no network calls)

---

#### `model/scripts/chebi/chebi_classify_pablo_http.py`
- **Purpose**: HTTP-based ChEBI classifier (older/alternative variant)
- **Status**: **PROBABLY OBSOLETE** - Older HTTP-based approach
- **Notes**: Not examined in detail; likely superseded by local classifier

---

#### `model/scripts/chebi/chebi_classify_pablo_hybrid_http.py`
- **Purpose**: HTTP-based hybrid ChEBI classifier (older/alternative variant)
- **Status**: **PROBABLY OBSOLETE** - Older HTTP-based approach
- **Notes**: Not examined in detail; chunks processing may have been used for large datasets

---

### C. Model Classes (root level `model/`)

#### `model/base_model.py`
- **Purpose**: Baseline CCS regression model (no ontology features)
- **Architecture**:
  ```
  Input (fingerprints + adduct one-hot + m/z)
    → fc1 (1024) + LeakyReLU + Dropout(0.2)
    → fc2 (256) + LeakyReLU + Dropout(0.2)
    → fc3 (64) + LeakyReLU
    → CCS Output (regression, single value)
  ```
- **Class**: `CCSRegressor(nn.Module)`
- **Key Functions**:
  - `__init__()`: Initialize layers (fc1, fc2, fc3, output)
  - `forward()`: Pass data through network
  - `get_fingerprint_columns()`: Extract V\d+ columns
  - `get_column_name()`: Flexible column name resolution
  - `build_feature_matrix()`: Construct input tensor (fingerprints + adduct + m/z)
- **Loss**: MSELoss (regression)
- **Status**: **ACTIVE** - Reference baseline model
- **Training**: Use `train_ccs_fingerprints.csv` with deterministic splits

---

#### `model/chebi_model.py`
- **Purpose**: Multitask neural network (CCS regression + ChEBI multilabel classification)
- **Architecture**:
  ```
  Input (fingerprints + adduct one-hot + m/z)
    → fc1 (1024) + LeakyReLU + Dropout(0.2)
    → fc2 (256) + LeakyReLU + Dropout(0.2)
    → fc3 (64) + LeakyReLU
    → embedding (64-dim latent space)
    ├─ CCS head: 1 output (regression)
    └─ Ontology head: num_ontology_labels outputs (multilabel logits)
  ```
- **Class**: `CCSRegressor(nn.Module)`
  - Constructor: `__init__(input_dim, hidden_dims=(1024, 256, 64), num_ontology_labels)`
  - Forward: `forward(x) → (ccs_pred, ontology_logits, embedding)`
- **Key Functions**:
  - `get_fingerprint_columns()`: Same as base model
  - Additional multilabel utilities (TBD in full inspection)
- **Loss**: 
  - CCS head: MSELoss
  - Ontology head: BCEWithLogitsLoss (multilabel)
  - Total: `loss = mse_ccs + lambda_ontology * bce_ontology` (initial λ = 0.1)
- **Status**: **ACTIVE** - Ontology-aware model (follows copilot-instructions.md)
- **Notes**: 
  - Embedding space intended for latent chemical ontology relationship learning
  - Do NOT apply sigmoid before BCEWithLogitsLoss
  - No sigmoid on logits; let loss function handle it

---

### D. Encoder Classes (in `model/enconders/`)

#### `model/enconders/adduct_encoder.py`
- **Purpose**: One-hot encode adduct values for model input
- **Class**: `AdductOneHotEncoder`
- **Methods**:
  - `fit(adducts)`: Learn unique adducts
  - `transform(adducts)` → numpy array (one-hot encoded)
  - `save_encoder(file_path)`: Save converter dict to JSON
  - `load_encoder(file_path)`: Load converter from JSON
- **Status**: **ACTIVE** - Used by both base and ChEBI models
- **Notes**: Simple one-hot encoding; handles string normalization

---

#### `model/enconders/chebi_encoder.py`
- **Purpose**: (Comments only; not implemented yet)
- **Concept**: Embedding space learning to encourage similar ChEBI classes to be nearby in latent space
- **Status**: **NOT IMPLEMENTED** - Only contains conceptual notes
- **Notes**: Part of future ontology embedding research

---

### E. Utility & Validation Scripts (root level `model/`)

#### `model/sanity_checks.py`
- **Purpose**: (Not implemented; comments only)
- **Status**: **INCOMPLETE** - Only placeholder comments about CCS variance checking
- **Notes**: Should implement pre-training validation checks

---

#### `dataset_cleaning.py` (root level)
- **Purpose**: (Not actively used; test code snippets)
- **Content**: Commented-out dataset loading examples
- **Status**: **PROBABLY OBSOLETE** - Only contains test snippets
- **Notes**: Likely early exploration code; safe to ignore or delete

---

### F. Benchmark Scripts (in `benchmark/scripts/`)

#### `benchmark/scripts/run_benchmark.py`
- **Purpose**: Orchestrate external CCS predictor benchmarking (DeepCCS, DarkChem)
- **Input**: 
  - `--input` CSV with smiles/adduct/ccs columns
  - `--config` YAML file (default: `configs/benchmark_models.yaml`)
  - `--output-dir` for predictions
- **Key Functions**:
  - `_setup_logging()`: Configure logging
  - `_load_config()`: Parse YAML
  - `_resolve_repo_path()`: Handle internal/external repo paths
  - `_build_extra_wrapper_args()`: Build CLI arguments from config
  - `_run_model()`: Execute wrapper via conda subprocess
- **Config Format**: YAML with:
  - `external_root`: Path to external repos
  - `models`: Dict of {model_name: {enabled, conda_env, wrapper_args, etc.}}
- **Status**: **ACTIVE** - Used for benchmark comparisons
- **Output**: Predictions CSV per model in `predictions/{model_name}/predictions.csv`

---

#### `benchmark/scripts/aggregate_metrics.py`
- **Purpose**: Compute comparison metrics across all benchmark predictions
- **Input**: Predictions CSVs from `predictions/{model_name}/predictions.csv`
- **Metrics Computed**:
  - n (sample count)
  - MAE (mean absolute error)
  - RMSE (root mean squared error)
  - MPE (mean percent error)
  - MAPE (mean absolute percent error)
  - std_abs_error / std_pct_error
  - outliers_gt_10pct (count)
- **Output**: 
  - `predictions/reports/benchmark_predictions.csv` (aggregated predictions)
  - `predictions/reports/metrics.csv` (metrics table)
- **Status**: **ACTIVE** - Used for benchmark evaluation
- **Key Functions**:
  - `_load_predictions()`: Load and normalize prediction CSV
  - `_compute_metrics()`: Calculate all metrics
  - `_collect_prediction_files()`: Auto-discover predictions

---

#### `benchmark/scripts/wrappers/deepccs.py` (in benchmark/scripts/wrappers/)
- **Purpose**: Wrapper to call DeepCCS external model
- **Input**: CSV with smiles/adduct/ccs
- **Logic**:
  - Validates adduct support (M+H, M+Na, M-H, M-2H)
  - Calls DeepCCS CLI tool via subprocess
  - Parses output predictions
- **Output**: `predictions.csv` with predicted_ccs column
- **Status**: **ACTIVE** - Part of benchmark suite
- **Dependencies**: Requires `deepccs` conda environment and external DeepCCS repo

---

#### `benchmark/scripts/wrappers/darkchem.py`
- **Purpose**: Wrapper to call DarkChem external model
- **Input**: CSV with smiles/adduct/ccs
- **Logic**:
  - Groups adducts into {protonated, sodiated, deprotonated}
  - Discovers pre-trained networks in configured path
  - Calls DarkChem predictor via subprocess
  - Aggregates predictions
- **Output**: `predictions.csv` with predicted_ccs column
- **Status**: **ACTIVE** - Part of benchmark suite
- **Dependencies**: Requires `darkchem` conda environment and external DarkChem repo

---

## 5. Recommended Current Workflow

Execute these steps in order to prepare data for model training:

### Step 1: Verify CCS Replicate Handling
```bash
conda run -n chebi_ccs python model/scripts/chebi/check_ccs_replicates.py \
  --input-dir data/raw_datasets \
  --output-dir data/clean_datasets/ccs_replicate_check \
  --cv-threshold 5
```
**Output**: Per-dataset reports; identifies rows with high CCS variance

---

### Step 2: Rebuild Covered Dataset from ChEBI Classification
```bash
conda run -n chebi_ccs python model/scripts/data_management/build_final_covered_dataset.py
```
**Prerequisites**: ChEBI classification must be completed first (results in `predictions/chebi/chunks/`)  
**Output**: `data/model/final_covered_ccs.csv` (~50-55k rows with ChEBI metadata)

---

### Step 3: Build Final Fingerprints Dataset
```bash
conda run -n chebi_ccs python model/scripts/data_management/build_final_fingerprint_dataset.py
```
**Input**: `data/model/final_covered_ccs.csv`  
**Output**: `data/model/final_covered_ccs_fingerprints.csv` (16,892 rows × 2,221 columns)

---

### Step 4: Create 80/10/10 Train/Val/Test Splits
```bash
conda run -n chebi_ccs python model/scripts/data_management/splitter.py
```
**Input**: `data/model/final_covered_ccs_fingerprints.csv`  
**Output**: 
- `data/model/train_ccs_fingerprints.csv` (80%)
- `data/model/val_ccs_fingerprints.csv` (10%)
- `data/model/test_ccs_fingerprints.csv` (10%)
- `data/model/split_manifest.json` (metadata with random_state=42)

---

### Step 5: Prepare ChEBI Multilabel Ontology Labels (For Ontology Model Only)
```bash
conda run -n chebi_ccs python model/scripts/data_management/prepare_chebi_multilabel_dataset.py \
  --input-csv data/model/final_covered_ccs.csv \
  --output-csv data/model/chebi_ontology_labels.csv \
  --min-class-count 25
```
**Input**: `data/model/final_covered_ccs.csv` (with chebi_classes as JSON)  
**Output**: `data/model/chebi_ontology_labels.csv` (binary multilabel format: ontology__CHEBI:XXXXX)

---

### Step 6: Train Base Model (Reference Baseline)
```bash
conda run -n chebi_ccs python model/base_model.py \
  --train data/model/train_ccs_fingerprints.csv \
  --val data/model/val_ccs_fingerprints.csv \
  --test data/model/test_ccs_fingerprints.csv
```
**Output**: Model artifacts in `model/artifacts/` (TBD: exact structure from code inspection)

---

### Step 7: Train ChEBI Multitask Model (Ontology-Aware)
```bash
conda run -n chebi_ccs python model/chebi_model.py \
  --train data/model/train_ccs_fingerprints.csv \
  --val data/model/val_ccs_fingerprints.csv \
  --test data/model/test_ccs_fingerprints.csv \
  --ontology data/model/chebi_ontology_labels.csv \
  --lambda-ontology 0.1
```
**Output**: Model artifacts (separate from base model for comparison)

---

### Step 8: Run External Model Benchmarks (Optional)
```bash
python benchmark/scripts/run_benchmark.py \
  --input data/model/test_ccs_fingerprints.csv \
  --config benchmark/configs/benchmark_models.yaml \
  --output-dir predictions
```
**Output**: Predictions from DeepCCS and DarkChem in `predictions/{model_name}/predictions.csv`

---

### Step 9: Aggregate Benchmark Metrics (Optional)
```bash
python benchmark/scripts/aggregate_metrics.py \
  --predictions predictions \
  --output predictions/reports/metrics.csv
```
**Output**: Comparison metrics table

---

## 7. Critical Differences: Split Scripts

### `model/scripts/data_management/splitter.py`
- **Type**: Utility library (NOT executable)
- **Purpose**: Reusable splitting functions
- **Functions**:
  - `split_train_test()`: 80/20
  - `split_train_val_test()`: 80/10/10
  - `save_split()`: Write 80/20 splits to CSV
  - `save_split_train_val_test()`: Write 80/10/10 splits to CSV with optional manifest
- **Usage**: Imported by other scripts; not run directly
- **Parameters**: Configurable via function arguments

---

### `model/scripts/data_management/splitter.py`
- **Type**: Standalone executable script
- **Purpose**: Split the final fingerprints dataset into 80/10/10 train/val/test
- **Contains**: Both utility functions and entry point (not a wrapper)
- **Inputs**: Hardcoded paths (can be parameterized via function arguments)
  - INPUT: `data/model/final_covered_ccs_fingerprints.csv`
  - OUTPUT: Train/Val/Test in `data/model/`
- **Parameters**: Hardcoded (val_size=0.1, test_size=0.1, random_state=42)
- **Status**: **CURRENTLY USED** - This is the official split script for the project
- **Command**: `python model/scripts/data_management/splitter.py`

---

### Summary
- **`splitter.py`** = Standalone executable script for splitting the final model dataset (includes both functions and entry point)

---

## 6. File Status Summary Table

| File | Purpose | Input | Output | Current Status | Notes |
|------|---------|-------|--------|----------------|----|
| `model/base_model.py` | CCS regression model (baseline) | fingerprints + adduct + m/z | CCS prediction | **ACTIVE** | Reference model without ontology |
| `model/chebi_model.py` | Multitask model (CCS + ontology) | fingerprints + adduct + m/z + ontology labels | CCS + ontology predictions | **ACTIVE** | Follows copilot-instructions.md |
| `model/sanity_checks.py` | Data validation (not implemented) | N/A | N/A | **INCOMPLETE** | Only comments; needs implementation |
| `model/encoders/adduct_encoder.py` | One-hot encode adducts | adduct strings | one-hot matrix | **ACTIVE** | Used by both models |
| `model/encoders/chebi_encoder.py` | ChEBI embedding (conceptual) | N/A | N/A | **NOT IMPLEMENTED** | Research idea only |
| `model/scripts/data_management/splitter.py` | Split utilities library | CSV dataframe | train/val/test splits | **ACTIVE** | Imported, not run directly |
| `model/scripts/data_management/splitter.py` | Final dataset 80/10/10 split | `final_covered_ccs_fingerprints.csv` | train/val/test CSVs + manifest | **ACTIVE** | Official split script (standalone) |
| `model/scripts/data_management/build_final_covered_dataset.py` | Rebuild with ChEBI matches | unified_ccs + ChEBI JSON | final_covered_ccs.csv | **ACTIVE** | Part of pipeline (moved) |
| `model/scripts/data_management/build_final_fingerprint_dataset.py` | Append fingerprints | final_covered_ccs + raw fingerprints | final_covered_ccs_fingerprints.csv | **ACTIVE** | Part of pipeline (moved) |
| `model/scripts/data_management/prepare_chebi_multilabel_dataset.py` | ChEBI labels → binary multilabel | final_covered_ccs_fingerprints.csv | final_covered_ccs_fingerprints_multilabel.csv | **ACTIVE** | For ontology model training |
| `model/scripts/chebi/check_ccs_replicates.py` | Validate CCS replicates | raw CSV/TSV files | clean datasets + reports | **ACTIVE** | General validation |
| `model/scripts/data_management/check_and_correct_metlin_ims_ccs.py` | METLIN_IMS-specific CCS handling | METLIN raw TSV | corrected CSV + reports | **PROBABLY OBSOLETE** | Superseded by check_ccs_replicates.py (moved) |
| `model/scripts/chebi/chebi_classify.py` | Local ChEBI classifier | JSONL (smiles/inchi) + chebi.obo | JSON with classifications | **ACTIVE** | Used by pipeline |
| `model/scripts/chebi/chebi_classify_pipeline.py` | Orchestrate ChEBI classification | unified_ccs.csv | unified_ccs_chebi.csv | **ACTIVE** | Higher-level wrapper |
| `model/scripts/chebi/chebi_classify_pablo_http.py` | HTTP ChEBI classifier (old) | N/A | N/A | **PROBABLY OBSOLETE** | Older approach |
| `model/scripts/chebi/chebi_classify_pablo_hybrid_http.py` | HTTP hybrid ChEBI (old) | N/A | N/A | **PROBABLY OBSOLETE** | Older approach |
| `benchmark/scripts/run_benchmark.py` | Orchestrate external models | test CSV + config YAML | predictions per model | **ACTIVE** | Benchmark suite |
| `benchmark/scripts/aggregate_metrics.py` | Compute benchmark metrics | predictions CSVs | metrics.csv | **ACTIVE** | Benchmark evaluation |
| `benchmark/scripts/wrappers/deepccs.py` | DeepCCS wrapper | test CSV | deepccs predictions.csv | **ACTIVE** | External model A |
| `benchmark/scripts/wrappers/darkchem.py` | DarkChem wrapper | test CSV | darkchem predictions.csv | **ACTIVE** | External model B |
| `dataset_cleaning.py` (root) | Early exploration code | N/A | N/A | **PROBABLY OBSOLETE** | Test snippets only |

---

## 8. Folder Organization Summary

**Before**: Mixed scripts in `model/scripts/` and `model/scripts/chebi/`

**After**: Organized into logical subfolders:
```
model/scripts/
├── data_management/     ← Scripts for data processing & splitting
│   ├── splitter.py
│   ├── build_final_covered_dataset.py
│   ├── build_final_fingerprint_dataset.py
│   └── check_and_correct_metlin_ims_ccs.py
└── chebi/              ← Scripts for ChEBI classification & validation
    ├── chebi_classify.py
    ├── chebi_classify_pipeline.py
    ├── check_ccs_replicates.py
    ├── prepare_chebi_multilabel_dataset.py
    ├── chebi_classify_pablo_http.py
    └── chebi_classify_pablo_hybrid_http.py
```

**Benefits**:
- Clearer separation of concerns
- Easier to find related scripts
- Better organization for multi-step pipelines

---

## 9. Files Requiring Future Review

These files should be examined or updated but are not critical for current operations:

1. **`model/sanity_checks.py`**
   - Currently: Only comments
   - Recommended Action: Implement pre-training validation checks (null counts, value ranges, fingerprint stats)

2. **`model/scripts/check_and_correct_metlin_ims_ccs.py`**
   - Currently: METLIN_IMS-specific implementation
   - Recommended Action: Consolidate with `check_ccs_replicates.py` or document why separate handling is needed

3. **`model/scripts/chebi/chebi_classify_pablo_http.py`** and **`chebi_classify_pablo_hybrid_http.py`**
   - Currently: Probably obsolete HTTP variants
   - Recommended Action: Archive or delete if confirmed unused

4. **`dataset_cleaning.py`**
   - Currently: Test code snippets
   - Recommended Action: Delete or move to scratch folder

5. **`model/encoders/chebi_encoder.py`**
   - Currently: Conceptual notes only
   - Recommended Action: Implement if ontology embedding space learning is prioritized

---

## 10. Open Questions & Uncertainties

1. **CCS Replicate Handling**
   - Are there raw source datasets with CCS1, CCS2, CCS3 replicate columns that need averaging?
   - Is `check_ccs_replicates.py` already applied to all raw datasets?
   - Are results stored and used during the `build_final_covered_dataset.py` step?

2. **ChEBI Classification Execution**
   - Has `chebi_classify_pipeline.py` been executed to produce `predictions/chebi/chunks/results_pablo_hybrid_chunk*.json`?
   - If not, what is the current state of ChEBI matching?

3. **Base Model Training**
   - Are there existing training logs/artifacts for the base model in `model/artifacts/first_model/`?
   - What are the training hyperparameters and stopping criteria?

4. **Ontology Label Filtering**
   - What is the optimal `min_class_count` threshold? (Current default: 25)
   - How many multilabel classes result from this filtering?

5. **Benchmark Dependencies**
   - Are external DeepCCS and DarkChem repos available and properly installed?
   - Have benchmarks been run successfully?

6. **Dataset Consistency**
   - Are train/val/test splits identical between base model and ontology model training?
   - Is the random_state=42 consistently used across all splits?

---

## 11. Key Configuration Files & Constants

### Random State for Reproducibility
- **Value**: `random_state=42`
- **Used in**: `splitter.py`
- **Importance**: CRITICAL - Ensures identical train/val/test splits across model comparisons

### Splits: Train/Val/Test
- **Train**: 80% (~41-44k rows)
- **Val**: 10% (~5-5.5k rows)
- **Test**: 10% (~5-5.5k rows)
- **Total**: ~50-55k rows (after ChEBI filtering)

### CCS CV Threshold
- **Value**: 5% (default)
- **Used in**: `check_ccs_replicates.py`, `check_and_correct_metlin_ims_ccs.py`
- **Purpose**: Discard rows where CCS replicates have std/mean > 5%

### ChEBI Multilabel Filtering
- **Min class count**: 25 (configurable)
- **Purpose**: Exclude sparse ontology classes
- **Used in**: `prepare_chebi_multilabel_dataset.py`

### Ontology Loss Weight
- **Initial value**: λ = 0.1
- **Formula**: `total_loss = MSE_CCS + lambda_ontology * BCE_ontology`
- **Configurable in**: `chebi_model.py` training loop

### Fingerprint Pattern
- **Column names**: V1, V2, ..., V2211 (2,211 total)
- **Pattern regex**: `^V\d+$`
- **Used in**: Both models via `get_fingerprint_columns()`

---

## 12. Next Steps for Project Continuation

### Immediate (Critical)
1. ✓ Confirm all ChEBI classification results exist in `predictions/chebi/chunks/`
2. Execute Steps 1-5 of recommended workflow to prepare data
3. Train base model (Step 6) to establish reference performance
4. Implement `model/sanity_checks.py` for data validation
5. Document exact training hyperparameters and stopping criteria

### Short-term
1. Train ontology model (Step 7) and compare against base model
2. Run external benchmarks (Steps 8-9) on test set
3. Archive or document obsolete scripts (HTTP variants, dataset_cleaning.py)
4. Verify train/val/test split consistency across both models

### Medium-term
1. Implement embedding space visualization for latent ChEBI relationships
2. Experiment with different `min_class_count` thresholds for ontology labels
3. Tune `lambda_ontology` weight (investigate if 0.1 is optimal)
4. Write comprehensive training/evaluation scripts

### Long-term
1. Implement `model/encoders/chebi_encoder.py` if ontology embedding is prioritized
2. Consolidate `check_and_correct_metlin_ims_ccs.py` into general pipeline
3. Archive external model wrappers if benchmarking is complete
4. Document thesis results and finalize repository for publication

---

## Appendix: Data Schema Reference

### `final_covered_ccs_fingerprints.csv`
**Core Columns (10)**:
- row_id (integer)
- smiles (string)
- adduct (string, e.g., "[M+H]+")
- ccs (float)
- inchi (string)
- name (string)
- mz (float)
- source_dataset (string)
- chebi_classes (string, JSON array)
- chebi_count (integer)

**Fingerprint Columns (2,211)**:
- V1, V2, ..., V2211 (float, normalized fingerprint bits)

**Total**: 2,221 columns

---

### `chebi_ontology_labels.csv`
**Fixed Columns**:
- row_id (integer)
- source_dataset (string)
- chebi_classes (string, JSON)
- chebi_count (integer)
- chebi_name (string)
- chebi_match_source (string)

**Dynamic Multilabel Columns** (number depends on min_class_count):
- ontology__CHEBI:XXXXX (binary: 0 or 1, one column per filtered class)

**Summary Columns**:
- ontology_label_count (integer)
- ontology_has_labels (binary)

---

**Document Version**: 1.1 (Updated with new folder organization)  
**Last Reviewed**: May 13, 2026  
**Reviewed By**: Repository Analysis Tool
