# ChEBI_CCS


## 1) What This Repository Contains

This project covers five main components:

1. **CCS Data Preparation**:
	- cleaning and unification
	- reconstruction of subset covered by ChEBI
	- generation of final fingerprint dataset (no descriptors)
	- CCS replicate consistency check and averaging
	- ontology label binarization for multilabel training

2. **ChEBI Classification & Ontology Integration**:
	- local classification over `chebi.obo` (fast, no HTTP dependency)
	- traversal of ChEBI is_a hierarchy for ancestor extraction
	- multilabel ontology label generation with filtering

3. **Base Model** (regression only, without ontology):
	- training with 80/10/10 `train/val/test` split
	- fingerprint + adduct + m/z → CCS prediction
	- MSE loss, RMSE/MAE/R² metrics
	- reference baseline for comparison

4. **Ontology-Aware Model** (multitask neural network):
	- shared embedding space for both tasks
	- Task 1: CCS regression (MSELoss)
	- Task 2: ChEBI multilabel classification (BCEWithLogitsLoss)
	- combined loss: MSE + λ × BCE (default λ = 0.1)
	- latent embedding export for analysis

5. **Benchmark of External Models**:
	- DeepCCS and DarkChem (external repos)
	- aggregation of comparable metrics

## 2) Current Data Flow Status

### Base Model Pipeline

- **Input**: `data/model/final_covered_ccs_fingerprints.csv`
  - schema: `row_id, smiles, adduct, ccs, inchi, name, mz, V1..V2211`

- **Splits** (80/10/10 deterministic):
  - `data/model/train_ccs_fingerprints.csv` (~41-44k rows)
  - `data/model/val_ccs_fingerprints.csv` (~5-5.5k rows)
  - `data/model/test_ccs_fingerprints.csv` (~5-5.5k rows)
  - Manifest: `data/model/split_manifest.json`

### Ontology-Aware Model Pipeline

- **ChEBI Matches**: `data/model/final_covered_ccs.csv`
  - schema: + `chebi_classes` (JSON list), `chebi_count`, `chebi_name`, `chebi_match_source`
  - produced by: `chebi_classify_pipeline.py`

- **Ontology Labels**: `data/model/final_covered_ccs_fingerprints_multilabel.csv`
  - schema: base fingerprint columns + `ont_lipid`, `ont_steroid`, `ont_benzenes`, etc.
  - multilabel binary columns for each ChEBI ancestor
  - produced by: `prepare_chebi_multilabel_dataset.py`

- **Filtered Ontology Labels** (recommended for training): 
  - `data/model/final_covered_ccs_fingerprints_multilabel_filtered.csv`
  - removes generic/sparse ontology classes using configurable filters
  - produced by: `filter_ontology_multilabel_dataset.py`

- **Shared Splits**: Uses the same train/val/test splits as base model
  - ensures fair comparison between both models

## 3) Installation

### Important: Environment Location

The `tfg_amalia` conda environment is installed locally at:
```
C:\Users\amali\miniconda3\envs\tfg_amalia
```

**For detailed setup instructions, see [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)**

All training commands in this repository use this environment. If you're working on a different machine, you'll need to create or recreate this environment.

### Quick Start: Option A (Existing Environment)

```bash
conda activate tfg_amalia
```

Verify the environment is active (prompt should show `(tfg_amalia)`).

### Quick Start: Option B (New Setup from YAML)

This creates a fresh environment with all dependencies:

```bash
conda env create -f environment.yml
conda activate tfg_amalia
```

**Expected output:**
```
Collecting package metadata (repodata.json): done
Solving environment: done
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate tfg_amalia
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

### Quick Start: Option C (pip-only setup)

If you prefer pip-only installation or have conda issues:

```bash
conda create -n tfg_amalia python=3.12
conda activate tfg_amalia
pip install -r requirements.txt
```

### Verify Installation

After activation, verify all packages are installed:

```bash
python -c "import torch; import pandas; import rdkit; print('✓ All core packages installed')"
```

Expected output:
```
✓ All core packages installed
```

### GPU Verification (Optional)

To verify PyTorch GPU support:

```bash
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

## 4) Environment Dependencies

### Core Packages Included

| Package | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.12 | Base runtime |
| **NumPy** | ≥2.4.3 | Numerical computing |
| **Pandas** | ≥3.0.1 | Data manipulation |
| **Scikit-Learn** | ≥1.8.0 | ML algorithms |
| **SciPy** | ≥1.17.1 | Scientific computing |
| **PyTorch** | 2.9.1 | Deep learning framework (CUDA 12.1) |
| **PyTorch Vision** | 0.24.1 | Computer vision utilities |
| **RDKit** | 2025.9.6 | Cheminformatics (SMILES, InChI) |
| **Matplotlib** | ≥3.10.9 | Plotting & visualization |
| **Seaborn** | ≥0.13.2 | Statistical visualization |
| **Pillow** | ≥12.1.1 | Image processing |
| **UMAP-Learn** | ≥0.5.11 | Dimensionality reduction (optional) |
| **PyYAML** | ≥6.0.3 | YAML configuration parsing |
| **Requests** | ≥2.32.5 | HTTP requests |
| **NetworkX** | ≥3.6.1 | Graph algorithms |
| **Jupyter** | ≥1.0.0 | Interactive notebooks |

### GPU Support

The environment includes **PyTorch with CUDA 12.1** support for GPU acceleration.

- **Device**: GPU support for both NVIDIA (CUDA) and AMD (ROCm) architectures
- **Memory**: For large datasets, GPU acceleration is recommended
- **Alternative**: CPU-only mode works but is significantly slower

To use CPU-only, modify the PyTorch installation:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 5) Relevant Project Structure

### Essential Scripts (Both Models)

**Data Management** (`model/scripts/data_management/`):
- `splitter.py`: Create deterministic train/val/test splits (80/10/10)
- `build_final_covered_dataset.py`: Merge ChEBI chunks into final dataset
- `build_final_fingerprint_dataset.py`: Append fingerprint vectors to covered dataset
- `check_and_correct_metlin_ims_ccs.py`: Validate CCS replicate data (CV% threshold)
- `check_ccs_replicates.py`: Detect and handle replicate CCS columns in raw data

**ChEBI Classification** (`model/scripts/chebi/`):
- `chebi_classify.py`: Local ChEBI classifier (rdkit + chebi.obo, no HTTP)
- `chebi_classify_pipeline.py`: Orchestrates classification pipeline
- `prepare_chebi_multilabel_dataset.py`: Create binary ontology labels from ChEBI matches
- `filter_ontology_multilabel_dataset.py`: Filter ontology labels (min count, blacklist, etc.)

**Model Training** (`model/`):
- `base_model.py`: Baseline CCS regression model
- `chebi_model.py`: Ontology-aware multitask model

### Supporting Files

**Requirements**:
- `assets/requirements/pipeline.txt`: Data preparation dependencies
- `assets/requirements/base_model.txt`: Model training dependencies
- `benchmark/requirements.txt`: External model benchmarking

**Benchmark**:
- `benchmark/scripts/run_benchmark.py`: Run external model predictions
- `benchmark/scripts/aggregate_metrics.py`: Aggregate benchmark metrics

## 6) Recommended End-to-End Workflow

### 6.1 Validate CCS Replicates in Raw Datasets

```
conda run -n tfg_amalia python model/scripts/data_management/check_ccs_replicates.py \
  --input-dir data/raw_datasets \
  --output-dir data/clean_datasets/ccs_replicate_check \
  --cv-threshold 5
```

Output: Reports per dataset with CV% statistics. Identifies rows to discard (CV > 5%).

### 6.2 Reconstruct Covered Dataset with ChEBI Matches

```
conda run -n tfg_amalia python model/scripts/data_management/build_final_covered_dataset.py
```

Input: `data/unified/unified_ccs.csv` + ChEBI chunk results  
Output: `data/model/final_covered_ccs.csv` (with ChEBI metadata)

### 6.3 Build Final Fingerprint Dataset (no Descriptors)

```
conda run -n tfg_amalia python model/scripts/data_management/build_final_fingerprint_dataset.py
```

Input: `data/model/final_covered_ccs.csv` + raw fingerprint tables  
Output: `data/model/final_covered_ccs_fingerprints.csv` (with V1..Vn columns)

### 6.4 Create Deterministic 80/10/10 Splits

```
conda run -n tfg_amalia python model/scripts/data_management/splitter.py
```

Input: `data/model/final_covered_ccs_fingerprints.csv`  
Output: 
- `train_ccs_fingerprints.csv` (80%)
- `val_ccs_fingerprints.csv` (10%)
- `test_ccs_fingerprints.csv` (10%)
- `split_manifest.json` (metadata)

### 6.5 Build Ontology Labels from ChEBI

**Step A: Create multilabel dataset with ChEBI ancestors**

```
conda run -n tfg_amalia python model/scripts/chebi/prepare_chebi_multilabel_dataset.py \
  --chunks-dir predictions/chebi/chunks \
  --ontology-obo data/ontology/chebi.obo \
  --fingerprint-csv data/model/final_covered_ccs_fingerprints.csv \
  --output-csv data/model/final_covered_ccs_fingerprints_multilabel_all_ancestors.csv \
  --manifest-json data/model/ontology_label_manifest_all_ancestors.json
```

**Step B: Filter ontology labels** (recommended for better generalization)

```
conda run -n tfg_amalia python model/scripts/chebi/filter_ontology_multilabel_dataset.py \
  --input-csv data/model/final_covered_ccs_fingerprints_multilabel_all_ancestors.csv \
  --input-manifest data/model/ontology_label_manifest_all_ancestors.json \
  --output-csv data/model/final_covered_ccs_fingerprints_multilabel_filtered.csv \
  --output-manifest data/model/ontology_label_manifest_filtered.json \
  --min-class-count 30 \
  --max-frequency-ratio 0.9
```

Output: Binary multilabel columns (ont_*) with rare/generic classes removed.

### 6.6 Train Base Model (Baseline)

```
conda run -n tfg_amalia python model/base_model.py \
  --train-input data/model/train_ccs_fingerprints.csv \
  --val-input data/model/val_ccs_fingerprints.csv \
  --test-input data/model/test_ccs_fingerprints.csv \
  --output-dir predictions/base \
  --epochs 30 \
  --batch-size 128 \
  --lr 1e-3
```

Expected outputs in `predictions/base`:
- `training_summary.json`: Per-epoch metrics (loss, RMSE, MAE)
- `training_curves.png`: Loss curves
- `test_predictions.csv`: Final test set predictions
- `train_split.csv`, `val_split.csv`, `test_split.csv`: Split info

### 6.7 Train Ontology-Aware Model

```
conda run -n tfg_amalia python model/scripts/classification_model/run_multitask_train.py \
  --train-input data/model/train_ccs_fingerprints.csv \
  --val-input data/model/val_ccs_fingerprints.csv \
  --test-input data/model/test_ccs_fingerprints.csv \
  --ontology-input data/model/final_covered_ccs_fingerprints_multilabel_filtered.csv \
  --output-dir predictions/ontology_model \
  --epochs 30 \
  --batch-size 128 \
  --lr 1e-3 \
  --lambda-ontology 0.1 \
  --ontology-threshold 0.5
```

The ontology model:
- Uses **exact same train/val/test splits** as base model (row_id matching)
- Merges ontology labels by row_id
- Trains on: fingerprints + adduct + m/z + ontology labels (during training)
- Total loss: `MSE_ccs + 0.1 × BCE_ontology`

Expected outputs in `predictions/ontology_model`:
- `training_summary.json`: Per-epoch metrics (CCS loss, ontology loss, CCS RMSE/MAE)
- `ontology_metrics.json`: Ontology prediction metrics (F1, Hamming, subset accuracy, etc.)
- `training_curves.png`: Loss curves for both tasks
- `embeddings_test.csv`: Latent embeddings (64-dim) from shared fc3 layer
- `test_predictions_full.csv`: All ontology predictions with logits
- `test_predictions_clean.csv`: Binary ontology predictions (threshold 0.5)
- `test_predictions_readable.csv`: Human-readable format with labels
- `train_split.csv`, `val_split.csv`, `test_split.csv`: Split info

### 6.8 Post-Training Analysis

**Visualize embeddings with PCA**

```
conda run -n tfg_amalia python model/scripts/chebi/visualize_embeddings.py \
  --embeddings-csv predictions/ontology_model/embeddings_test.csv \
  --test-predictions-csv predictions/ontology_model/test_predictions_clean.csv \
  --output-dir predictions/ontology_model/embedding_plots
```

**Optional: Visualize with UMAP** (requires `umap-learn` or `cuml`)

```
conda run -n tfg_amalia python model/scripts/chebi/visualize_embeddings.py \
  --embeddings-csv predictions/ontology_model/embeddings_test.csv \
  --test-predictions-csv predictions/ontology_model/test_predictions_clean.csv \
  --output-dir predictions/ontology_model/embedding_plots_umap \
  --use-umap
```

## 7) Base Model Metrics

The `training_summary.json` stores per-epoch and final metrics:

**Per epoch** (in `history`):
- `train_loss`, `val_loss`: MSE loss
- `train_rmse`, `val_rmse`: Root mean squared error
- `train_mae`, `val_mae`: Mean absolute error

**Final metrics** (train/val/test):
- RMSE: Root mean squared error
- MAE: Mean absolute error
- MEDAE: Median absolute error
- R²: Coefficient of determination

## 8) Ontology-Aware Model Metrics

The `ontology_metrics.json` stores:

**Per-task metrics** (CCS regression):
- RMSE, MAE, MEDAE, R² (same as base model)

**Multilabel classification metrics**:
- Exact match ratio (subset accuracy)
- Hamming loss
- Macro F1 score
- Micro F1 score
- Sample F1 score

**Threshold analysis**:
- Metrics computed at threshold 0.5 (configurable)

## 9) Benchmark (DeepCCS / DarkChem)

1. Configure external repos and environments according to:
	- `benchmark/README.md`
	- `benchmark/configs/benchmark_models.yaml`

2. Run benchmark:

```
cd benchmark
python scripts/run_benchmark.py \
  --input ../data/model/test_ccs_fingerprints.csv \
  --config configs/benchmark_models.yaml
```

3. Aggregate metrics:

```
python scripts/aggregate_metrics.py \
  --input ../data/model/test_ccs_fingerprints.csv \
  --predictions-dir predictions \
  --output-metrics reports/metrics.csv
```

## 10) Quick Reference: Critical Files

| File | Purpose | Owner |
|------|---------|-------|
| `data/model/final_covered_ccs_fingerprints.csv` | Dataset with fingerprints | Both models |
| `data/model/train_ccs_fingerprints.csv` | Training split (80%) | Both models |
| `data/model/val_ccs_fingerprints.csv` | Validation split (10%) | Both models |
| `data/model/test_ccs_fingerprints.csv` | Test split (10%) | Both models |
| `data/model/final_covered_ccs_fingerprints_multilabel_filtered.csv` | Ontology labels | Ontology model only |
| `predictions/base/` | Base model artifacts | Baseline |
| `predictions/ontology_model/` | Ontology model artifacts | Ontology-aware |

## 11) Environment Notes

Use the conda environment: **`tfg_amalia`**

All commands in this README use `conda run -n tfg_amalia python ...` for reproducibility.

To activate manually:
```
conda activate tfg_amalia
```
