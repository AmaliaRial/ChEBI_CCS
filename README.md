# ChEBI_CCS


## 1) What This Repository Contains

This project covers four main components:

1. CCS Data Preparation:
	- cleaning and unification
	- reconstruction of subset covered by ChEBI
	- generation of final fingerprint dataset (no descriptors)
	- CCS replicate consistency check and averaging
	- ontology label binarization for multilabel training

2. ChEBI Classification:
	- local classification over `chebi.obo`
	- HTTP/hybrid variants for additional coverage

3. Base Model (without ontology as explicit features):
	- training with 80/10/10 `train/val/test` split
	- regression metrics
	- training/validation curves

4. Benchmark of External Models:
	- DeepCCS and DarkChem (external repos)
	- aggregation of comparable metrics

## 2) Current Data Flow Status

- Base model main dataset:
  - `data/model/final_covered_ccs_fingerprints.csv`
  - base schema: `row_id, smiles, adduct, ccs, inchi, name, mz, V1..Vn`

- Splits:
  - `data/model/train_ccs_fingerprints.csv`
  - `data/model/val_ccs_fingerprints.csv`
  - `data/model/test_ccs_fingerprints.csv`

## 3) Installation

### Option A: Use Existing `chebi_ccs` Environment

```
conda activate chebi_ccs
pip install -r assets/requirements/pipeline.txt
pip install -r assets/requirements/base_model.txt
pip install -r benchmark/requirements.txt
```

### Option B: Create Environment from `environment.yml`

```
conda env create -f environment.yml
conda activate chebi_ccs
```

## 4) Relevant Project Structure

- `model/base_model.py`: base model training
- `model/chebi_model.py`: ontology-aware multitask model training
- `model/encoders/`: adduct encoder
- `model/scripts/chebi/chebi_classify_pipeline.py`: local ChEBI classification pipeline
- `model/scripts/data_management/prepare_chebi_multilabel_dataset.py`: convert ChEBI lists into binary ontology labels
- `model/scripts/chebi/check_ccs_replicates.py`: inspect CCS replicate columns and filter by CV
- `benchmark/scripts/run_benchmark.py`: runs benchmark wrappers
- `benchmark/scripts/aggregate_metrics.py`: aggregates benchmark metrics
- `assets/requirements/`: per-component requirements

## 5) Recommended End-to-End Workflow

### 5.1 Reconstruct Covered Dataset with ChEBI

```
conda run -n chebi_ccs python model/scripts/chebi/build_final_covered_dataset.py
```

### 5.2 Build Final Fingerprint Dataset (no Descriptors)

```
conda run -n chebi_ccs python model/scripts/chebi/build_final_fingerprint_dataset.py
```

### 5.3 Create 80/10/10 Splits

```
conda run -n chebi_ccs python model/scripts/data_management/splitter.py
```

### 5.4 Build Ontology Labels from ChEBI

``` 
conda run -n chebi_ccs python model/scripts/data_management/prepare_chebi_multilabel_dataset.py --min-class-count 30
```

This creates a compact multilabel table keyed by `row_id` with `ontology__*` columns.

### 5.5 Check CCS Replicates in Raw Datasets

``` 
conda run -n chebi_ccs python model/scripts/chebi/check_ccs_replicates.py --input-dir data/raw_datasets --output-dir data/clean_datasets/ccs_replicate_check --cv-threshold 5
```

Accepted and discarded rows are written separately, together with `ccs_average`, `ccs_std` and `ccs_cv_percent`.

### 5.6 Train Base Model

```
conda run -n chebi_ccs python model/base_model.py --epochs 30 --batch-size 128 --output-dir predictions/base
```

Expected outputs in `predictions/base`:
- `training_summary.json`
- `training_curves.png`
- `test_predictions.csv`
- `train_split.csv`, `val_split.csv`, `test_split.csv`

### 5.7 Train Ontology-Aware Model

``` 
conda run -n chebi_ccs python model/chebi_model.py --train-input data/model/train_ccs_fingerprints.csv --val-input data/model/val_ccs_fingerprints.csv --test-input data/model/test_ccs_fingerprints.csv --ontology-input data/model/chebi_ontology_labels.csv --output-dir predictions/chebi --lambda-ontology 0.1
```

The ontology trainer merges labels by `row_id`, so it uses the exact same train/val/test splits as the baseline model.

Expected outputs in `predictions/chebi`:
- `training_summary.json`
- `ontology_metrics.json`
- `training_curves.png`
- `test_predictions.csv`
- `embeddings_test.csv`
- `train_split.csv`, `val_split.csv`, `test_split.csv`

## 6) Base Model Metrics

The `training_summary.json` file stores:

- per epoch (`history`):
  - `train_loss`, `val_loss`
  - `train_rmse`, `val_rmse`
  - `train_mae`, `val_mae`

- final metrics for train/val/test:
  - RMSE
  - MAE
  - MEDAE
  - $R^2$

## 7) Benchmark (DeepCCS / DarkChem)

1. Configure external repos and environments according to:
	- `benchmark/README.md`
	- `benchmark/configs/benchmark_models.yaml`

2. Run benchmark:

```
cd benchmark
python scripts/run_benchmark.py --input ..\data\model\test_ccs_fingerprints.csv --config configs/benchmark_models.yaml
```

3. Aggregate metrics:

```
python scripts/aggregate_metrics.py --input ..\data\model\test_ccs_fingerprints.csv --predictions-dir predictions --output-metrics reports/metrics.csv
```