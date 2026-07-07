# ChEBI_CCS: CCS Prediction with ChEBI Ontology Integration

A multitask deep learning framework for predicting Collision Cross Section (CCS) values with optional ChEBI chemical ontology supervision.

## Overview

This repository contains two trainable models:

1. **Base Model**: CCS regression only (baseline)
   - Input: molecular fingerprints + adduct + m/z
   - Output: CCS prediction
   - Loss: MSE

2. **Ontology-Aware Model**: Multitask learning
   - Input: fingerprints + adduct + m/z + ChEBI ontology labels
   - Output: CCS prediction + multilabel classification
   - Loss: MSE + λ × BCE (λ defaults to 0.1)
   - Supports λ ∈ {0.1 - 2.0} for comparison

Both models use identical 80/10/10 train/val/test splits on **16,892 molecules** (ChEBI-covered compounds) with **2,259 input features** and **546 ontology labels** (filtered).

## Data Files

### Ready-to-Use Datasets

All datasets are pre-processed and available in `data/model/`:

| File | Size | Columns | Used By |
|------|------|---------|---------|
| `final_covered_ccs_fingerprints.csv` | 16,892 rows | fingerprints (V1–V2048) + adduct + ccs + m/z | Reference dataset |
| `final_covered_ccs_fingerprints_multilabel_filtered.csv` | 16,892 rows | above + 546 binary ontology labels | Ontology model training |
| `ontology_label_manifest_filtered.json` | — | metadata for 546 ontology classes | Reference |
| `split_manifest.json` | — | split indices and random_state | Reference |

### Train/Val/Test Splits

Located in `predictions/base/`:

| File | Rows | Purpose |
|------|------|---------|
| `train_split.csv` | 13,514 (80%) | Training |
| `val_split.csv` | 1,689 (10%) | Validation |
| `test_split.csv` | 1,689 (10%) | Testing |

**Schema for all CSVs:**
- `row_id`: Unique identifier
- `smiles`: SMILES string
- `adduct`: Ionization adduct (e.g., "[M+H]+")
- `ccs`: CCS target value 
- `m/z`: Mass-to-charge ratio
- `V1` to `V2048`: Morgan fingerprint features (radius 2, 2048 bits)
- `ontology_true__ont_*`: Binary ontology labels (546 columns in multilabel CSV)

## Installation

### 1. Clone & Navigate

```bash
git clone <repository_url>
cd ChEBI_CCS
```

### 2. Setup Environment

**Option A: Use conda (recommended)**

```bash
conda env create -f environment.yml
conda activate tfg_amalia
```

**Option B: Use pip**

```bash
conda create -n tfg_amalia python=3.12
conda activate tfg_amalia
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; import pandas; import rdkit; print('✓ Ready')"
```

**GPU Check** (optional):
```bash
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

### Key Dependencies

- Python 3.12
- PyTorch 2.9.1 (CUDA 12.1)
- RDKit 2025.9.6
- Pandas, NumPy, scikit-learn
- See `environment.yml` for full list

## Quick Start: Training Models

All datasets are pre-processed and ready to use. No data preparation needed.

### Train Base Model (CCS Regression Only)

```bash
conda activate tfg_amalia

python -m model.base_model \
  --train-input predictions/base/train_split.csv \
  --val-input predictions/base/val_split.csv \
  --test-input predictions/base/test_split.csv \
  --output-dir predictions/baseline_model \
  --epochs 65 \
  --batch-size 128 \
  --lr 0.0008733543414433369
```

**Inputs:**
- `predictions/base/train_split.csv` (13,514 rows)
- `predictions/base/val_split.csv` (1,689 rows)
- `predictions/base/test_split.csv` (1,689 rows)

**Outputs in** `predictions/base_new/`:
- `training_summary.json` — Per-epoch metrics (loss, RMSE, MAE, R²)
- `training_curves.png` — Loss curves
- `test_predictions.csv` — Predictions on test set

---

### Train Ontology-Aware Model

```bash
conda activate tfg_amalia

python -m model.scripts.classification_model.run_multitask_train \
  --train-input predictions/base/train_split.csv \
  --val-input predictions/base/val_split.csv \
  --test-input predictions/base/test_split.csv \
  --ontology-input data/model/final_covered_ccs_fingerprints_multilabel_filtered.csv \
  --output-dir predictions/final_ontology_lambda_1.8 \
  --epochs 65 \
  --batch-size 128 \
  --lr 0.0008733543414433369 \
  --hidden-dims 320 64 160 384 480 \
  --dropout 0.0043597782845490735 \
  --lambda-ontology 1.8 \
  --ontology-threshold 0.5 \
  --device cuda
```

**Inputs:**
- Base splits: same as above
- Ontology labels: `data/model/final_covered_ccs_fingerprints_multilabel_filtered.csv` (546 binary labels)

**Outputs in** `predictions/ontology_model_new/`:
- `training_summary.json` — Per-epoch metrics (CCS & ontology loss)
- `ontology_metrics.json` — Classification metrics (F1, Hamming, subset accuracy)
- `training_curves.png` — Loss curves for both tasks
- `test_predictions_full.csv` — CCS predictions + ontology logits & probabilities
- `embeddings_test.csv` — Latent embeddings (64-dim) from shared layer

**Key Parameters:**
- `--lambda-ontology`: Weight for ontology loss (0.1, 0.5, or 1.0)
- `--ontology-threshold`: Classification threshold for multilabel (default 0.5)
- `--device`: "auto" (GPU if available), "cpu", or "cuda"

---

### Compare Results

Final Pre-trained models available in:
- `predictions/final_baseline_optimized/` — Baseline (λ = N/A)
- `predictions/final_ontology_optimized/` — λ = 1.8
- `predictions/lambda_sweep/` — λ = (0.1, 2.0)
- `predictions/final_benchmark_test/` — Benchmark values on test split

All use identical train/val/test splits. Compare `test_predictions.csv` or `ontology_metrics.json` across directories.


## Output Metrics & Files

### Base Model Outputs

**training_summary.json** (per-epoch):
```json
{
  "history": {
    "train_loss": [...],
    "val_loss": [...],
    "train_rmse": [...],
    "val_rmse": [...]
  },
  "test_metrics": {
    "rmse": 12.34,
    "mae": 9.56,
    "medae": 8.12,
    "r2": 0.892
  }
}
```

**test_predictions.csv**:
- `row_id`, `CCS_true`, `CCS_pred`, `error`

---

### Ontology Model Outputs

**training_summary.json**:
```json
{
  "history": {
    "train_ccs_loss": [...],
    "train_ontology_loss": [...],
    "val_ccs_loss": [...],
    "val_ontology_loss": [...]
  },
  "test_ccs_metrics": {
    "rmse": 12.34,
    "mae": 9.56,
    "r2": 0.892
  }
}
```

**ontology_metrics.json**:
```json
{
  "ontology_metrics": {
    "exact_match_ratio": 0.234,
    "hamming_loss": 0.156,
    "f1_macro": 0.567,
    "f1_micro": 0.678
  },
  "label_columns": 546,
  "threshold": 0.5
}
```

**test_predictions_full.csv**:
- `row_id`, `CCS_true`, `CCS_pred`
- `ontology_true__ont_*`: Ground truth (0/1)
- `ontology_logit__ont_*`: Model logits
- `ontology_prob__ont_*`: Probabilities (sigmoid of logits)

**embeddings_test.csv**:
- `row_id`, `embedding_0` to `embedding_63` (64-dim latent space)

## Project Structure

```
data/
├── model/
│   ├── final_covered_ccs_fingerprints.csv                    [Main dataset, 16,892 × 2,259]
│   ├── final_covered_ccs_fingerprints_multilabel_filtered.csv [+ 546 ontology labels]
│   ├── ontology_label_manifest_filtered.json                 [Ontology metadata]
│   └── split_manifest.json                                   [Split info]
│
└── ontology/
    └── chebi.obo                                             [ChEBI hierarchy]

model/
├── base_model.py                      [Baseline model]
├── chebi_model.py                     [Ontology-aware model]
└── scripts/
    └── classification_model/
        └── run_multitask_train.py     [CLI for ontology model]

predictions/
├── final_baseline_optimized/                              [Pre-trained baseline]
│   ├── train_split.csv
│   ├── val_split.csv
│   ├── test_split.csv
│   ├── training_summary.json
│   ├── training_curves.png
│   └── test_predictions.csv
│
├── final_ontology_optimized/                    [Pre-trained ontology model, λ=1.8]
├── final_benchmark_test
```
