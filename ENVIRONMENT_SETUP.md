# Environment Setup Guide

## Quick Reference

**Environment Name**: `tfg_amalia`

**Local Location (Windows)**:
```
C:\Users\amali\miniconda3\envs\tfg_amalia
```

**Python Version**: 3.12

---

## Installation Methods

### Method 1: Create from `environment.yml` (Recommended)

This is the **standard** way to set up the project with all dependencies:

```bash
cd c:\Users\amali\repos\ChEBI_CCS
conda env create -f environment.yml
conda activate tfg_amalia
```

**Time to Complete**: ~5-10 minutes (depends on internet speed)

### Method 2: Update Existing Environment

If you already have `tfg_amalia` but want to update packages:

```bash
conda activate tfg_amalia
conda env update -f environment.yml --prune
```

### Method 3: Create with `requirements.txt` (pip-only)

```bash
conda create -n tfg_amalia python=3.12
conda activate tfg_amalia
pip install -r requirements.txt
```

**Note**: This approach may have version conflicts with conda-installed packages.

### Method 4: Activate Existing Local Environment

If the environment is already installed at `C:\Users\amali\miniconda3\envs\tfg_amalia`:

```bash
conda activate tfg_amalia
```

---

## Verify Installation

### Check Conda Environment

```bash
conda env list
```

**Expected Output**:
```
# conda environments:
#
base                     C:\Users\amali\miniconda3
tfg_amalia             * C:\Users\amali\miniconda3\envs\tfg_amalia
```
(asterisk * indicates active environment)

### Check Python & Core Packages

```bash
python --version
pip list | grep -E "torch|pandas|rdkit|scikit-learn"
```

**Expected Output**:
```
Python 3.12.13
pandas                3.0.1
rdkit                 2025.9.6
scikit-learn          1.8.0
torch                 2.9.1+rocm7.2.1
```

### Check GPU Support (If Available)

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Key Dependencies

### Essential for Training

| Package | Purpose |
|---------|---------|
| **torch** | Deep learning framework |
| **pandas** | Data manipulation |
| **rdkit** | Cheminformatics (SMILES/InChI parsing) |
| **scikit-learn** | ML algorithms |
| **numpy** | Numerical computing |

### Optional for Analysis

| Package | Purpose |
|---------|---------|
| **umap-learn** | Dimensionality reduction for embeddings |
| **matplotlib** | Plotting |
| **seaborn** | Statistical visualization |
| **jupyter** | Interactive notebooks |

---

## Troubleshooting

### Issue: "conda: command not found"

**Solution**: 
- Ensure miniconda/anaconda is installed
- Add conda to PATH: `C:\Users\amali\miniconda3\Scripts`

### Issue: "ImportError: No module named 'torch'"

**Solution**:
```bash
conda activate tfg_amalia
pip install torch==2.9.1
```

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size or switch to CPU:
```bash
# Run with CPU
CUDA_VISIBLE_DEVICES="" python model/base_model.py --device cpu
```

### Issue: Environment not found

**Solution**: Recreate it:
```bash
conda env create -f environment.yml -n tfg_amalia --force-reinstall
```

---

## Running Training Commands

After activation, all commands use the standard format:

```bash
conda run -n tfg_amalia python script.py [args]
```

Or with direct activation:

```bash
conda activate tfg_amalia
python script.py [args]
```

---

## Conda Useful Commands

### List All Environments

```bash
conda env list
```

### Remove Environment

```bash
conda env remove -n tfg_amalia
```

### Export Current Environment

```bash
conda env export -n tfg_amalia > tfg_amalia_export.yml
```

### Deactivate Current Environment

```bash
conda deactivate
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `environment.yml` | Full environment definition (recommended for setup) |
| `requirements.txt` | pip-only package list (alternative method) |
| `assets/requirements/base_model.txt` | Base model specific dependencies |
| `assets/requirements/pipeline.txt` | Data pipeline dependencies |
| `benchmark/requirements.txt` | Benchmark script dependencies |

---

## Notes for Contributors

- **Always test** installation on a clean environment before committing changes
- **Update `environment.yml`** when adding new dependencies: `conda env export -n tfg_amalia > environment.yml`
- **Keep Python version at 3.12** for compatibility with rdkit and pytorch
- **GPU support**: Environment includes CUDA 12.1; CPU fallback is automatic
