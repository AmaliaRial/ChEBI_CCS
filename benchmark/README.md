# CCS Benchmark Tool

Run CCS predictions with DeepCCS and DarkChem.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create/update conda environments (safe way):
```bash
# Check existing envs first
conda env list

# Only create if missing
conda create -n deepccs python=3.8 -y
conda create -n darkchem python=3.9 -y

# Install/update DeepCCS env
conda run -n deepccs pip install numpy pandas scikit-learn tensorflow==2.13.0 keras==2.13.1
conda run -n deepccs pip install -e C:/Users/amali/repos/external/DeepCCS/core

# Install/update DarkChem env
conda run -n darkchem pip install numpy pandas scipy scikit-learn rdkit tensorflow==2.13.0 keras==2.13.1
conda run -n darkchem pip install -e C:/Users/amali/repos/external/darkchem
```

If an env already exists, skip `conda create ...` and just run the `conda run -n ...` install/update lines.

If your repos are in another folder, replace `C:/Users/amali/repos/external/...` with your real path.


3. Edit config:
Open `configs/benchmark_models.yaml`:
- Set `external_root` to your repos folder (e.g., `C:/repos/external`)
- Update paths if needed
- For DarkChem: point `network_dir` to your trained model folder

4. Run:
```bash
python scripts/run_benchmark.py --input your_data.csv --config configs/benchmark_models.yaml
```

## Input Format

Your CSV needs:
- `smiles` column
- `adduct` column (e.g., [M+H]+, [M-H]-)
- `ccs` column (true values)

## Output

Predictions go to:
- `predictions/deepccs/predictions.csv`
- `predictions/darkchem/predictions.csv`

Each has: `_row_id`, `smiles`, `adduct`, `predicted_ccs`

## Get Metrics

Overall:
```bash
python scripts/aggregate_metrics.py \
  --input your_data.csv \
  --predictions-dir predictions \
  --output-metrics reports/metrics.csv
```

By dataset (if you have `source_dataset` column):
```bash
python scripts/aggregate_metrics.py \
  --input your_data.csv \
  --predictions-dir predictions \
  --output-metrics reports/metrics_by_dataset.csv \
  --by-dataset
```

Metrics includes:
- `n`: sample count
- `mae`: mean absolute error
- `rmse`: root mean squared error
- `mpe`: mean percentage error
- `mape`: mean absolute percentage error
- `std_abs_error`: standard deviation of absolute errors
- `std_pct_error`: standard deviation of percentage errors
- `outliers_gt_10pct`: count of predictions >10% off

## Config Details

`benchmark_models.yaml` structure:
```yaml
external_root: "/path/to/your/repos"
models:
  deepccs:
    enabled: true
    repo_path: "DeepCCS"
    conda_env: "deepccs"
  
  darkchem:
    enabled: true
    repo_path: "darkchem"
    conda_env: "darkchem"
    wrapper_args:
      network_dir: "darkchem-weights"
      property_index: 1
```

**DarkChem notes:**
- `network_dir`: folder with `arguments.txt` file (required)
- `property_index`: which property column to use as CCS (usually 1)

**DeepCCS notes:**
- Only supports: M+H, M+Na, M-H, M-2H adducts
- Filters out SMILES with unsupported tokens

## Troubleshooting

**"Conda environment not found"**
- Make sure you created the envs and they match the names in config

**"Network not found"** (DarkChem)
- Check `network_dir` points to folder with `arguments.txt`
- Can be absolute path or relative to repo

**Wrong predictions** (DarkChem)
- Try `property_index: 0` or `property_index: 1` in config
- Check model output to see which column is CCS

**Import errors**
- Activate the env and run `pip install -e .` in the repo folder
