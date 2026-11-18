# Reproduction Manifest

## Baseline contract

The refactor must preserve the historical experiment definition:

- Seed: `1`
- Split: customer-level 70/15/15 (`split_utils.customer_split`), stratified by whether a customer ever converts
- Validation: five-fold `StratifiedGroupKFold` grouped by `client_id` (paper protocol)
- Primary dataset: `final_data_100k_64.parquet`
- Primary result report: `predict/result_xgb.json`

`verify_baseline.py` compares the retained XGBoost report with its saved metric contract.
It is a report-regression test, not a model rerun.

## Run with the original data

Set paths explicitly; do not copy data into the repository:

```powershell
$env:MFA_DATASET_PATH = 'D:\path\to\final_data_100k_64.parquet'
$env:MFA_RESULTS_DIR = 'D:\path\to\new-results'
python research\xgb.py
python research\verify_baseline.py --result "$env:MFA_RESULTS_DIR\result_xgb.json"
```

An exact match can require a matched Python/library environment and the same hardware settings.
The experiments ran on Python 3.12.11 (notebook metadata, conda env `ljh_312`). Their package
versions were not recorded; `requirements.txt` pins the 2026-11 review environment (Python 3.13.2),
in which `verify_baseline.py` passes. `pytorch-tabnet` was not installed there and stays unpinned.

## Top-K targeting scripts

- `predict/compute_tau_targeting_metrics.py` is the script `experiments.ipynb` ran to produce the
  stored `predict/tau_sweep_results/tau_targeting_metrics.{json,csv}`.
- `compute_tau_targeting_metrics.py` (this folder) is the maintained version that uses
  `split_utils` (customer-level split, five grouped folds). Set the data and result paths
  (`research/tau_sweep_data`, `research/tau_sweep_results`) before running it.

The backup `result_xgb_1_00x.invalid_20260813_230614.json` (a broken 1.00x result renamed before
the rerun, see `experiments.ipynb`) was removed in the 2026-11 review; `result_xgb_1_00x.json` is
the valid rerun.
