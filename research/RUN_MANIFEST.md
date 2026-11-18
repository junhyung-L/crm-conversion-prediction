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
The original dependency versions are not stored in this repository, so version confirmation remains user input required.
