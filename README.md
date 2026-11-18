# A Data-Centric AI Framework for E-Commerce Purchase Conversion Prediction via Dynamic Marketing Knowledge Integration

[한국어](README.ko.md) · [Case study](PORTFOLIO.md)

**Junhyung Lee** (first author), Kyungwon Kim (corresponding). *마케팅연구* (Korean Marketing Association), forthcoming November 2026.

Predicts which CRM message recipients will buy within 24 hours, when only
0.1234% of messages convert. Four groups of features derived from marketing
theory (purchase cycle, temporal response, marketing fatigue, channel
effectiveness) are added to a baseline and compared across nine models on
172 million REES46 messages.

**Result.** XGBoost with the domain features had the best F1 (0.3806), F2
(0.4549), and PR-AUC (0.3620) of the nine models on the original-distribution
holdout; recall rose from 0.3074 to 0.5230 over the baseline features.
Contacting the top 0.78% of messages captured 95.83% of conversions (precision
15.26%, lift 123.7x); the top 10,000 reached 53.20% precision (lift 431.1x).

**Scope.** Offline holdout on one platform's observational data: predictive
contribution, not causal effect. Data not included. Headline numbers are the
paper's (Tables 3 and 7, full test set); the stored
`research/predict/result_xgb.json` is the 10M-row sample run behind the
half-life table (Table 6: PR-AUC 0.3534, F1 0.3833), which
`python research\verify_baseline.py` checks without data.

## Research question

Can theory-based dynamic features improve conversion prediction more than
model complexity does, when the positive class is extremely rare?

| Feature group | Theory | Example signals |
|---|---|---|
| ① Purchase and repurchase cycle | Customer lifetime value, cooling-off effect | Days since last purchase, repurchase hazard, post-purchase refractory period |
| ② Temporal response | Context-dependent preference, calendar effects | Preferred hour/day match, payday and quarter-end proximity |
| ③ Marketing fatigue | Habituation | Exponentially decayed exposure (half-life 24h push, 48h email, 72h SMS), 30-day open and click rates |
| ④ Channel effectiveness | Information entropy, path dependency | Topic novelty, similarity to past converting campaigns |

## Data

- **Source:** Kaggle REES46 *E-commerce multichannel direct messaging
  2021–2023*: 172 million messages, about 1.85 million customers, 1,907
  campaigns, two years
- **Label:** purchase completed within 24 hours of a message; when several
  messages precede one purchase, the last one before it gets the credit
  (last-touch)
- **Positive rate:** 0.1234%
- **Split (manuscript):** customer-level stratified 70/15/15; test set
  25,798,280 messages; 5-fold stratified cross-validation and Optuna on PR-AUC

## Results

**Nine models, original-distribution holdout** (baseline → with domain features):

| Model | Precision | Recall | F1 | F2 | PR-AUC |
|---|---|---|---|---|---|
| **XGBoost** | 0.4199 → 0.2991 | 0.3074 → **0.5230** | 0.3550 → **0.3806** | 0.3248 → **0.4549** | 0.3177 → **0.3620** |
| Random Forest | 0.2371 → 0.3041 | 0.6456 → 0.4819 | 0.3468 → 0.3729 | 0.4802 → 0.4315 | 0.2887 → 0.3382 |
| TabNet | 0.2806 → 0.2690 | 0.3960 → 0.4781 | 0.3285 → 0.3443 | 0.3659 → 0.4138 | 0.2826 → 0.3182 |
| RNN-LSTM | 0.2519 → 0.2983 | 0.4290 → 0.4149 | 0.3174 → 0.3470 | 0.3761 → 0.3848 | 0.2661 → 0.3176 |
| ML average | 0.3285 → 0.3016 | 0.4765 → 0.5025 | 0.3509 → 0.3768 | 0.4025 → 0.4432 | 0.3032 → 0.3501 |
| DL average | 0.2456 → 0.2596 | 0.4669 → 0.4474 | 0.3150 → 0.3273 | 0.3872 → 0.3896 | 0.2633 → 0.2997 |

All nine models are in [PORTFOLIO.md](PORTFOLIO.md). Domain features raised
PR-AUC in both families (ML +15.5%, DL +13.9%), and tree ensembles stayed ahead
of the neural models.

**Which feature group matters** (XGBoost, one group added to the baseline):
marketing fatigue gave the largest gain (recall +72.58%, F2 +41.66%, PR-AUC
+10.40%), then temporal response (recall +40.07%, PR-AUC +7.47%). Purchase
cycle (+1.08%) and channel effectiveness (+0.21%) changed PR-AUC little on their
own.

**Top-K targeting** (test set, 25,798,280 messages):

| Top K | Share | Precision@K | Recall | Lift |
|---|---:|---:|---:|---:|
| 10,000 | 0.04% | 53.20% | 16.71% | 431.1x |
| 50,000 | 0.19% | 31.16% | 48.94% | 252.5x |
| 200,000 | 0.78% | 15.26% | 95.83% | 123.7x |

**Fatigue half-life sensitivity** (0.50x–1.50x of the base half-lives, 10M-row
sample run kept in `research/predict/tau_sweep_results/`): F1 0.3783–0.3873,
PR-AUC 0.3527–0.3613.

**SHAP:** fatigue features act as an early negative screen, temporal features
tune send timing, and purchase-cycle and recent-click features finalise the
conversion probability.

![Figure 1. Research framework](images/paper/figure-01.png)

*Figure 1. Four CRM tables, baseline and four domain-feature groups, nine models, evaluation.*

![Figure 2. SHAP contribution ranking by knowledge category](images/paper/figure-02.png)

*Figure 2. Features ranked by SHAP contribution, coloured by feature group.*

![Figure 3. SHAP summary plot](images/paper/figure-03.png)

*Figure 3. Direction and size of each feature's contribution.*

![Figure 4. SHAP rank shift between non-purchase and purchase predictions](images/paper/figure-04.png)

*Figure 4. How each feature's SHAP rank changes from non-purchase (left) to
purchase (right) predictions. Last-purchase recency (#27 → #7) and last-click
recency (#25 → #9) rise sharply for buyers, while push recency (#18 → #26) and
marketing fatigue (#20 → #29) matter mainly for screening out non-buyers.*

## How to run

```powershell
pip install -r requirements.txt
$env:MFA_DATASET_PATH = 'D:\path\to\final_data_100k_64.parquet'
$env:MFA_RESULTS_DIR = 'D:\path\to\new-results'
python research\preprocessing_fast.py      # features (needs the raw REES46 files)
python research\xgb.py                     # Optuna + 5-fold CV + holdout
python research\run_xgb_tau_sweep.py       # half-life sensitivity
```

The experiments ran on Python 3.12.11; their package versions were not
recorded, so `requirements.txt` pins the 2026-11 review environment (Python
3.13.2). Without data, `python research\verify_baseline.py` checks the stored
result file. Other models: `research/RF.py`, `MLP.py`, `CNN.py`, `RNN.py`, `LSTM.py`,
`CNNLSTM.py`, `RNNLSTM.py`, `tabnet.py`. See
[research/RUN_MANIFEST.md](research/RUN_MANIFEST.md).

## Limitations

From the paper (section 5.4):

- Observational data: associations and predictive contribution, not causal
  effects; SHAP values are model attributions.
- One platform; fatigue half-lives and purchase cycles may differ in other
  business models.
- Half-lives are fixed exploratory settings, not learned per customer.
- Customer-level isolation, but no forward-chaining validation across years or
  quarters.
- Compute and energy efficiency were not measured.

## Repository layout

```text
research/
├── preprocessing_fast.py      # feature engineering
├── split_utils.py             # customer-level split and grouped folds
├── xgb.py, RF.py, MLP.py, tabnet.py
├── CNN.py, RNN.py, LSTM.py, CNNLSTM.py, RNNLSTM.py
├── run_xgb_tau_sweep.py       # half-life sensitivity
└── predict/                   # stored results
images/paper/                  # figures
```

[Case study](PORTFOLIO.md) · [Academic summary](docs/ACADEMIC_SUMMARY.md)
