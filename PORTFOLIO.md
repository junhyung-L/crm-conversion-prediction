# Purchase Conversion Prediction with Dynamic CRM Features

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## At a glance

This project treats purchase conversion prediction as a targeting problem in an e-commerce CRM setting where the positive rate is only 0.1234%. Rather than increasing model complexity alone, it turns marketing theory into customer-level, time-varying features: purchase-cycle readiness, response timing, accumulated message fatigue, and channel/content fit.

On the manuscript's original-distribution holdout, domain-enhanced XGBoost reached F1 0.3806 and PR-AUC 0.3620, with recall rising from 0.3074 to 0.5230 over the baseline feature set. Contacting the top 0.78% of scores captured 95.83% of conversions at a precision of 15.26% (lift 123.7x); the top 10,000 messages reached 53.20% precision (431.1x).

## Framing the prediction problem

Static customer summaries miss an important part of marketing behaviour: the same customer can be just after a purchase, at a preferred activity hour, or overexposed to a repeated message theme. I organized this context into four feature families.

| Feature family | Operational signal | Purpose |
|---|---|---|
| Purchase and repurchase cycle | customer-specific intervals, repurchase readiness, post-purchase refractory period | represent position in an individual purchase cycle |
| Temporal response | preferred hour/day alignment and proximity to payday or quarter-end | distinguish sending time from customer context |
| Marketing fatigue | recency-weighted exposure with channel-specific half-lives | model repeated exposure and recovery together |
| Channel effectiveness | topic freshness and similarity to prior converting paths | identify a more suitable contact point |

Fatigue is a decayed sum rather than a message count: recent messages weigh more, the effect fades over time, and each channel has its own half-life. It is a predictive signal, not a measured causal effect.

## Data and preparation boundary

The manuscript uses Kaggle REES46’s *E-commerce multichannel direct messaging 2021–2023*: approximately 172 million messaging/response logs across roughly two years, 1.85 million customers, and 1,907 campaigns. It combines four relational tables: interaction logs, campaign metadata, customer attributes, and temporal context.

The documented preparation keeps marketing campaigns tied to customer profiles or promotions and excludes operational transaction notifications, then rebuilds each customer's message history in time order. The manuscript specifies a stratified 70/15/15 train/validation/test split that preserves the original class distribution, with imputation and scaling statistics fitted on training data only.

The executable artefacts retained in this repository have a narrower boundary. `RUN_MANIFEST.md` describes `final_data_100k_64.parquet`, seed 1, a stratified 15% holdout, and three-fold `StratifiedKFold`; the retained XGBoost JSON records a 1,499,900-row holdout confusion matrix with 64 features. This is not the same scale or split as the manuscript experiment, so the two are kept separate.

## Model comparison and findings

The manuscript compares nine models: Random Forest, XGBoost, MLP, CNN, RNN, LSTM, CNN-LSTM, RNN-LSTM, and TabNet. Because accuracy is misleading at this prevalence, it reports PR-AUC, F1, F2, and recall alongside precision. The manuscript describes PR-AUC-driven Optuna optimisation, five-fold validation, early stopping, and pruning.

| XGBoost holdout metric | Baseline features | Domain features |
|---|---:|---:|
| Precision | 0.4199 | 0.2991 |
| Recall | 0.3074 | 0.5230 |
| F1 | 0.3550 | 0.3806 |
| F2 | 0.3248 | 0.4549 |
| PR-AUC | 0.3177 | 0.3620 |

The precision–recall trade-off matters here: more converters are captured, while precision falls. Choosing a preferred operating point would require campaign contact cost and the cost of unnecessary messages. The 13.1% figure in the manuscript is a TabNet **F2-score** change (0.3659 to 0.4138), not an observed marketing-cost reduction.

The manuscript's top-K analysis reports that the top 0.78% by predicted score captured 95.83% of holdout conversions. Precision@K at that cut was 15.26% (lift 123.7x over random sending); at the top 10,000 messages it was 53.20% (431.1x). It is a ranking result on the holdout, not a measured campaign outcome.

## Decisions

**1. Where to spend the modelling effort.**
Options: stack more model capacity (seven of the nine compared models are neural or tabular-DL), or encode marketing knowledge as features and keep a tree model.
Choice: theory-derived features on XGBoost, with the deep models kept as comparison points.
Why: at a 0.1234% positive rate on sparse tabular logs, the missing information is customer context (cycle position, timing, exposure history), which more layers cannot recover from static summaries.
Cost: every feature family needs a hand-specified definition, and the gain came as higher recall with lower precision (0.4199 to 0.2991), not a uniform improvement.

**2. How to model fatigue.**
Options: count messages received; or sum exposures with exponential decay and a different half-life per channel.
Choice: decayed sum, with initial half-lives of 24 hours for push, 48 hours for email, and 72 hours for SMS (`research/preprocessing_fast.py`).
Why: a count treats a message from yesterday and one from last month the same; decay lets exposure and recovery both enter the feature.
Cost: the half-lives are assumptions, not learned. The retained sweep scaling all of them from 0.50x to 1.50x keeps PR-AUC between 0.3527 and 0.3613, so the result is not fragile to that choice, but individual half-lives were not estimated.

**3. How to handle imbalance and choose the operating point.**
Options: resample the training data to balance classes; or keep the original distribution, tune `scale_pos_weight`, and pick a threshold afterwards.
Choice: the second. In the retained code, Optuna maximises PR-AUC over three-fold stratified validation (`research/xgb.py`), and the decision threshold is the F1-maximising point on validation predictions, not on the test split.
Why: the holdout keeps the real 0.1234% rate, so the reported precision and recall describe what a campaign would actually see.
Cost: F1 weighs a missed buyer and a wasted message equally; a real campaign would set the threshold from contact cost, which is not available.

**4. What to claim from the targeting analysis.**
Options: translate the top-0.78% result into money saved; or report it as a ranking simulation.
Choice: ranking simulation only.
Why: there is no send log, unit cost, or control group, so savings or uplift cannot be computed.
Cost: the headline is less striking than a savings figure, but every number in it can be defended.

## What remains useful—and what remains open

SHAP and ablation analyses provide an interpretable view of associations: fatigue and temporal-response features act as early filtering signals, while purchase-cycle and recent high-engagement signals contribute later in the model’s ranking. They do not establish treatment effects.

The repository retains preprocessing, model, and half-life sensitivity scripts plus compact JSON/CSV result files, but not the source Parquet files or trained model binaries. A stronger next iteration would version the data and row counts, fix package versions and split timestamps in one manifest, evaluate time-ordered generalisation, and add a randomized or uplift design before making campaign-impact claims.

## Evidence

- [Run manifest](research/RUN_MANIFEST.md)
- [Retained XGBoost result](research/predict/result_xgb.json)
- [Preprocessing](research/preprocessing_fast.py), [XGBoost experiment](research/xgb.py), and [half-life sweep](research/run_xgb_tau_sweep.py)
