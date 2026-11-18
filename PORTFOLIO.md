# Purchase Conversion Prediction with Dynamic Marketing Knowledge

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## Summary

First-author paper (with Kyungwon Kim, corresponding author; DIVE Lab, Incheon National University), forthcoming in *마케팅연구* (Korean Marketing Association), November 2026: *A Data-Centric AI Framework for E-Commerce Purchase Conversion Prediction via Dynamic Marketing Knowledge Integration*.

The study asks whether marketing theory, turned into time-varying features, helps more than model complexity when only 0.1234% of CRM messages lead to a purchase. On 172 million REES46 messages, XGBoost with four theory-based feature groups had the best F1 (0.3806), F2 (0.4549), and PR-AUC (0.3620) of nine models, and lifted recall from 0.3074 to 0.5230. Sending only the top 0.78% of messages captured 95.83% of conversions at 15.26% precision (123.7x lift).

## Challenge

- **Extreme imbalance.** One purchase per ~810 messages: accuracy and ROC-AUC say little, so the paper evaluates with PR-AUC, F1, F2, Precision@K, and lift.
- **Static customer summaries miss context.** The same customer may have just bought, be at a preferred hour, or be over-exposed to one channel.
- **Fair model comparison.** Deep models given raw sequences and trees given summaries would compare information, not architectures.

## Approach

| Feature group | Theory | Signals |
|---|---|---|
| ① Purchase and repurchase cycle | Customer lifetime value, cooling-off effect | Days since last purchase, repurchase hazard, post-purchase refractory period |
| ② Temporal response | Context-dependent preference, calendar effects | Preferred hour/day match, payday and quarter-end proximity |
| ③ Marketing fatigue | Habituation | Channel-specific decayed exposure (24h push, 48h email, 72h SMS), 30-day open/click rates |
| ④ Channel effectiveness | Information entropy, path dependency | Topic novelty, similarity to past converting campaigns |

Label: a purchase within 24 hours of a message, credited to the last message before it. Split: customer-level stratified 70/15/15; 5-fold stratified cross-validation; Optuna maximising validation PR-AUC.

## Decisions

**1. Features over model capacity.** Options: deeper sequence models on raw logs, or theory-based features on a tree model. Choice: features, with nine models compared. Why: at 0.1234% positives on sparse tabular logs, the missing piece is customer context. Cost: every feature needs a hand-written definition, and the gain came as higher recall at lower precision (0.4199 → 0.2991).

**2. Same information for every model.** Options: give neural models raw sequences, or give all models the same feature vector. Choice: the same vector; sequence models receive it as a length-1 sequence. Why: otherwise differences reflect input information, not architecture. Cost: recurrent models cannot use their sequence strength, which the paper states.

**3. Fatigue as a decayed sum.** Options: count messages, or sum exposures with channel-specific exponential decay. Choice: decay with half-lives of 24h (push), 48h (email), 72h (SMS). Why: yesterday's message should weigh more than last month's. Cost: the half-lives are assumptions; a 0.50x–1.50x sweep kept PR-AUC within 0.3527–0.3613.

**4. Report targeting as Precision@K and lift.** Options: convert the top-K result into money saved, or report hit rate and lift by budget. Choice: Precision@K and lift. Why: no unit cost or control group, and the operating point depends on channel cost. Cost: a less striking headline, but every number is defensible.

## Results

**Nine models, original-distribution holdout** (baseline → with domain features):

| Model | Precision | Recall | F1 | F2 | PR-AUC |
|---|---|---|---|---|---|
| **XGBoost** | 0.4199 → 0.2991 | 0.3074 → 0.5230 | 0.3550 → **0.3806** | 0.3248 → **0.4549** | 0.3177 → **0.3620** |
| Random Forest | 0.2371 → 0.3041 | 0.6456 → 0.4819 | 0.3468 → 0.3729 | 0.4802 → 0.4315 | 0.2887 → 0.3382 |
| MLP | 0.2561 → 0.2602 | 0.4144 → 0.4635 | 0.3165 → 0.3333 | 0.3688 → 0.4009 | 0.2675 → 0.3072 |
| CNN | 0.2680 → 0.2392 | 0.3290 → 0.3549 | 0.2954 → 0.2858 | 0.3147 → 0.3236 | 0.2302 → 0.2546 |
| RNN | 0.2381 → 0.2598 | 0.4695 → 0.4965 | 0.3159 → 0.3411 | 0.3931 → 0.4200 | 0.2629 → 0.3195 |
| LSTM | 0.2163 → 0.2514 | 0.6035 → 0.5084 | 0.3185 → 0.3364 | 0.4444 → 0.4221 | 0.2688 → 0.3166 |
| CNN-LSTM | 0.2083 → 0.2393 | 0.6272 → 0.4155 | 0.3127 → 0.3037 | 0.4473 → 0.3621 | 0.2648 → 0.2647 |
| RNN-LSTM | 0.2519 → 0.2983 | 0.4290 → 0.4149 | 0.3174 → 0.3470 | 0.3761 → 0.3848 | 0.2661 → 0.3176 |
| TabNet | 0.2806 → 0.2690 | 0.3960 → 0.4781 | 0.3285 → 0.3443 | 0.3659 → 0.4138 | 0.2826 → 0.3182 |

**Feature-group ablation (XGBoost):** fatigue recall +72.58%, F2 +41.66%, PR-AUC +10.40%; temporal response recall +40.07%, PR-AUC +7.47%; purchase cycle PR-AUC +1.08%; channel effectiveness PR-AUC +0.21%.

**Top-K targeting:** top 10,000 messages (0.04%): precision 53.20%, lift 431.1x; top 200,000 (0.78%): precision 15.26%, recall 95.83%, lift 123.7x.

**SHAP:** fatigue features screen out non-responders first, temporal features adjust timing, and purchase-cycle and recent-click features finalise the probability.

## Reproducibility check

A code review in November 2026 found two places where the public code differed
from the paper's protocol: rows were split at random rather than by customer,
and the first-purchase recency feature could see a first purchase dated after
the message. The code now follows the paper (`research/split_utils.py`:
customer-level 70/15/15 split and grouped 5-fold CV; first purchases after the
send time are masked in `preprocessing_fast.py`). A re-check on the 10M-row
sample, one model per setting with the stored hyperparameters:

| 10M-row sample, XGBoost (stored params) | PR-AUC | F1 | Top-0.78% recall |
|---|---:|---:|---:|
| All features, previous code | 0.3678 | 0.3863 | 96.38% |
| All features, paper protocol | 0.3647 | 0.3755 | 97.37% |
| Baseline, previous code | 0.3426 | 0.3713 | 94.00% |
| Baseline, paper protocol | 0.3107 | 0.3615 | 94.47% |

The full model is essentially unchanged, and the gain from the domain features
over the baseline grows (PR-AUC +7.4% → +17.4%). ROC-AUC stays near 0.997 in
every setting, which reflects the mass of easy negatives rather than leakage.

## Limitations

- Observational data: predictive contribution, not causal effect; the paper proposes A/B tests, propensity matching, or difference-in-differences next.
- One platform; half-lives and purchase cycles may differ elsewhere.
- Fixed exploratory half-lives rather than per-customer estimates.
- No forward-chaining validation across years or quarters.
- Compute and energy efficiency not measured.

## Next

Learn half-lives per customer, add forward-chaining validation, and test whether fatigue-aware targeting changes purchases in a randomised campaign.
