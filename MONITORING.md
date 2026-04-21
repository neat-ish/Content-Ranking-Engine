# Model Monitoring & Deployment Guide

**Project:** Content-Ranking-Engine
**Dataset:** MSLR-WEB10K -- Microsoft Learning to Rank
**Champion Model:** LambdaMART (LightGBM lambdarank objective) -- selected via A/B test in notebook 07

**Deployment context:** Offline batch ranking pipeline -- query-level document scoring and re-ranking for content retrieval systems

This pipeline scores a fixed candidate set of documents per query using the trained LambdaMART model and returns a ranked list. The model runs in batch (per search session or pre-computed for static query sets), consuming feature-engineered query-document pairs and producing relevance scores that feed into a downstream ranking layer or directly into the served result list.

The model does not perform first-stage retrieval. It assumes a candidate set has already been generated (e.g. by BM25 or an ANN index). Monitoring runs on a labelled evaluation window using the same five business KPIs defined in notebooks 02 and 06 (NDCG@10, MRR, P@1, Frustration Rate, Hit@3).

---

## 1. Introduction & Context

Search and content ranking are among the highest-leverage ML problems in internet products. A ranking model that lifts NDCG@10 by even a few points translates directly to users finding relevant content faster and fewer zero-result experiences. This project trains and evaluates a full spectrum of learning-to-rank models on MSLR-WEB10K -- from BM25 through gradient-boosted LambdaMART to neural listwise LSTM -- and governs the deployment decision with a rigorous A/B test that requires both statistical significance (paired t-test, bootstrap CI) and practical significance (Cohen's d, MDE comparison) before recommending rollout. The champion LambdaMART model demonstrates consistent superiority over the BM25 production baseline across all five KPIs and across all query difficulty segments, including hard queries with no highly relevant documents. Monitoring is designed to detect the specific failure modes of learning-to-rank systems: distribution shift in query-document features, degradation in NDCG on the confirmed-label window, and regression on the query difficulty segments that are most sensitive to model quality.

## 2. Business KPIs

### 2.1 Primary KPIs

| KPI | Definition | Target | Alert Threshold |
|---|---|---|---|
| NDCG@10 | Normalised Discounted Cumulative Gain -- quality of top-10 results | > BM25 baseline + MDE | Drop > 0.02 from deployment baseline |
| MRR | Mean Reciprocal Rank -- how far a user must scroll to find the first relevant result | > BM25 baseline | Drop > 0.03 from deployment baseline |
| P@1 | Precision at 1 -- is the single top result relevant (label >= 1)? | > BM25 baseline | Drop > 0.03 from deployment baseline |
| Frustration Rate | % of queries where top 3 results are all label=0 | < BM25 baseline | Increase > 0.02 from deployment baseline |
| Hit@3 (label >= 3) | % of queries with at least one highly relevant doc in top 3 | > BM25 baseline | Drop > 0.02 from deployment baseline |

NDCG@10 is the primary tracking metric because it is the standard academic and industry benchmark for ranking quality and is directly optimised by LambdaMART. Frustration Rate is the most user-facing KPI -- a user who sees three irrelevant results at the top of a search page is the most impactful failure mode.

### 2.2 Operational KPIs

| KPI | Definition | Alert Threshold |
|---|---|---|
| Feature PSI | PSI on top-10 features by gain vs. training distribution | Alert > 0.10 / Retrain > 0.20 |
| Score distribution PSI | PSI on predicted relevance score distribution | Alert > 0.10 / Retrain > 0.20 |
| Query difficulty distribution | % of queries in each difficulty bucket over time | > 5% shift from training distribution |
| Avg docs per query | Mean candidate set size per query | Alert if > 30% deviation from training mean |

### 2.3 Monitoring Cadence

| Frequency | What to Check | Owner |
|---|---|---|
| Daily | NDCG@10, Frustration Rate, score distribution PSI | Data Science |
| Weekly | All 5 KPIs, feature PSI on top-10 gain features, query difficulty distribution | Data Science |
| Monthly | Full feature PSI (all features), NDCG by difficulty bucket, retraining review | Data Science |
| Quarterly | Champion model re-evaluation on fresh labelled data, architecture review | Data Science + Product |

---

## 3. Model Monitoring

### 3.1 Score Distribution Drift (PSI)

PSI is computed monthly on predicted relevance scores and on the top-10 input features by LambdaMART gain importance, comparing the current period against the training distribution.

| PSI | Interpretation | Action |
|---|---|---|
| < 0.10 | Stable | No action |
| 0.10 - 0.20 | Moderate shift | Monitor closely; cross-reference with KPI trend |
| > 0.20 | Significant shift | Plan retrain if a high-gain feature is affected |

A PSI > 0.20 on any of the top-5 features by gain (BM25 whole-doc, PageRank, TF-IDF max, BM25 x PageRank interaction, QualityScore) is a hard retrain trigger. PSI > 0.20 on lower-gain features requires monitoring but not immediate retraining.

### 3.2 Ranking Performance Monitoring

Evaluate on a labelled evaluation window -- queries with confirmed relevance judgements, or a held-out evaluation set refreshed periodically.

- NDCG@10 on a rolling evaluation window -- alert if drop > 0.02 from deployment baseline
- All five KPIs tracked on the same window
- NDCG by query difficulty bucket -- alert if any single bucket drops > 0.03 (difficulty-specific degradation indicates a population shift in a specific query type)
- Feature importance rank stability -- if the top-3 features by gain change substantially between evaluation windows, the underlying document-query distribution has shifted

NDCG@10 is the primary tracking metric. Frustration Rate is the secondary alert because it is the most user-visible failure mode.

### 3.3 Retraining Criteria

Retrain when any two of the following are true simultaneously:

- Feature PSI > 0.20 on any top-5 gain feature
- NDCG@10 drops > 0.02 from deployment baseline on a labelled evaluation window
- Frustration Rate increases > 0.02 from deployment baseline
- A material population event occurs (new content domain added, query distribution changes significantly)

---

## 4. Deployment Notes

This section is a lightweight handoff reference. The pipeline is batch-only by design.

### 4.1 Serving Pattern

- **Batch re-ranking:** For each query, retrieve the candidate set (via BM25 or ANN index), construct the 136-feature (or engineered feature) vector per candidate, score with LambdaMART, and return the ranked list.
- **Inference speed:** LambdaMART (LightGBM) scores ~1ms per document. For a candidate set of 100 documents, total re-ranking latency is ~100ms -- acceptable for batch and near-real-time use cases.
- **Real-time serving:** For sub-100ms latency requirements, pre-compute and cache scores for common queries. For tail queries, LambdaMART can score at request time. Neural models (CNN, LSTM) are not recommended for real-time serving without a two-stage retrieval architecture.

### 4.2 Artifacts for Versioning

Each deployment should store the following. All paths relative to the run directory.

Model artifacts (serving):
```
  models/lgbm_lambdamart.txt           -- champion LambdaMART model
  models/lgbm_pointwise.txt            -- pointwise challenger (for fallback)
  data/processed/feat_cols_lgbm.txt    -- exact ordered feature list (all engineered features)
  data/processed/feat_cols_neural.txt  -- top-64 MI-selected features (neural models)
  models/scaler_neural.pkl             -- StandardScaler fit on training data
  models/clip_bounds.pkl               -- [1st, 99th] percentile bounds from training data
```

Evaluation artifacts (model quality):
```
  outputs/results/final_model_results.csv    -- all 5 models x 5 KPIs on test set
  outputs/results/feature_importance.csv     -- gain and split importance for LambdaMART
  outputs/plots/03_lambdamart_learning_curve.png
  outputs/plots/04_cnn_training_curves.png
  outputs/plots/05_lstm_training_curves.png
  outputs/plots/06_kpi_comparison.png
  outputs/plots/06_difficulty_segmentation.png
```

A/B test artifacts (deployment evidence):
```
  outputs/results/ab_test_summary.csv         -- per-metric delta, CI, p-value, Cohen's d, verdict
  outputs/results/segment_ab_results.csv      -- heterogeneous effects by query difficulty
  outputs/plots/07_forest_plots.png           -- CI forest plots per metric
  outputs/plots/07_power_analysis.png         -- MDE vs required n table
  outputs/results/statistical_test_record.json -- full test state: t-stat, p-val, CI, Cohen's d
```

Data lineage:
```
  data/data_snapshot_metadata.json    -- source file, fold used, row counts, parse date
  data/processed/train_query_stats.csv -- query difficulty distribution at training time
```

Monitoring baselines:
```
  baselines/training_feature_distributions.csv  -- mean, std, percentiles per feature at training time
  baselines/score_distribution.csv              -- predicted relevance score histogram at deployment
  baselines/deployment_kpis.csv                 -- all 5 KPIs at deployment time, per difficulty bucket
```

### 4.3 Dependency Pinning

The scoring environment must match training exactly.

| Package | Risk if unpinned |
|---|---|
| lightgbm | Score values and feature importance can shift across minor versions |
| torch | Neural model checkpoint loading requires matching version |
| scikit-learn | StandardScaler and feature selection outputs can differ across versions |
| pandas / numpy | Data type defaults and aggregation behaviour can change |

---

*Nitish Patnaik | github.com/neat-ish*
