# Content-Ranking-Engine

End-to-end learning-to-rank pipeline on the MSLR-WEB10K dataset. Covers exploratory analysis of query-document pairs, feature engineering with cross-zone aggregations and mutual information selection, classical ranking models (LightGBM Pointwise and LambdaMART), neural rankers (CNN pointwise and Bidirectional LSTM listwise), full model comparison across five KPIs, and a statistically rigorous A/B test validating the champion model against the BM25 baseline. Champion model: LambdaMART, selected via A/B test in notebook 07.

---

## Business Context

Search and content ranking systems are at the core of every large-scale internet product. A ranking model that moves NDCG@10 by even a few points translates directly to users finding relevant content faster, fewer frustrating result pages, and measurable improvements in engagement and retention. This project benchmarks the full spectrum of modern ranking approaches -- from a strong classical baseline (BM25) through gradient-boosted tree rankers (LightGBM Pointwise, LambdaMART) to neural architectures (CNN pointwise, Bidirectional LSTM listwise) -- on the MSLR-WEB10K benchmark used in industry and academic research. All models are evaluated on the same five business-oriented KPIs (NDCG@10, MRR, P@1, Frustration Rate, Hit@3) to ensure comparisons are grounded in user-facing impact rather than model-internal metrics. The deployment decision is governed by a structured A/B test with both statistical significance (paired t-test, bootstrap CI) and practical significance (Cohen's d, MDE comparison), ensuring the champion model clears the bar required for a production rollout decision.

## Architecture

```
+---------------------+
|    00_setup         |
|                     |
|  Create project     |
|  directories        |
|  Parse LibSVM       |
|  format             |
|  Fold 1 only:       |
|  train / vali /     |
|  test               |
|  Sanity checks +    |
|  label distribution |
|  visualisation      |
+---------------------+
          |
          v
+---------------------+
|    01_eda           |
|                     |
|  Label imbalance    |
|  analysis           |
|  Query difficulty   |
|  bucketing          |
|  Feature value      |
|  range audit        |
|  Skewness (136      |
|  features)          |
|  Spearman corr      |
|  vs label           |
|  Feature group      |
|  analysis           |
|  BM25 + random      |
|  baselines          |
+---------------------+
          |
          v
+---------------------+
|  02_feature_        |
|  engineering        |
|                     |
|  Clip [1st, 99th]   |
|  (train-fit only)   |
|  StandardScaler     |
|  (train-fit only)   |
|  Cross-zone max /   |
|  mean / std         |
|  Zone dominance     |
|  (argmax)           |
|  BM25 x PageRank    |
|  interaction        |
|  Query-level        |
|  normalisation      |
|  MI feature         |
|  selection (top 64) |
|  Two output sets:   |
|  LGBM (all feats)   |
|  Neural (top 64)    |
+---------------------+
          |
          v
+---------------------+      +---------------------+
|  03_baseline_       |      |  models/            |
|  models             |      |                     |
|                     | ---> |  lgbm_pointwise.txt |
|  LightGBM           |      |  lgbm_lambdamart    |
|  Pointwise          |      |  .txt               |
|  (regression obj)   |      |  feat_cols_lgbm.txt |
|                     |      +---------------------+
|  LightGBM           |
|  LambdaMART         |
|  (lambdarank obj,   |
|  trunc at 10)       |
|                     |
|  Feature            |
|  importance (gain)  |
|  Learning curves    |
+---------------------+
          |
          v
+---------------------+      +---------------------+
|  04_cnn_ranker      |      |  models/            |
|                     |      |                     |
|  Pointwise FC       | ---> |  cnn_ranker.pt      |
|  ranker (64 ->      |      |  feat_cols_         |
|  128 -> 64 ->       |      |  neural.txt         |
|  32 -> 1)           |      +---------------------+
|  BatchNorm + ReLU   |
|  + Dropout(0.3)     |
|  MSE loss           |
|  AdamW + ReduceLR   |
|  Early stopping     |
|  (patience=10)      |
+---------------------+
          |
          v
+---------------------+      +---------------------+
|  05_lstm_ranker     |      |  models/            |
|                     |      |                     |
|  Listwise Bi-LSTM   | ---> |  lstm_ranker.pt     |
|  (2 layers,         |      +---------------------+
|  hidden=128)        |
|  ListMLE loss       |
|  (Plackett-Luce)    |
|  AdamW +            |
|  CosineAnnealing    |
|  Early stopping     |
|  (patience=8)       |
|  MAX_DOCS=100       |
+---------------------+
          |
          v
+---------------------+      +---------------------+
|  06_model_          |      |  outputs/results/   |
|  comparison         |      |                     |
|                     | ---> |  final_model_       |
|  All 5 models on    |      |  results.csv        |
|  5 KPIs             |      +---------------------+
|  Paired t-test      |
|  per KPI pair       |
|  Query difficulty   |
|  segmentation       |
+---------------------+
          |
          v
+---------------------+
|  07_ab_testing      |
|                     |
|  Control: BM25      |
|  Treatment:         |
|  LambdaMART         |
|                     |
|  Power analysis     |
|  (MDE table)        |
|  50/50 query        |
|  split              |
|                     |
|  Statistical:       |
|  - Paired t-test    |
|  - Bootstrap 95% CI |
|                     |
|  Practical:         |
|  - Cohen's d        |
|  - MDE comparison   |
|                     |
|  Heterogeneous      |
|  effects by query   |
|  difficulty         |
|                     |
|  Verdict:           |
|  DEPLOY /           |
|  DEPLOY (partial) / |
|  HOLD               |
+---------------------+
```

---

## Dataset

**Source:** [MSLR-WEB10K](https://www.microsoft.com/en-us/research/project/mslr/) -- Microsoft Learning to Rank

| Property | Value |
|---|---|
| Query-document pairs | ~1.2M |
| Queries | 10,000 |
| Features | 136 (f1 -- f136, LibSVM format) |
| Relevance labels | 0 (irrelevant) -- 4 (perfectly relevant) |
| Label distribution | ~52% label=0, heavy imbalance |
| Fold used | Fold 1 (standard benchmark fold) |
| Train / Val / Test split | Pre-defined by dataset |

---

## Notebook Summaries

### 00_setup.ipynb -- Environment, Data Parsing, and Sanity Checks

Creates project directories and extracts Fold 1 from the MSLR-WEB10K zip. Implements a LibSVM parser using `load_svmlight_file` to read the label-qid-feature format into pandas DataFrames. Runs sanity checks on all three splits (document counts, query counts, label range, missing values) and visualises the label distribution and docs-per-query distribution.

**Decision:** Only Fold 1 is used. This is the standard benchmark fold used in research papers for head-to-head model comparisons on MSLR-WEB10K, ensuring results are comparable to published baselines.

### 01_eda.ipynb -- Exploratory Data Analysis

**Goal:** To understand the dataset structure, feature properties, and establish pre-model baselines before any feature engineering.

Key findings that directly shaped downstream decisions:

- **Label imbalance:** ~52% of documents are label=0. Accuracy is a meaningless metric. All models use ranking metrics (NDCG, MRR) throughout.
- **Feature value range:** Feature values span from large negative to large positive values. Clipping at [1st, 99th] percentile is required before scaling.
- **Skewness:** Majority of the 136 features are highly skewed (|skew| > 1). Standard normality assumptions do not hold -- reinforces the need for clipping over z-score outlier removal.
- **Spearman correlation with label:** Top-correlated features are concentrated in the BM25, TF-IDF, and PageRank families. This guided the feature group analysis and interaction feature design in notebook 02.
- **Query difficulty:** Queries are bucketed into four difficulty levels based on presence of highly relevant (label >= 3) documents. This bucketing is reused in notebooks 06 and 07 for heterogeneous effects analysis.
- **Pre-model baselines established:** Random ranking (NDCG@10 ~ 0.18) and BM25 single feature f75 (NDCG@10 ~ 0.38) are computed on the test set. All models must beat BM25.

### 02_feature_engineering.ipynb -- Feature Engineering and Selection

**Goal:** Clean, transform, and augment the 136 raw features into a modelling-ready set, with two distinct output formats for tree and neural models.

**Key decisions:**

- **Train-only clip bounds:** Percentile bounds for outlier clipping are computed on the training split only, then applied to val and test. This is a hard leakage prevention rule -- using val or test statistics to clip training data would introduce future information.
- **Train-only scaler fit:** Same principle. `StandardScaler` is fit on training data only.
- **Cross-zone aggregations:** Features f1-f136 are grouped into 5-zone families (e.g. BM25 over body, anchor, title, URL, whole-document). For each family, max, mean, and std across zones are computed. This compresses 5 correlated features into 3 that capture peak signal, average signal, and zone spread -- reducing redundancy while retaining information.
- **BM25 x PageRank interaction:** A high-authority page (high PageRank) about the exact query topic (high BM25) is more relevant than either signal alone. The interaction feature captures this joint effect.
- **Mutual Information for feature selection:** MI (not Spearman) is used for neural model feature selection because it captures non-linear relationships. Top 64 features by MI are used for CNN and LSTM; LightGBM uses all features since tree models handle redundancy natively.
- **Two output feature sets:** `train_lgbm.csv` (all engineered features, unscaled) for LightGBM; `train_neural.csv` (top-64 features, scaled) for CNN and LSTM.

Features created:

| Category | Examples |
|---|---|
| Cross-zone aggregations | `bm25_max`, `bm25_mean`, `bm25_std`, `tfidf_max`, `lmir_abs_std` |
| Zone dominance | `bm25_argmax`, `tfidf_argmax` |
| Interaction features | `bm25_x_pagerank`, `tfidf_x_qualityscore` |
| Query-level normalisation | `f80_qnorm`, `f95_qnorm` (feature value / query mean) |

### 03_baseline_models.ipynb -- LightGBM Pointwise and LambdaMART

**Goal:** Establish strong classical ranking baselines and determine which LightGBM objective (pointwise regression vs. listwise LambdaMART) performs better.

**LightGBM Pointwise** uses a regression objective (`rmse`) treating relevance labels as continuous scores. Documents are ranked by predicted score at inference time. This is the simpler approach -- the model never sees ranking structure during training.

**LightGBM LambdaMART** uses the `lambdarank` objective with `lambdarank_truncation_level=10`. Gradients are computed to directly optimise NDCG@10 by weighting document swaps by their impact on the metric. The model explicitly learns that moving a label=4 document from rank 5 to rank 1 matters more than moving a label=1 document.

**Decision to use lambdarank_truncation_level=10:** This focuses gradient computation on the top-10 positions -- the only positions that matter for NDCG@10 and the positions users actually see. Without truncation, the model wastes gradient budget on swaps at rank 50+ that have no business impact.

Feature importance is computed using gain (not split count). A feature used once with high gain is more valuable than one used hundreds of times with marginal gain. Engineered features (cross-zone aggregations, BM25 x PageRank interaction) appear in the top-25 by gain, validating the feature engineering decisions from notebook 02.

### 04_cnn_ranker.ipynb -- CNN Pointwise Ranker

**Goal:** Train a neural pointwise ranker that learns feature interactions directly from the 64-dimensional feature vector.

**Architecture:** Originally designed as a 1D CNN to exploit the zone-grouped feature layout (adjacent features belong to the same semantic family). Replaced with a fully-connected equivalent (64 -> 128 -> 64 -> 32 -> 1) due to a known MPS/Conv1d instability on Apple Silicon (PyTorch 2.5.1, Python 3.11, M4 hardware). The FC architecture is architecturally equivalent for learning feature interactions across the full input.

| Component | Detail |
|---|---|
| Architecture | FC: 64 -> 128 -> 64 -> 32 -> 1 |
| Regularisation | BatchNorm1d + ReLU + Dropout(0.3) after blocks 1 and 2 |
| Loss | Pointwise MSE |
| Optimiser | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Stopping | Early stopping (patience=10), 50 epochs max |

**Decision:** MSE loss is used for pointwise training. The model scores each document independently -- ranking is achieved by sorting predicted scores at inference time. This is the same approach as LightGBM Pointwise but with a neural architecture.

### 05_lstm_ranker.ipynb -- Bidirectional LSTM Listwise Ranker

**Goal:** Train a listwise ranker that processes all documents for a query simultaneously as a sequence, enabling each document's score to be informed by its context within the result set.

**Why listwise vs. pointwise:** The CNN in notebook 04 scores each document independently. The LSTM receives the entire query's document list as a padded sequence. This allows the model to learn relative relevance -- "given the other documents in this result set, how relevant is this one?" -- which is closer to how a real ranking system should reason.

| Component | Detail |
|---|---|
| Architecture | Input projection (64 -> 128, LayerNorm, ReLU) + Bi-LSTM (2 layers, hidden=128) + FC (256 -> 64 -> 1) |
| Loss | ListMLE (Plackett-Luce listwise loss) |
| Optimiser | AdamW (lr=5e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=30, eta_min=1e-5) |
| Stopping | Early stopping (patience=8), 40 epochs max |
| MAX_DOCS | 100 per query (80th percentile of query length) |

**Decision to use ListMLE loss:** ListMLE computes the probability of the ideal permutation (sorted by true label) given the model's scores, penalising the model when it assigns high scores to low-relevance documents. This is a direct optimization signal for ranking quality, unlike MSE which treats relevance prediction as regression.

**Decision on bidirectional LSTM:** A forward-only LSTM encodes documents in one direction. Bidirectional LSTM encodes each document informed by both the documents ranked above and below it -- giving richer context for the relevance score assignment.

### 06_model_comparison.ipynb -- Full Model Comparison

**Goal:** Compare all five models on the same five KPIs on the held-out test set, with statistical significance testing between pairs.

CNN and LSTM scores are loaded from saved results (computed at end of notebooks 04 and 05) rather than re-running inference. LightGBM models are re-scored live since inference is fast.

**KPIs reported:**

| KPI | Definition | Direction |
|---|---|---|
| NDCG@10 | Normalised Discounted Cumulative Gain at 10 | Higher is better |
| MRR | Mean Reciprocal Rank -- position of first relevant doc | Higher is better |
| P@1 | Precision at 1 -- is the top result relevant? | Higher is better |
| Frustration Rate | % of queries with no relevant doc in top 3 | Lower is better |
| Hit@3 (label >= 3) | % of queries with a highly relevant doc in top 3 | Higher is better |

**Query difficulty segmentation:** Results are broken out across four difficulty buckets (No relevant docs, Slightly relevant only, Few highly relevant, Many highly relevant). LambdaMART maintains its advantage across all difficulty levels -- a deployment requirement, since a model that degrades on hard queries should not roll out uniformly.

**Decision:** LambdaMART is selected as champion based on consistent superiority across all five KPIs. The LSTM achieves competitive NDCG@10 but underperforms on Frustration Rate. The CNN underperforms both LightGBM models on all KPIs, confirming that the listwise ranking signal in LambdaMART is more effective than pointwise MSE for this dataset.

### 07_ab_testing.ipynb -- A/B Test

**Goal:** Validate whether LambdaMART produces a statistically and practically significant improvement over BM25, with results that would support a production deployment decision.

**Control:** BM25 (f75 -- whole-document BM25 score). The strongest single feature and the production baseline for keyword-based search systems.

**Treatment:** LambdaMART. The champion from notebook 06.

**Power analysis:** Required sample size per arm is computed across multiple MDE values using the standard formula `n = 2 * (z_alpha/2 + z_beta)^2 * sigma^2 / MDE^2`. Baseline sigma is estimated from the BM25 per-query NDCG@10 distribution. This ensures the chosen split size (half the 2,000 test queries per arm) is adequately powered.

**Statistical tests:**

| Test | What it measures |
|---|---|
| Paired t-test on per-query metrics | Is the difference larger than query-level variance? |
| Bootstrap 95% CI (2,000 resamples) | Uncertainty range on the delta without normality assumption |
| Cohen's d | Effect size -- is the difference large enough to act on? |

**Heterogeneous effects analysis:** The A/B test is repeated within each query difficulty bucket. A model that produces a significant improvement overall but harms hard queries (No relevant docs bucket) should only roll out selectively. Both p-value and Cohen's d are reported per segment.

**Decision logic:**

```
For each metric (NDCG@10, Frustration Rate, Hit@3):
  Statistical significance: p < 0.05
  Practical significance:   |Cohen's d| >= 0.2 AND |delta| >= MDE

Verdict:
  All metrics stat + practical sig  ->  DEPLOY
  Primary metric sig, secondaries   ->  DEPLOY (monitor secondaries)
  mixed or neither significant
  Any metric shows significant harm ->  HOLD
```

---

## Key Design Decisions Summary

| Decision | Rationale |
|---|---|
| Train-only clip and scale fit | Prevents val/test statistics leaking into training transformations |
| Mutual Information for feature selection | Captures non-linear relationships that Spearman correlation misses |
| Two feature sets (LGBM vs Neural) | LightGBM handles redundancy natively; neural models benefit from top-64 MI-selected scaled features |
| lambdarank_truncation_level=10 | Focuses gradient budget on top-10 positions; rank 50+ swaps have no business impact |
| Gain over split for feature importance | A feature used once with high gain outweighs one used frequently with marginal gain |
| ListMLE loss for LSTM | Direct listwise optimisation signal; MSE treats ranking as regression and ignores rank structure |
| Bidirectional LSTM | Each document's score is informed by context from both above and below in the ranking |
| FC layers for CNN (not Conv1d) | MPS/Conv1d instability on Apple Silicon (M4, PyTorch 2.5.1, Python 3.11) |
| A/B test on BM25 vs LambdaMART | Tests the business-relevant decision: does ML ranking beat the production keyword baseline? |
| Heterogeneous effects by difficulty | A model that harms hard queries should not roll out uniformly |

---

## Results

| Model | NDCG@10 | MRR | P@1 | Frustration Rate | Hit@3 (label>=3) |
|---|---|---|---|---|---|
| Random Ranking | ~0.18 | ~0.28 | ~0.25 | ~0.55 | ~0.10 |
| BM25 (f75) | ~0.38 | ~0.55 | ~0.50 | ~0.30 | ~0.30 |
| LightGBM Pointwise | ~0.43 | ~0.60 | ~0.55 | ~0.25 | ~0.38 |
| LambdaMART | ~0.47 | ~0.65 | ~0.60 | ~0.20 | ~0.43 |
| CNN Ranker | ~0.33 | ~0.50 | ~0.46 | ~0.28 | ~0.32 |
| LSTM Ranker | ~0.17 | ~0.29 | ~0.26 | ~0.40 | ~0.18 |

**Champion:** LambdaMART -- best on all five KPIs, confirmed by A/B test in notebook 07.

Note: Exact values are produced at notebook runtime. LSTM underperforms on this dataset due to the near-random label ordering within queries making listwise context less beneficial than direct NDCG gradient optimisation.

---

## Project Structure

```
Content-Ranking-Engine/
|
|-- data/
|   |-- raw/
|   |   |-- MSLR-WEB10K.zip          # Source dataset (gitignored)
|   |   |-- Fold1/                   # Extracted: train.txt, vali.txt, test.txt
|   |-- processed/
|       |-- train.csv / val.csv / test.csv
|       |-- train_lgbm.csv / val_lgbm.csv / test_lgbm.csv
|       |-- train_neural.csv / val_neural.csv / test_neural.csv
|       |-- feat_cols_lgbm.txt
|       |-- feat_cols_neural.txt
|
|-- models/
|   |-- lgbm_pointwise.txt
|   |-- lgbm_lambdamart.txt
|   |-- cnn_ranker.pt
|   |-- lstm_ranker.pt
|
|-- outputs/
|   |-- plots/                        # Training curves, KPI charts, forest plots
|   |-- results/
|       |-- final_model_results.csv
|       |-- ab_test_summary.csv
|       |-- segment_ab_results.csv
|
|-- 00_setup.ipynb
|-- 01_eda.ipynb
|-- 02_feature_engineering.ipynb
|-- 03_baseline_models.ipynb
|-- 04_cnn_ranker.ipynb
|-- 05_lstm_ranker.ipynb
|-- 06_model_comparison.ipynb
|-- 07_ab_testing.ipynb
|-- MONITORING.md
|-- requirements.txt
|-- .gitignore
|-- README.md
```

---

## Setup

```bash
git clone https://github.com/<your-username>/Content-Ranking-Engine.git
cd Content-Ranking-Engine
pip install -r requirements.txt
```

Download MSLR-WEB10K from the Microsoft Research link above and place `MSLR-WEB10K.zip` in `data/raw/`. Then run notebooks in order: 00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07.

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
lightgbm
torch
joblib
jupyter
```

---

## Monitoring

See `MONITORING.md` for the full monitoring guide: business KPIs, ranking model drift detection, retraining criteria, and deployment notes.

---

*Nitish Patnaik | github.com/neat-ish*
