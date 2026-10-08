# Student Academic Performance & At-Risk Prediction System Using Machine Learning

**Machine Learning Mini-Project Report**

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Objectives](#4-objectives)
5. [Dataset Description](#5-dataset-description)
6. [System Architecture](#6-system-architecture)
7. [Data Preprocessing](#7-data-preprocessing)
8. [Algorithms](#8-algorithms)
   - [8.1 Multivariate Linear Regression](#81-multivariate-linear-regression)
   - [8.2 Logistic Regression](#82-logistic-regression)
   - [8.3 Support Vector Machine](#83-support-vector-machine)
   - [8.4 Ensemble Learning / Random Forest](#84-ensemble-learning--random-forest)
9. [Implementation](#9-implementation)
10. [Results](#10-results)
    - [10.6 The decision threshold and probability calibration](#106-the-decision-threshold-and-probability-calibration)
    - [10.7 Robustness check on the second subject file](#107-robustness-check-on-the-second-subject-file)
11. [Model Comparison](#11-model-comparison)
12. [Screenshots](#12-screenshots)
13. [Advantages](#13-advantages)
14. [Limitations](#14-limitations)
15. [Future Scope](#15-future-scope)
16. [Conclusion](#16-conclusion)
17. [References](#17-references)

---

## 1. Abstract

Academic failure is straightforward to identify after the fact and difficult to
reverse once a final grade has been published. This project investigates whether
failure can instead be anticipated from information a school already holds about
a student, early enough for intervention to be possible.

Four supervised learning algorithms are implemented on the UCI Student
Performance dataset (395 secondary-school students, 33 attributes): Multivariate
Linear Regression predicts the final grade on a 0–20 scale, while Logistic
Regression, a Support Vector Machine with a radial basis function kernel, and a
Random Forest ensemble classify each student as *At Risk* or *Not At Risk* of
falling below the pass mark.

The project's central methodological constraint is the exclusion of all grade
columns from the feature matrix. The dataset's intermediate assessments, `G1` and
`G2`, correlate with the final grade at r = +0.80 and r = +0.90 respectively;
using them constitutes target leakage and defeats the early-warning objective.
The effect of this exclusion was measured rather than assumed: a parallel set of
models fitted *with* the intermediate grades achieves a Random Forest F1 score of
0.8679 against 0.5581 for the leakage-free models, and a regression R² of 0.7241
against 0.1415.

The leakage-free results are moderate and are reported as such. All three
classifiers attain ROC-AUC between 0.6858 and 0.7242, materially above the 0.500
expected from random guessing, establishing that student background and
behavioural attributes do carry predictive information. However, F1 scores near
0.56 indicate that approximately half of flagged students would be false alarms.
Statistical analysis of the model comparison shows the three classifiers to be
indistinguishable: the F1 gap between the best and second-best model is 0.0033
against a cross-validation standard deviation of 0.0826. The regression task
proves substantially harder, with a cross-validated R² of −0.0781, indicating
performance no better than predicting the cohort mean.

Permutation importance analysis identifies prior academic failure as the dominant
predictor, with an importance of 0.2786 against 0.0502 for the next-ranked
feature. Notably, this contradicts the Random Forest's built-in impurity
importance, which ranks school absences first — a demonstration of the documented
cardinality bias in impurity-based measures.

The system is delivered as a seven-section Streamlit application presenting
predictions, dataset analysis, model comparison, explainability and a dedicated
demonstration of the avoided data leakage.

**Keywords:** educational data mining, at-risk prediction, ensemble learning,
Random Forest, Support Vector Machine, logistic regression, data leakage,
permutation importance.

---

## 2. Introduction

Machine learning applied to educational data — commonly termed *educational data
mining* or *learning analytics* — seeks to extract actionable patterns from
records that institutions already maintain. A recurring application is the
early-warning system: a model that identifies students likely to require
additional support while there remains time to provide it.

The appeal is clear. Institutional interventions such as tutoring, mentoring and
counselling are resource-constrained, and directing them requires some basis for
prioritisation. A model that ranks students by estimated risk offers such a
basis, provided its limitations are understood.

The difficulty is equally clear and is frequently understated in the literature.
Educational outcomes depend substantially on factors that institutional datasets
do not record: household circumstances, health, motivation, the quality of prior
schooling, and employment obligations outside school. A model restricted to
recorded attributes is therefore working with an incomplete picture by
construction.

This project engages that difficulty directly rather than circumventing it. A
common shortcut in published work on this dataset is the inclusion of
intermediate grade columns as predictors, which produces reported accuracies
above 90%. Such models are, however, of limited practical value: they predict the
third-period grade from the second-period grade, by which point a struggling
student is already identifiable without any model. This project excludes those
columns, quantifies the resulting loss in apparent performance, and reports the
honest figures.

A secondary contribution is methodological rigour in the model comparison. Three
classifiers are tuned using identical cross-validation folds and an identical
scoring function, and the resulting differences are tested against the
fold-to-fold variation before any ranking is asserted.

---

## 3. Problem Statement

Given a set of academic, demographic and behavioural attributes recorded for a
secondary-school student — and explicitly excluding any measurement of the
academic outcome being predicted — construct and evaluate machine learning models
that:

1. estimate the student's final academic score as a continuous value; and
2. classify the student as either *At Risk* or *Not At Risk* of failing.

The models must be trained and evaluated without target leakage, must be
compared on a methodologically sound basis, and must be presented in a form that
permits a non-specialist to interpret both the predictions and their
reliability.

Formally, let **x** ∈ ℝ³⁰ denote the vector of student attributes excluding all
grade columns. Two mappings are sought:

- a regression function *f*: **x** → *ŷ* ∈ [0, 20], estimating the final grade;
- a classification function *g*: **x** → {0, 1}, where 1 denotes *At Risk*,
  defined as a final grade below the institutional pass mark.

---

## 4. Objectives

| No. | Objective |
|---|---|
| 1 | Obtain and validate a legitimate public dataset of student academic performance. |
| 2 | Conduct exploratory analysis to characterise the data and identify issues affecting model design. |
| 3 | Construct a preprocessing pipeline that handles missing values, scales numeric features and encodes categorical features without leakage. |
| 4 | Identify and exclude all features that constitute target leakage, documenting the justification for each exclusion. |
| 5 | Implement Multivariate Linear Regression for the continuous prediction task. |
| 6 | Implement Logistic Regression, a Support Vector Machine and a Random Forest ensemble for the classification task. |
| 7 | Evaluate all models using appropriate regression and classification metrics, including baselines. |
| 8 | Compare the classifiers on a fair basis and assess the statistical significance of observed differences. |
| 9 | Analyse feature importance using more than one measure and interpret the results correctly with respect to causality. |
| 10 | Quantify the effect of the data leakage that was avoided. |
| 11 | Deliver an interactive web application presenting predictions, analysis and explanations. |
| 12 | Document all results, limitations and ethical considerations without overstatement. |

---

## 5. Dataset Description

### 5.1 Source and provenance

The dataset is the **Student Performance** dataset of Cortez and Silva (2008),
published in the UCI Machine Learning Repository as dataset 320 (DOI
10.24432/C5TG7T). It comprises records of students in secondary education at two
Portuguese schools, compiled from school reports and student questionnaires.

Two files are published: a Mathematics subject file of 395 students and a
Portuguese-language subject file of 649 students, each with 33 attributes. This
project uses the Mathematics file as its primary dataset.

Prior to training, the loaded file is validated against the row and column counts
documented by UCI. A mismatch raises an exception and halts execution, preventing
the possibility of training on a truncated or substituted file.

### 5.2 Structure

| Property | Value |
|---|---|
| Students (instances) | 395 |
| Attributes | 33 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Attribute types | 16 integer, 17 categorical |
| Final grade `G3` — mean | 10.42 |
| Final grade `G3` — median | 11.0 |
| Final grade `G3` — range | 0–20 |
| Final grade `G3` — standard deviation | 4.58 |

### 5.3 Attribute inventory

**Numeric attributes retained as model inputs (13):** `age`, `Medu` (mother's
education, 0–4), `Fedu` (father's education, 0–4), `traveltime` (1–4),
`studytime` (1–4), `failures` (0–3), `famrel` (1–5), `freetime` (1–5), `goout`
(1–5), `Dalc` (1–5), `Walc` (1–5), `health` (1–5), `absences` (0–75).

**Categorical attributes retained as model inputs (17):** `school`, `sex`,
`address`, `famsize`, `Pstatus`, `Mjob`, `Fjob`, `reason`, `guardian`,
`schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`,
`romantic`.

**Attributes excluded (4):** `G3`, `at_risk`, `G2`, `G1`. Justification is given
in Section 7.2.

### 5.4 Findings from exploratory analysis

Three characteristics of the data were identified during exploratory analysis,
each of which influenced subsequent design decisions.

**(a) A discontinuity at zero in the target distribution.** Thirty-eight students
(9.6% of the cohort) recorded a final grade of exactly 0, with no students
recording grades of 1, 2 or 3 — the lowest non-zero grade is 4.

Two further observations establish these records as administrative artefacts
rather than genuine assessments. First, **all thirty-eight earned a non-zero
first-period grade**, `G1` ranging from 4 to 12, indicating that they were
attending and being assessed. Second, **all thirty-eight record exactly zero
absences**. A student cannot simultaneously exhibit perfect attendance and have
failed to sit the final assessment; the combination is not a plausible academic
record and is more consistent with a placeholder entered for a student who left
the roll or whose final mark was never recorded.

These records were retained. They represent the students an early-warning system
is most concerned with, and their removal — while it would improve the reported
R² — would constitute selective exclusion of inconvenient observations. The
consequence is a bimodal target distribution and a cluster of large regression
residuals, both of which are reported in Section 10.

**(b) A masked relationship between absences and the final grade.** The Pearson
correlation between `absences` and `G3` across the full cohort is +0.034, which
is to say effectively zero. This figure is, however, an artefact of the
subpopulation identified in (a): all 38 students recording a final grade of zero
also record exactly zero absences. Excluding them, the correlation becomes
**−0.213** — a sign reversal and a sixfold increase in magnitude.

The substantive conclusion is therefore that absence does relate negatively to
attainment among students with a genuine recorded grade, but that the
record-keeping artefact suppresses the relationship in the aggregate statistic.
Both figures are reported, since the aggregate figure alone would be misleading.
The finding also illustrates why an annual absence total is a fragile feature,
which informs the recommendation in Section 15.

**(c) Class imbalance establishing a high accuracy baseline.** Applying the
pass-mark threshold yields 130 at-risk students (32.9%) and 265 not-at-risk
students (67.1%). A trivial classifier predicting the majority class for every
instance therefore attains 67.1% accuracy while identifying no at-risk students
whatsoever. This figure is the reference point against which all accuracy
measurements in this report must be read, and it motivates the use of F1 rather
than accuracy as the primary comparison metric.

### 5.5 Target variable definition

**Regression target.** The attribute `G3`, the final-period grade, a continuous
value on the interval [0, 20].

**Classification target.** A binary attribute `at_risk`, derived as:

```
at_risk = 1   if G3 < 10     (At Risk)
at_risk = 0   if G3 ≥ 10     (Not At Risk)
```

The threshold of 10 is the pass mark of the Portuguese 0–20 grading scale. It is
therefore the domain's own definition of academic failure rather than a value
selected to produce a convenient class balance. The resulting distribution was
examined before the threshold was adopted and is reported in Section 5.4(c).

### 5.6 Exclusion of the second subject file

The Portuguese-language file is included in the repository and the training
script accepts it via a command-line argument, but the two files are not
combined. Three hundred and eighty-two students are common to both files;
concatenation would therefore place the same individual in both the training and
test partitions, constituting a form of leakage that would inflate all reported
metrics.

---

## 6. System Architecture

### 6.1 Processing pipeline

```
                        ┌──────────────────────────────┐
                        │  Raw data: student-mat.csv   │
                        │       395 × 33               │
                        └──────────────┬───────────────┘
                                       │  validate against UCI shape
                                       ▼
                        ┌──────────────────────────────┐
                        │  Derive at_risk from G3      │
                        │  at_risk = 1 if G3 < 10      │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │  Exclude G3, at_risk, G2, G1 │
                        │  → 30 features               │
                        │  (enforced by assertion)     │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │  Train / test split 80 : 20  │
                        │  stratified, random_state=42 │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │  ColumnTransformer           │
                        │  numeric:     impute + scale │
                        │  categorical: impute + 1-hot │
                        │  → 43 columns                │
                        └──────────────┬───────────────┘
                                       │
        ┌──────────────┬───────────────┼───────────────┬──────────────┐
        ▼              ▼               ▼               ▼              │
┌──────────────┐ ┌───────────┐ ┌─────────────┐ ┌──────────────┐       │
│   Linear     │ │ Logistic  │ │     SVM     │ │   Random     │       │
│  Regression  │ │Regression │ │    (RBF)    │ │   Forest     │       │
│   TASK 1     │ │  TASK 2   │ │   TASK 2    │ │  ENSEMBLE    │       │
└──────┬───────┘ └─────┬─────┘ └──────┬──────┘ └──────┬───────┘       │
       │               │              │               │               │
       └───────────────┴──────────────┴───────────────┘               │
                                       │                              │
                                       ▼                              │
                        ┌──────────────────────────────┐              │
                        │  Evaluation + tie analysis   │              │
                        │  metrics · CV · importance   │              │
                        └──────────────┬───────────────┘              │
                                       │                              │
                   ┌───────────────────┴──────────────────┐           │
                   ▼                                      ▼           │
        ┌────────────────────┐                ┌────────────────────┐  │
        │  models/*.pkl      │                │  results/*.csv     │  │
        │  fitted pipelines  │                │  metrics, figures  │  │
        └─────────┬──────────┘                └─────────┬──────────┘  │
                  └──────────────────┬───────────────────┘             │
                                     ▼                                │
                        ┌──────────────────────────────┐              │
                        │  app.py (Streamlit)          │◄─────────────┘
                        │  loads only — never trains   │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │  Predicted grade (0–20)      │
                        │  At Risk / Not At Risk       │
                        │  Risk probability            │
                        └──────────────────────────────┘
```

### 6.2 Separation of training and inference

The architecture enforces a strict separation between model fitting and model
use. The module `src/train_models.py` is the only component that fits an
estimator; `app.py` exclusively loads previously fitted pipelines via `joblib`.

This separation has two consequences. First, the application starts quickly and
deterministically, since no computation beyond deserialisation occurs at launch.
Second, and more importantly, the preprocessing applied to a user's form
submission is performed by the identical fitted transformer object used during
training, eliminating any possibility of divergence between training-time and
inference-time preprocessing.

Where the serialised model files are absent, the application reports the
condition and instructs the user to execute the training script. It does not
retrain implicitly.

---

## 7. Data Preprocessing

### 7.1 Transformation pipeline

All preprocessing is encapsulated in a scikit-learn `ColumnTransformer` nested
within each model's `Pipeline`:

| Branch | Features | Transformations |
|---|---|---|
| Numeric | 13 | `SimpleImputer(strategy="median")` → `StandardScaler()` |
| Categorical | 17 | `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(drop="if_binary", handle_unknown="ignore")` |

The transformation expands 30 input features to 43 columns.

**Prevention of preprocessing leakage.** Because the transformer is a pipeline
step rather than a separate preliminary operation, invoking
`pipeline.fit(X_train, y_train)` computes the scaler's means and standard
deviations and the encoder's category vocabulary from the training partition
exclusively. The test partition cannot influence these statistics. Fitting a
scaler on the complete dataset prior to partitioning is a common source of
optimistic bias in reported results; the pipeline structure renders it
impossible.

**Imputation.** The dataset contains no missing values, so the imputers are
inactive during training. They are retained because the serialised pipeline is
also used for inference on user-supplied data, where a missing field is a
realistic occurrence.

**Scaling.** Feature scaling is necessary for two of the three classifiers.
Logistic Regression's `C` parameter penalises coefficient magnitude, making the
penalty scale-dependent; the SVM's RBF kernel is a function of squared Euclidean
distance, so an unscaled `absences` attribute spanning 0–75 would contribute
squared differences up to 5625 against a maximum of 9 for `studytime` spanning
1–4. Predictions would consequently be governed by the measurement units of the
features rather than their relevance.

Random Forest is invariant to monotonic rescaling, since its splits are
threshold comparisons whose ordering is preserved under any monotonic transform.
Scaling is nonetheless applied uniformly so that a single preprocessing
specification governs all models.

**Encoding.** One-hot encoding is used for categorical attributes because integer
label encoding would impose an artificial ordinal relationship on nominal
categories — for instance implying `Mjob = teacher` exceeds `Mjob = health` —
which distance-based and penalty-based models would act upon.

The parameter `drop="if_binary"` emits a single indicator column for two-level
attributes, avoiding a pair of perfectly collinear dummies and preserving the
interpretability of the Logistic Regression coefficients. The parameter
`handle_unknown="ignore"` encodes an unseen category as a zero vector rather
than raising an exception, which is necessary for robust inference on
user-supplied input; the application detects and reports such cases.

**Treatment of ordinal attributes.** The Likert-style attributes (`Medu`, `Fedu`,
`traveltime`, `studytime`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`,
`health`) are ordered scales and are treated as numeric rather than one-hot
encoded. This preserves their ordering, limits the feature count, and permits
coefficient interpretation per unit increase. The approach assumes equal spacing
between adjacent levels, which is a simplification and is recorded as such in
Section 14.

### 7.2 Feature exclusion for leakage prevention

| Excluded attribute | Correlation with `G3` | Justification |
|---|---|---|
| `G3` | 1.000 | The regression target and the source of the classification label. Inclusion would constitute direct target leakage. |
| `at_risk` | — | The classification target, derived deterministically from `G3`. |
| `G2` | +0.905 | Second-period grade: an intermediate measurement of the outcome under prediction. Inclusion constitutes target leakage and, additionally, defeats the early-warning objective, since a grade issued at this point leaves no interval for intervention. |
| `G1` | +0.801 | First-period grade. Excluded on the same grounds. |

The exclusion is enforced programmatically. The function
`split_features_target()` in `src/data_preprocessing.py` raises an
`AssertionError` if any forbidden attribute is present in the constructed feature
matrix, causing the training run to terminate rather than produce an inflated
result.

### 7.3 Partitioning

The data is partitioned 80:20 with `random_state=42` to ensure reproducibility,
yielding 316 training and 79 test instances.

The classification partition employs `stratify=y`, preserving the class
proportion at 32.9% at-risk in both partitions. Without stratification, a random
79-instance test partition could plausibly exhibit a materially different class
balance, rendering its metrics incomparable with those of the training
distribution. The regression partition does not use stratification, the target
being continuous.

---

## 8. Algorithms

### 8.1 Multivariate Linear Regression

**Purpose.** Prediction of the final grade `G3` as a continuous quantity.

**Formulation.** The model expresses the target as a linear combination of the
input features:

```
ŷ = b₀ + b₁x₁ + b₂x₂ + ... + b₄₃x₄₃
```

where *ŷ* is the predicted grade, *x₁...x₄₃* are the encoded features, *b₀* is
the intercept and *b₁...b₄₃* are the fitted coefficients. The designation
*multivariate* refers to the plurality of input variables — thirty student
attributes, forty-three after encoding — predicting a single continuous output.

**Estimation.** Coefficients are obtained by ordinary least squares, minimising
the residual sum of squares:

```
minimise  Σᵢ (yᵢ − ŷᵢ)²
```

**Rationale for inclusion.** Linear regression is the appropriate baseline for a
continuous target: it is the least complex model capable of addressing the task,
and proceeding to a more elaborate method without first establishing its
performance would be unjustified. Its coefficients are additionally directly
interpretable.

**Observed outcome.** The model performs poorly, and this outcome is itself
informative. Section 10.1 reports a test R² of 0.1415 accompanied by a
cross-validated R² of −0.0781. The negative cross-validated value indicates
performance inferior to a constant predictor on some partitions, which
constitutes evidence that the relationship between the retained attributes and
the exact final grade is either non-linear, weak, or absent in this data.

### 8.2 Logistic Regression

**Purpose.** Classification of students as *At Risk* or *Not At Risk*.

**Formulation.** A linear combination of features is transformed by the logistic
(sigmoid) function to yield a probability:

```
z = b₀ + b₁x₁ + ... + b₄₃x₄₃

P(at risk | x) = σ(z) = 1 / (1 + e^(−z))
```

Equivalently, the model is linear in the log-odds of the positive class:

```
log( P / (1 − P) ) = b₀ + b₁x₁ + ... + b₄₃x₄₃
```

The sigmoid maps the real line onto the open interval (0, 1), ensuring the output
is a valid probability. A threshold, by default 0.5, converts the probability to
a class label.

**Estimation.** Coefficients are obtained by maximum likelihood estimation,
equivalently by minimising the logistic loss, with an L2 penalty controlled by
the regularisation parameter `C`.

**Configuration.** Grid search selected `C = 0.01`, corresponding to strong
regularisation. The parameter `class_weight="balanced"` was applied, reweighting
the minority at-risk class in inverse proportion to its frequency.

**Rationale for inclusion.** Logistic Regression is the linear baseline for
classification. It produces calibrated probabilities directly, and its signed
coefficients indicate the direction of each feature's association with the
outcome, supporting the interpretability analysis in Section 10.4.

### 8.3 Support Vector Machine

**Purpose.** Classification of students as *At Risk* or *Not At Risk*, testing
whether a non-linear decision boundary confers advantage over the linear model.

**Formulation.** The SVM seeks the separating hyperplane maximising the margin —
the perpendicular distance to the nearest instances of either class. Those
nearest instances are the *support vectors*, and they alone determine the
solution. The motivation for margin maximisation is robustness: a boundary
positioned close to training instances is sensitive to small perturbations in
unseen data, whereas a wide margin tolerates them.

Since the classes are not linearly separable, the soft-margin formulation
permits bounded violation, with the parameter `C` governing the trade-off between
margin width and training error. Small `C` admits more training errors in
exchange for a wider margin and a simpler model; large `C` penalises training
errors more heavily.

**Kernel.** The radial basis function kernel is employed:

```
K(xᵢ, xⱼ) = exp( −γ ‖xᵢ − xⱼ‖² )
```

The kernel computes inner products in an implicitly defined higher-dimensional
feature space without constructing that space explicitly — the *kernel trick* —
thereby permitting a non-linear decision boundary at the computational cost of a
linear one. The parameter γ determines the rate at which an instance's influence
decays with distance: large γ confines influence to a local neighbourhood,
producing a flexible boundary with elevated overfitting risk, while small γ
yields a smoother boundary.

**Configuration.** Grid search selected `C = 10.0` and `γ = 0.01`. The selected γ
is comparatively small, corresponding to a relatively smooth decision boundary,
which is consistent with a training set of only 316 instances.

Probability estimates are obtained via `probability=True`, which fits a Platt
scaling model internally. It should be noted that `SVC.predict()` determines its
label from the sign of the decision function rather than from the calibrated
probability, so the two may disagree for instances close to the boundary. The
application detects and explains this condition rather than presenting it as an
inconsistency.

**Observed outcome.** The non-linear kernel conferred no material advantage. The
SVM's F1 of 0.5614 is statistically indistinguishable from the linear Logistic
Regression's 0.5333 (see Section 11), suggesting that such structure as exists in
this data is not meaningfully non-linear.

### 8.4 Ensemble Learning / Random Forest

**Purpose.** Classification of students as *At Risk* or *Not At Risk*. This is
the project's designated ensemble method and its primary classification model.

#### 8.4.1 Ensemble learning

Ensemble learning denotes the combination of multiple models such that the
aggregate outperforms any constituent. The necessary condition is error
diversity: if all constituents commit identical errors, aggregation confers no
benefit. The principal families are *bagging*, in which constituents are trained
independently on resampled data and their outputs combined by voting or averaging
— reducing variance — and *boosting*, in which constituents are trained
sequentially to correct their predecessors' errors — reducing bias.

#### 8.4.2 Bagging

Bagging, from *bootstrap aggregating*, proceeds in two stages.

In the *bootstrap* stage, multiple samples are drawn from the training set with
replacement, each of the same cardinality as the original. Because sampling is
with replacement, each sample omits approximately 37% of the original instances
and duplicates others, so each constituent model is fitted to a distinct
realisation of the data.

In the *aggregation* stage, one model is fitted per sample and their predictions
are combined — by majority vote for classification, by averaging for regression.

The variance reduction achieved is greatest for high-variance base learners, of
which unpruned decision trees are the canonical example: their individual errors
are substantially uncorrelated and therefore attenuate under averaging.

#### 8.4.3 Random Forest

A Random Forest augments bagging with random feature subsetting. Each tree is
grown on its own bootstrap sample and, at every candidate split, is restricted to
a randomly selected subset of the available features.

```
        Individual Decision Tree
                   │
                   ▼
            Multiple Trees
       (bootstrap samples, with replacement)
                   │
                   ▼
               Bagging
        (train in parallel, then vote)
                   │
                   ▼
            Random Forest
    (bagging + random feature subsetting)
                   │
                   ▼
           Final Prediction
         (majority vote of 400 trees)
```

**The necessity of feature subsetting is demonstrable in this project's own
results.** The attribute `failures` dominates the importance ranking, with a
permutation importance of 0.2786 against 0.0502 for the next-ranked attribute.
Under plain bagging, the overwhelming majority of trees would select `failures`
as their initial split and the resulting trees would be highly correlated,
negating the benefit of aggregation. Restricting each split to a random subset —
here `max_features="sqrt"`, approximately 6 of 43 encoded features — compels most
trees to identify structure elsewhere, decorrelating them and rendering the
aggregation effective.

**Why a Random Forest constitutes an ensemble whereas a single decision tree does
not.** A single tree fitted to 316 instances may partition the feature space
until nearly every terminal node contains a single instance, thereby memorising
the training data. The forest fits 400 such trees and aggregates by majority
vote, so the individual overfitting substantially cancels.

**Configuration.** `n_estimators=400`, `max_features="sqrt"` and
`min_samples_leaf=3` (the latter two selected by grid search), with
`class_weight="balanced_subsample"` reweighting the minority class within each
bootstrap sample.

**Observed overfitting.** The variance reduction achieved is partial rather than
complete, and the project reports this. Train-to-test F1 degradation is as
follows:

| Model | Train F1 | Test F1 | Degradation |
|---|---|---|---|
| Logistic Regression | 0.5498 | 0.5333 | +0.0164 |
| SVM (RBF) | 0.7719 | 0.5614 | +0.2105 |
| Random Forest | 0.9763 | 0.5581 | +0.4182 |

The Random Forest exhibits a degradation of 0.4182 notwithstanding 400 trees,
feature subsetting and a minimum leaf size of 3. The accurate statement is that
the ensemble *reduced* overfitting without eliminating it. The Logistic
Regression's degradation of 0.0164 is attributable to the strong regularisation
selected by grid search (`C = 0.01`), which constrains the model's capacity to
memorise.

---

## 9. Implementation

### 9.1 Technology stack

| Component | Technology | Version tested |
|---|---|---|
| Language | Python | 3.13 |
| Data manipulation | pandas | 3.0.6 |
| Numerical computing | NumPy | 2.5.3 |
| Machine learning | scikit-learn | 1.9.1 |
| Interactive visualisation | Plotly | 7.1.0 |
| Static figures | Matplotlib, seaborn | 3.11.2, 0.13.2 |
| Web interface | Streamlit | 1.65.0 |
| Model serialisation | joblib | 1.6.0 |

No deep learning framework, container runtime, orchestration system or cloud
service is employed. The complete project executes on a standard laptop, and the
training run completes in approximately 55 seconds.

### 9.2 Module organisation

| Module | Responsibility |
|---|---|
| `src/config.py` | Paths, random seed, at-risk threshold, feature inventories, exclusion justifications, human-readable labels. |
| `src/data_preprocessing.py` | Dataset loading and shape validation, target derivation, feature/target separation with leakage assertion, construction of the `ColumnTransformer`. |
| `src/evaluate_models.py` | Regression and classification metrics, baseline computation, confusion matrices, ROC points, comparison table construction, statistical tie assessment. |
| `src/train_models.py` | The complete training pipeline; the executable entry point. |
| `src/prediction.py` | Deserialisation of fitted pipelines, input validation, single-instance prediction, identification of influential features. |
| `src/visualization.py` | All Plotly figure constructors. |
| `app.py` | Streamlit application comprising seven sections. |

A single configuration module holds all shared constants, so the feature
inventory, the at-risk threshold and the random seed each have exactly one
definition.

### 9.3 Training procedure

Execution of `python src/train_models.py` performs the following sequence:

1. Load the raw file and validate its shape against the published specification.
2. Derive the `at_risk` target from `G3`.
3. Construct the feature matrix, excluding all leakage-prone attributes, with the
   exclusion verified by assertion.
4. Partition the data 80:20, stratified for the classification task.
5. Construct the preprocessing `ColumnTransformer`.
6. Fit the Linear Regression pipeline.
7. Fit and tune the Logistic Regression pipeline.
8. Fit and tune the SVM pipeline.
9. Fit and tune the Random Forest pipeline.
10. Evaluate all models on the held-out partition and by 5-fold stratified
    cross-validation.
11. Compute impurity importance, permutation importance and Logistic Regression
    coefficients.
12. Execute the data-leakage comparison: refit all models with `G1` and `G2`
    included, evaluate, report, and discard without serialisation.
13. Serialise the fitted pipelines and metadata; write metric tables and static
    figures.

A concise training summary is printed to standard output.

### 9.4 Hyper-parameter selection

Each classifier was tuned by `GridSearchCV` over the same five stratified folds,
scored by F1 on the at-risk class.

| Model | Search space | Selected |
|---|---|---|
| Logistic Regression | `C ∈ {0.01, 0.1, 1.0, 10.0}` | `C = 0.01` |
| SVM (RBF) | `C ∈ {0.1, 1.0, 10.0}` × `γ ∈ {scale, 0.01, 0.1}` | `C = 10.0`, `γ = 0.01` |
| Random Forest | `max_features ∈ {sqrt, 0.3}` × `min_samples_leaf ∈ {1, 3, 5}` | `max_features = sqrt`, `min_samples_leaf = 3` |

The search spaces are deliberately small, comprising only those parameters that
materially alter each algorithm's behaviour. An exhaustive search would increase
runtime and explanatory burden without commensurate benefit on 316 training
instances.

**The uniform application of tuning is a methodological requirement rather than a
convenience.** An initial implementation of this project tuned only the SVM and
compared it against untuned alternatives, which rendered the comparison
uninterpretable: an observed difference could be attributed either to the
algorithm or to the tuning. Applying an identical procedure to all three models
is what permits the comparison in Section 11 to be read as a comparison of
algorithms.

### 9.5 Evaluation protocol

Two evaluation procedures are applied, and both are reported.

**Held-out test partition (79 instances).** This partition is not accessed during
training or hyper-parameter selection, and consequently provides an unbiased
estimate.

**Five-fold stratified cross-validation on the training partition.** This
procedure yields a mean and standard deviation, the latter quantifying the
sensitivity of the estimate to the choice of partition. Its necessity follows
from the size of the test partition: with 79 instances, the reclassification of a
single student alters F1 by approximately 0.02, so a single partition cannot
resolve differences of the magnitude observed between models.

The cross-validated figures are optimistic, since the same folds were used for
hyper-parameter selection. The held-out figures remain the unbiased estimate.
Both are reported; neither is presented alone.

### 9.6 Test suite

The project includes a pytest suite of 69 tests, executed with `pytest` from the
repository root. The tests are directed at the claims the documentation makes
rather than at implementation internals:

- that the leakage assertion **fires** when a grade column is introduced, rather
  than merely being present in the source;
- that the preprocessor's scaling statistics derive from the training partition
  and not from the full dataset;
- that the metrics recorded in `results/` correspond to the committed models;
- that the documented data characteristics hold, including the 38 zero-grade
  records all exhibiting zero absences and the +0.034 / −0.213 reversal in the
  absences correlation;
- that increasing the false-negative cost never raises the decision threshold;
- that the prediction interface rejects invalid input rather than failing.

The suite was validated by mutation: removing the leakage assertion, transposing
the cost weights, and altering a published metric each cause it to fail. A suite
incapable of failing provides no evidence.

### 9.7 Error handling

The application handles the following conditions without exposing a traceback:
absent or corrupted dataset file; absent, corrupted or version-incompatible model
files; non-numeric or non-finite values in numeric input fields; absent values
for required features; categorical values not present in the training data
(reported as a warning, the prediction proceeding on the remaining features); and
unanticipated exceptions during prediction or rendering, caught at the section
boundary and reported with a diagnostic message.

---

## 10. Results

All figures in this section derive from the training run and are reproduced from
`results/model_results.csv`.

### 10.1 Regression — final grade prediction

Held-out test partition, 79 students.

| Model | MAE | MSE | RMSE | R² (test) | R² (5-fold CV) |
|---|---|---|---|---|---|
| **Linear Regression** | **3.3953** | **17.6037** | **4.1957** | **0.1415** | **−0.0781 ± 0.1396** |
| Baseline (predict mean) | 3.6459 | 20.7041 | 4.5502 | −0.0097 | — |

**Interpretation.** On the held-out partition the model explains 14.15% of the
variance in the final grade and improves upon the constant predictor's RMSE,
4.1957 against 4.5502 grade points. The improvement is, however, marginal, and
the cross-validated R² of −0.0781 ± 0.1396 is negative, indicating performance
inferior to the constant predictor on a subset of folds. The positive test R² is
therefore attributable in substantial part to the particular partition obtained.

The mean absolute error of 3.3953 grade points on a 0–20 scale provides the most
interpretable characterisation: a prediction is typically in error by
approximately three and a half grade points, which on a scale where the pass mark
is 10 is not a useful degree of precision.

The conclusion is that prediction of an exact final grade from the retained
attributes, absent any prior assessment, is not achievable on this dataset. The
model correctly demonstrates the technique; it does not constitute a usable
predictor.

That RMSE (4.1957) exceeds MAE (3.3953) by a substantial margin reflects the
presence of a small number of large residuals, predominantly attributable to the
38 students recording a final grade of zero. The residual plot
(`results/figures/05_residual_plot.png`) exhibits these as a distinct band of
residuals near −10.

### 10.2 Classification — at-risk prediction

Held-out test partition, 79 students, 32.9% at risk.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 (mean ± sd) |
|---|---|---|---|---|---|---|
| SVM (RBF) | 0.6835 | 0.5161 | 0.6154 | **0.5614** | 0.7242 | 0.4602 ± 0.0381 |
| Random Forest | **0.7595** | **0.7059** | 0.4615 | 0.5581 | 0.6858 | 0.4573 ± 0.0826 |
| Logistic Regression | 0.6456 | 0.4706 | **0.6154** | 0.5333 | 0.6981 | 0.4916 ± 0.0751 |
| Baseline (majority class) | 0.6709 | 0.0000 | 0.0000 | 0.0000 | — | — |

**Interpretation.** The baseline row establishes the context for every other
figure. A classifier predicting the majority class universally attains 67.09%
accuracy while identifying none of the at-risk students, yielding precision,
recall and F1 of zero.

Against this reference, two observations follow. First, only the Random Forest
exceeds the baseline on accuracy (0.7595 against 0.6709); the Logistic Regression
and SVM fall at or below it. This is a consequence of `class_weight="balanced"`,
which deliberately exchanges accuracy for recall on the minority class. Second,
all three models exceed the baseline substantially on recall and F1, which are
the metrics relevant to the application.

ROC-AUC values between 0.6858 and 0.7242, against 0.500 for random assignment,
establish that the retained attributes carry genuine predictive information. F1
values near 0.56 establish that the magnitude of that information is modest:
approximately half of the students flagged by these models are false alarms.

Cross-validated F1 (0.4573 to 0.4916) falls below test F1 (0.5333 to 0.5614) for
all three models, indicating that the particular test partition obtained is
favourable. Reporting only the held-out figures would have overstated performance
by approximately 0.07 F1.

### 10.3 Confusion matrices

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 35 | 18 | **10** | 16 |
| SVM (RBF) | 38 | 15 | **10** | 16 |
| Random Forest | 48 | 5 | **14** | 12 |

The false-negative count is the quantity of principal concern, representing
at-risk students not identified by the model.

The matrices expose the precision–recall trade-off in concrete terms. The Random
Forest raises 17 alarms, of which 12 are correct, and misses 14 of the 26
genuinely at-risk students. The Logistic Regression raises 34 alarms, of which 16
are correct, and misses 10. The Random Forest is the more conservative model: its
superior accuracy and precision are obtained at the cost of identifying fewer of
the students the system exists to identify.

Neither configuration is unconditionally preferable. The appropriate operating
point depends on the relative cost of an unnecessary intervention against a
missed student, which is an institutional policy determination rather than a
modelling one.

### 10.4 Feature importance

Two importance measures were computed for the Random Forest.

| Rank | Impurity importance | Value | Permutation importance | Value |
|---|---|---|---|---|
| 1 | `absences` | 0.0932 | **`failures`** | **0.2786** |
| 2 | `failures` | 0.0902 | `absences` | 0.0502 |
| 3 | `goout` | 0.0571 | `goout` | 0.0321 |
| 4 | `age` | 0.0514 | `reason` | 0.0287 |
| 5 | `health` | 0.0459 | `studytime` | 0.0281 |

**The two measures disagree, and the disagreement is instructive.** Impurity
importance, obtained from scikit-learn's `feature_importances_`, aggregates the
reduction in Gini impurity attributable to each feature across all splits in all
trees. It is subject to a documented bias toward continuous and
high-cardinality features (Strobl et al., 2007), since such features present a
tree with a greater number of candidate split points. The attribute `absences`
assumes dozens of distinct values; `studytime` assumes four.

Permutation importance randomly permutes a single column of the held-out
partition and measures the resulting degradation in test F1. It addresses the
question directly — the extent to which model performance depends upon a given
feature — and is not subject to the cardinality bias.

The two measures diverge precisely as that bias predicts: impurity importance
ranks `absences` first, while permutation importance ranks `failures` first by a
factor of 5.5 over the next-ranked attribute. Where they conflict, the
permutation measure is the more reliable, and the application presents it as the
default view.

**Substantive finding.** The most informative available predictor of academic
difficulty is prior academic difficulty. Beyond `failures`, permutation
importance declines from 0.2786 to 0.0502, a reduction exceeding fivefold. No
family, lifestyle or demographic attribute in this dataset is individually a
strong predictor. This is a property of the data rather than a deficiency of the
models.

**Interpretive caution.** Feature importance characterises the attributes upon
which this particular model, fitted to 316 students at two Portuguese schools in
2008, was found to depend. It does not establish causation. It is not evidence
that reducing a student's absences would improve that student's grade. Several of
the attributes are plausibly proxies for circumstances the dataset does not
record, including household stability, employment outside school, health and the
quality of prior schooling. The appropriate formulation is that these features
were influential in the model's predictions.

### 10.5 Logistic Regression coefficients

The five largest coefficients by absolute magnitude, on the standardised feature
scale:

| Feature | Coefficient | Direction |
|---|---|---|
| `failures` | +0.2454 | increases estimated risk |
| `goout` | +0.1408 | increases estimated risk |
| `age` | +0.1149 | increases estimated risk |
| `famrel` | −0.0924 | decreases estimated risk |
| `absences` | +0.0685 | increases estimated risk |

Each coefficient represents the change in the log-odds of being at risk per
one-standard-deviation increase in the corresponding feature. Because all numeric
features were standardised within the same pipeline, the coefficients are
mutually comparable.

Two qualifications apply. First, interpretation is contingent upon the
preprocessing: these values pertain to the standardised scale rather than the
original units, and a one-hot column's coefficient is relative to the omitted
level. Second, grid search selected strong regularisation (`C = 0.01`), which
shrinks all coefficients toward zero; consequently their relative magnitudes and
signs are interpretable while their absolute magnitudes are not.

The ranking corroborates the Random Forest permutation importance: `failures` is
the dominant attribute under both models, which are otherwise unrelated in
construction.

### 10.6 Quantification of the avoided data leakage

To establish the consequence of excluding `G1` and `G2` empirically rather than
by assertion, the training script fits a parallel set of models with those
attributes included and evaluates them under an identical protocol. These models
are reported and then discarded; they are neither serialised nor used for
inference.

| Task | Model | Metric | Leakage-free (reported) | With `G1`+`G2` | Change |
|---|---|---|---|---|---|
| Regression | Linear Regression | R² | **0.1415** | 0.7241 | +0.5826 |
| Regression | Linear Regression | RMSE | **4.1957** | 2.3784 | −1.8173 |
| Classification | Logistic Regression | F1 | **0.5333** | 0.8077 | +0.2744 |
| Classification | Logistic Regression | ROC-AUC | **0.6981** | 0.9601 | +0.2620 |
| Classification | Logistic Regression | Accuracy | **0.6456** | 0.8734 | +0.2278 |
| Classification | SVM (RBF) | F1 | **0.5614** | 0.8302 | +0.2688 |
| Classification | SVM (RBF) | ROC-AUC | **0.7242** | 0.9659 | +0.2417 |
| Classification | SVM (RBF) | Accuracy | **0.6835** | 0.8861 | +0.2026 |
| Classification | Random Forest | F1 | **0.5581** | 0.8679 | +0.3098 |
| Classification | Random Forest | ROC-AUC | **0.6858** | 0.9666 | +0.2808 |
| Classification | Random Forest | Accuracy | **0.7595** | 0.9114 | +0.1519 |

**Interpretation.** Inclusion of the intermediate grades raises Random Forest F1
from 0.5581 to 0.8679 and regression R² from 0.1415 to 0.7241. A report citing
the latter column would present a considerably more favourable impression while
being considerably less informative, since the apparent improvement derives
almost entirely from the model being supplied with a close approximation of the
answer. A substantial portion of published work on this dataset reports
accuracies exceeding 90% on precisely this basis.

Two independent considerations justify the exclusion. The first is
methodological: the intermediate grades are measurements of the outcome under
prediction, so their inclusion constitutes target leakage and the resulting
metrics do not estimate performance on the stated task. The second is practical:
an early-warning system must issue its warning while intervention remains
possible, and a model requiring the second-period grade in order to predict the
third offers no such warning.

### 10.6 The decision threshold and probability calibration

Sections 10.1 to 10.5 report every classification metric at a probability
cut-off of 0.50 and present the resulting probabilities without examining them.
Both omissions are addressed here.

#### 10.6.1 The decision threshold

The 0.50 cut-off is the scikit-learn default used by `predict()`. It is a
convention rather than a derived quantity, and on this dataset it is
consequential: **76% of test
instances fall within 0.1 of it under Logistic Regression** and
33% under Random Forest. For those
instances the classification is determined by the constant rather than by the
model.

An alternative cut-off was therefore selected by minimising expected
misclassification cost, under the stated assumption that **one false negative is
equivalent in cost to 3 false positives** — the rationale
being that the cost of a false positive is a tutor's conversation with a student
who would have passed, whereas the cost of a false negative is a struggling
student receiving no intervention. The ratio is a policy parameter recorded in
`src/config.py` as `FN_COST_RATIO`; a sweep across alternative values is reported
in the application.

**Selection procedure.** The threshold is a parameter estimated from data, so
selecting it on the test partition would constitute the same methodological
error as the inclusion of `G2`. It is therefore selected from **out-of-fold
predictions on the training partition**: `cross_val_predict` refits the pipeline
on each of the five folds and predicts the held-out fold, so every probability
entering the selection originates from a model fitted without that instance. The
selected cut-off is subsequently applied, unaltered, to the held-out test
partition.

| Model | Cut-off | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0.500 (default) | 0.4706 | 0.6154 | 0.5333 | 0.6456 |
| Logistic Regression | 0.398 (cost-selected) | 0.3571 | 0.9615 | 0.5208 | 0.4177 |
| SVM (RBF) | 0.500 (default) | 0.5161 | 0.6154 | 0.5614 | 0.6835 |
| SVM (RBF) | 0.271 (cost-selected) | 0.4182 | 0.8846 | 0.5679 | 0.5570 |
| Random Forest | 0.500 (default) | 0.7059 | 0.4615 | 0.5581 | 0.7595 |
| Random Forest | 0.304 (cost-selected) | 0.3333 | 0.8077 | 0.4719 | 0.4051 |

**Interpretation.** At the default cut-off Random Forest fails to identify 14 of
the 26 genuinely at-risk instances in the test partition. At the cost-selected
cut-off its recall rises from 0.4615 to
0.8077, corresponding to approximately 5
unidentified instances.

The cost is correspondingly real: precision falls from
0.7059 to
0.3333 and accuracy from
0.7595 to
0.4051, the latter substantially below the
majority-class baseline of 0.6709. The model raises considerably more alarms.
Whether that constitutes an improvement is contingent on available intervention
capacity, which is the reason the cost ratio is an explicit and documented
parameter.

Two observations support the chosen ratio without being offered as
justification for it. First, at 3:1 the cost-optimal
cut-off coincides with the out-of-fold F1 maximum for all three models, so the
cost-based and symmetric-metric criteria agree at that point. Second, for the
SVM the selected cut-off improves F1 on the test partition as well
(0.5614 to 0.5679), so for that
model the adjustment involves no trade-off at all.

Full figures are recorded in `results/threshold_tuning.csv`.

#### 10.6.2 Probability calibration

The application presents the estimated risk as a percentage accompanied by a
meter, a presentation that implies the value is a calibrated probability. That
implication was tested using the Brier score — the mean squared error of a
probabilistic forecast — against the reference of a model which disregards its
inputs and returns the cohort base rate of 32.9%, scoring **0.2208**.

| Model | Brier score | Exceeds reference? | Mean calibration error | Observed range | Share in [0.3, 0.7] |
|---|---|---|---|---|---|
| Logistic Regression | 0.2251 | **No** | 0.1640 | 0.344 – 0.778 | 97.5% |
| SVM (RBF) | 0.1956 | Yes | 0.0967 | 0.146 – 0.737 | 59.5% |
| Random Forest | 0.1990 | Yes | 0.1303 | 0.197 – 0.733 | 81.0% |

**Interpretation.** The Logistic Regression probabilities are not merely
imprecise but **inferior to the constant base-rate forecast**, scoring
0.2251 against 0.2208. Its output spans
only 0.344 to
0.778, with
97.5% of instances between 0.3 and
0.7. The cause is identifiable: the grid search selected `C = 0.01`,
corresponding to strong regularisation, which shrinks the coefficients toward
zero and correspondingly compresses the range of the logistic output. A model
constrained never to be confident cannot produce a confident probability.

The SVM and Random Forest estimates do exceed the reference, so they carry
probabilistic information, but mean calibration errors of
0.0967 and
0.1303 indicate only
approximate calibration.

**Consequence for the system.** Every percentage the application displays should
be interpreted as an ordinal risk ranking relative to other students, and not as
an absolute probability of failure. The prediction interface states this
explicitly adjacent to the meter and identifies Logistic Regression as the
affected case. A calibration step — isotonic regression or Platt scaling fitted
on a held-out fold — is the appropriate remedy and is recorded in Section 15.

Figures are recorded in `results/calibration.csv`. The calibration curve is
constructed from 5 equal-count bins over a 79-instance test
partition, approximately 15 instances per bin, and is accordingly noisy.

### 10.7 Robustness check on the second subject file

All preceding results derive from the Mathematics file. A result obtained on a
single 395-instance dataset constitutes weak evidence, so the identical pipeline
was executed on the Portuguese-language file (649 instances) via
`python src/train_models.py --dataset por`. Complete figures are recorded in
`results/robustness_por.csv`.

| Quantity | mat (Mathematics, 395) | por (Portuguese, 649) |
|---|---|---|
| At-risk proportion | 32.9% | 15.4% |
| Majority-class baseline accuracy | 0.6709 | 0.8462 |
| Highest F1 | 0.5614 (SVM) | 0.4314 (Logistic Regression) |
| ROC-AUC range | 0.6858 – 0.7242 | 0.7550 – 0.8100 |
| Tie verdict | tie (gap 0.0033 < noise 0.0826) | tie (gap 0.0223 < noise 0.1435) |
| Leading permutation-importance feature | `failures` (0.2786) | `failures` (0.0648) |
| Regression R² (test) | 0.1415 | 0.1602 |
| Regression R² (5-fold CV) | **−0.0781** | **+0.2707** |

**Three principal findings replicate**, which materially strengthens the
confidence that may be placed in them.

First, **the statistical tie holds on both files**. In each case the margin
separating the best and second-best classifier falls well inside the
cross-validation standard deviation. The conclusion that these three algorithms
are not distinguishable on this problem is therefore not an artefact of a single
partition of a single dataset.

Second, **`failures` is the leading feature under both cohorts**, by permutation
importance, confirming prior academic failure as the most informative available
predictor.

Third, **the limitations of accuracy as a metric are more pronounced, not less**.
On the Portuguese file no model exceeds the majority-class baseline accuracy at
all — 0.7769, 0.8000 and 0.8231 against 0.8462 — while all three attain a
ROC-AUC between 0.7550 and 0.8100. Ranking by accuracy would lead to the
conclusion that all three models are without value, which the ROC-AUC figures
contradict.

**One conclusion, however, does not generalise, and the qualification is
material.** Section 10.1 characterises the regression task as intractable on the
basis of a cross-validated R² of −0.0781. On the Portuguese file the same model
attains a cross-validated R² of **+0.2707 ± 0.0685**, with root mean squared
error falling from 4.1957 to 2.8618 grade points. The claim that a final grade
cannot be predicted from background attributes alone is therefore too strong as a
general proposition. The accurate statement is that it cannot be predicted *on
the Mathematics cohort*, whose 38 zero-grade records and wider grade dispersion
render the target unusually difficult, and that on a larger cohort with a less
dispersed target the same linear model carries modest but genuine predictive
signal.

The comparison additionally provides a clear illustration of the divergence
between F1 and ROC-AUC under class imbalance. Reducing the positive-class
proportion from 32.9% to 15.4% lowers every F1 score while raising every
ROC-AUC. F1 is evaluated at a fixed 0.5 threshold against a rarer class and is
consequently sensitive to the base rate; ROC-AUC assesses the ranking across all
thresholds and is substantially less so.

The two files remain uncombined throughout, for the reason given in Section 5.6.

---

## 11. Model Comparison

### 11.1 Ranking and its statistical significance

Ranked by F1 on the at-risk class, the ordering is SVM (0.5614), Random Forest
(0.5581), Logistic Regression (0.5333).

This ordering should not be relied upon, and the project establishes this
programmatically rather than by assertion. The function `assess_tie()` in
`src/evaluate_models.py` compares the margin between the leading models against
the cross-validation standard deviation, which is a direct estimate of the
sampling variability of the metric.

| Quantity | Value |
|---|---|
| F1, best model (SVM) | 0.5614 |
| F1, second model (Random Forest) | 0.5581 |
| **Margin** | **0.0033** |
| **Cross-validation standard deviation (maximum)** | **0.0826** |
| Ratio of noise to margin | 25.0 |
| Test partition size | 79 |

The margin separating first and second place is 0.0033. The cross-validation
standard deviation is 0.0826, exceeding the margin by a factor of twenty-five. On
a test partition of 79 instances, the reclassification of two students alters F1
by more than the entire observed margin.

The defensible conclusion is therefore:

> On this dataset, Logistic Regression, an RBF-kernel Support Vector Machine and
> a Random Forest ensemble perform equivalently at the task of identifying
> at-risk students. The differences between them are smaller than the variation
> observed between cross-validation folds, and selection between them on this
> evidence would constitute over-interpretation.

**Random Forest is retained as the project's primary model** on grounds
independent of the 0.0033 margin: it satisfies the ensemble-learning requirement,
it attains the highest accuracy (0.7595) and precision (0.7059) of the three
models, and it provides the feature-importance analysis. It is not retained on
the grounds of having outperformed the alternatives, which it did not.

Supporting evidence for the tie verdict is available from the tuning process
itself. Prior to hyper-parameter selection the Logistic Regression attained F1
0.5806; after grid search selected `C = 0.01` it attained 0.5333. A procedure
that moves a single model by 0.047 F1 cannot resolve a 0.0033 difference between
models.

### 11.2 Interpretation of the equivalence

The expectation prior to the experiment is that the ensemble method should
outperform the linear model, and that the kernel method should outperform both.
Neither expectation is borne out. A heavily regularised Logistic Regression —
`C = 0.01`, the least complex model in the project — performs equivalently to a
400-tree Random Forest and to an RBF-kernel SVM.

The explanation is that the limiting factor is the information content of the
data rather than the capacity of the learner. With 316 training instances and a
single dominant predictor, there is insufficient exploitable structure for
additional model capacity to capture. This is consistent with the train-to-test
degradation reported in Section 8.4: the more flexible models fit the training
data substantially better (Random Forest train F1 0.9763) without any
corresponding improvement in generalisation.

The practical implication, developed in Section 15, is that effort directed at
improved features would be more productive than effort directed at improved
algorithms.

### 11.3 Regression comparison

A single regression model is reported. The regression component of this project
exists to demonstrate Multivariate Linear Regression, and the introduction of
additional regressors — ridge regression, lasso, a regression tree — purely to
populate a comparison table would demonstrate nothing not already established by
the classification comparison, while increasing the project's explanatory burden.

The comparison that is material is against the constant predictor, and it is
reported in Section 10.1: the linear model improves upon predicting the cohort
mean on the held-out partition and fails to do so under cross-validation.

---

## 12. Screenshots

The application comprises seven sections. Screenshots are located in
`docs/screenshots/`.

| Figure | File | Content |
|---|---|---|
| 1 | `01_overview.png` | Headline metrics, interpretation panels, class and grade distributions, pipeline diagram |
| 2 | `02_student_prediction.png` | Input form, predicted grade gauge, risk verdict, probability meter, input profile, influential features |
| 3 | `03_data_analysis.png` | Grade distribution, class balance, per-feature box plots, absences scatter, correlation heatmap |
| 4 | `04_model_comparison.png` | Metric table with baseline, grouped comparison chart, cross-validation stability, selectable confusion matrix, ROC curves, regression diagnostics |
| 5 | `05_model_explainability.png` | Permutation and impurity importance, Logistic Regression coefficients, interpretation |
| 6 | `06_data_leakage.png` | Exclusion justifications, leakage comparison chart and table |
| 7 | `07_about_the_project.png` | Problem statement, methodology, ensemble explanation, limitations, ethics, references |

Static figures for inclusion in printed documentation are located in
`results/figures/`:

| File | Content |
|---|---|
| `01_score_distribution.png` | Histogram of final grades with at-risk threshold |
| `02_class_distribution.png` | At-risk class balance |
| `03_correlation_heatmap.png` | Correlation among numeric features and grades |
| `04_actual_vs_predicted.png` | Regression actual against predicted |
| `05_residual_plot.png` | Regression residuals |
| `06_confusion_matrices.png` | Confusion matrices for all three classifiers |
| `07_model_comparison.png` | Grouped metric comparison |
| `08_feature_importance.png` | Random Forest impurity importance |
| `09_roc_curves.png` | ROC curves for all three classifiers |

---

## 13. Advantages

**Methodological integrity.** All leakage-prone attributes are excluded, the
exclusion is enforced by assertion rather than convention, and the cost of the
exclusion is quantified. Preprocessing is encapsulated within pipelines, so
training-partition statistics cannot be contaminated by the test partition.

**Fair model comparison.** All three classifiers are tuned over identical
cross-validation folds using an identical scoring function, so observed
differences are attributable to the algorithms rather than to unequal tuning
effort.

**Statistically honest reporting.** The margin between models is tested against
the fold-to-fold variation before any ranking is asserted, and a statistical tie
is reported where the margin falls within the noise. Baselines are reported for
both tasks, so no metric is presented without the reference point required to
interpret it.

**Dual-measure explainability.** Two independent importance measures are computed
and their disagreement is identified and explained rather than suppressed.
Causal interpretation is explicitly disclaimed throughout.

**Reproducibility.** A fixed random seed, committed data files, committed result
tables and documented library versions permit exact reproduction. The training
script executes in approximately 55 seconds on commodity hardware.

**Appropriate scope.** The project employs four classical algorithms implemented
with scikit-learn. No deep learning framework, container runtime or cloud service
is introduced, and every component is explicable in a viva examination.

**Separation of concerns.** Training and inference are strictly separated: the
application loads serialised pipelines and never fits a model, so the
preprocessing applied at inference is performed by the identical fitted object
used during training.

**Robust error handling.** Absent or corrupted data and model files, invalid
input and unseen categorical values are each handled with a diagnostic message
rather than a traceback.

---

## 14. Limitations

**Dataset size and age.** The dataset comprises 395 students at two Portuguese
schools, collected in 2008. The test partition of 79 instances means the
reclassification of a single student alters F1 by approximately 0.02. No claim of
transferability to other educational systems or periods is supported.

**Modest predictive performance.** F1 scores near 0.56 indicate that
approximately half of flagged students are false alarms. The cross-validated
regression R² of −0.0781 indicates performance no better than predicting the
cohort mean.

**The zero-grade subpopulation.** Thirty-eight students recorded a final grade of
zero while simultaneously recording zero absences and a non-zero first-period
grade, a combination more consistent with withdrawal or an unrecorded mark than
with genuine assessment. Their retention is the defensible choice but distorts
both tasks: the target distribution becomes bimodal, the regression model incurs
a cluster of large residuals, and at least one feature correlation is suppressed
in the aggregate (Section 5.4(b)).

**Model equivalence.** The three classifiers are statistically indistinguishable,
so no claim of algorithmic superiority is supported by these results.

**Self-reported attributes.** Study time, alcohol consumption, free time and
family relationship quality derive from student questionnaires and carry the
associated measurement error.

**Absence of causal warrant.** The analysis establishes association only. No
result supports a claim that modifying a feature would modify an outcome.

**Probability calibration.** The estimates are only approximately calibrated,
and the Logistic Regression estimates are inferior to a constant base-rate
forecast (Section 10.6.2). The percentages the application displays are ordinal
risk rankings, not absolute probabilities.

**The cost ratio is a judgement.** The 3:1 false-negative to false-positive
ratio underlying the cost-selected threshold is documented and swept, but it is
an assumption about institutional priorities rather than a finding.

**Ordinal encoding assumption.** The Likert-style attributes are treated as
evenly spaced numeric values, which assumes the interval between adjacent levels
is constant.

**Optimistic cross-validation estimates.** Hyper-parameters were selected using
the same folds from which cross-validated scores are reported, so those scores
are optimistic. The held-out estimates are unbiased.

**No fairness audit.** The models use sensitive attributes including sex, family
size, parental cohabitation status and parental education. No analysis of
differential error rates across these groups was performed, and such an analysis
would be a prerequisite for any operational use.

**Single train/test partition for the headline figures.** Although
cross-validation is reported alongside, a repeated or nested cross-validation
protocol would yield more stable estimates. This was not undertaken, in the
interest of keeping the methodology explicable.

---

## 15. Future Scope

**Threshold optimisation against an explicit cost model.** Given the dispersion
of operating points observed in Section 10.3 — Random Forest at precision 0.7059
and recall 0.4615, Logistic Regression at 0.4706 and 0.6154 — selecting the
decision threshold from a stated ratio of the cost of a missed student to the
cost of a false alarm would likely improve practical utility more than any change
of algorithm, at negligible implementation cost.

**Temporal engagement features.** The most promising direction for improved
performance is better features rather than better models. Weekly submission
records, learning-management-system access logs and assignment timeliness would
permit the modelling of trajectories rather than annual totals. The absence
finding in Section 5.4(b) supports this directly: an annual total proved
susceptible to a record-keeping artefact that suppressed a genuine relationship,
whereas a within-year trend would be considerably harder to distort in the same
way.

**Larger and more diverse data.** Records from multiple institutions, education
systems and cohorts would permit assessment of generalisation, which the present
dataset cannot support.

**Deployment as a periodic screening instrument.** An appropriate operational
form is a termly report that ranks students by estimated risk for a tutor's
attention, explicitly framed as prioritisation rather than determination.

**Per-instance explanation.** Shapley additive explanations or an equivalent
local attribution method would permit a flagged student to be accompanied by the
specific attribute values that elevated the estimate, rather than by global
importance rankings alone.

**Additional ensemble methods.** Gradient boosting would permit assessment of
whether a bias-reducing ensemble extracts more from this weak signal than the
variance-reducing ensemble employed here.

**Probability calibration analysis.** The reported probabilities are presently
taken at face value, and those of the SVM are Platt-scaled approximations. A
calibration curve would establish whether a stated 70% risk corresponds to a 70%
observed failure rate.

**Fairness assessment.** Differential error rates across the sensitive attributes
present in the feature set should be measured, and mitigation applied where
disparities are identified.

---

## 16. Conclusion

This project implemented and compared four machine learning algorithms on a real
dataset of secondary-school student performance, subject to the constraint that
no model may receive any measurement of the predicted outcome as an input. The
findings are as follows.

**The at-risk classification task is partially tractable.** All three classifiers
attained ROC-AUC between 0.6858 and 0.7242 against 0.500 for random assignment,
establishing that student background and behavioural attributes carry genuine
information about academic difficulty. At F1 near 0.56, the resulting system is
appropriate as a screening instrument for prioritising institutional attention
and is not appropriate as a basis for determination.

**The grade-regression task is not tractable on the Mathematics cohort.** The
cross-validated R² of −0.0781 indicates performance no better than predicting the
cohort mean, and the mean absolute error of 3.3953 grade points on a 0–20 scale
is not a useful degree of precision. Establishing this cleanly, rather than
reporting the more favourable single-partition R² of 0.1415 without
qualification, is itself a legitimate result. The robustness check of Section
10.7 bounds the claim, however: on the 649-instance Portuguese file the same
model attains a cross-validated R² of +0.2707, so the finding is a property of
this cohort rather than of the problem in general.

**Model complexity conferred no advantage.** A Logistic Regression with strong
regularisation performed equivalently to a 400-tree Random Forest and to an
RBF-kernel Support Vector Machine; the three models are statistically
indistinguishable, with a margin of 0.0033 against a cross-validation standard
deviation of 0.0826. Where the signal is weak and the sample small, the limiting
factor is the data rather than the learner.

**Prior academic failure is the dominant predictor.** Permutation importance
assigns `failures` an importance of 0.2786 against 0.0502 for the next-ranked
attribute, a ratio of 5.5. This was corroborated independently by the Logistic
Regression coefficients. The accompanying methodological finding is that the
Random Forest's built-in impurity importance ranks `absences` first instead, an
instance of the documented cardinality bias arising in this project's own
results.

**The majority of the apparently attainable performance was leakage.** Inclusion
of the intermediate grades `G1` and `G2` would have raised Random Forest F1 from
0.5581 to 0.8679 and regression R² from 0.1415 to 0.7241. The refusal of that
performance is the project's central methodological decision, and Section 10.6
quantifies precisely what it cost.

All machine learning concepts required by the syllabus are demonstrated:
multivariate linear regression, logistic regression, support vector machines with
kernel methods, ensemble learning by bagging, preprocessing pipelines,
stratified train/test partitioning, cross-validation, confusion matrices,
regression and classification metrics, feature importance, and data
visualisation.

The project's distinguishing characteristic is its treatment of a weak result as
a finding requiring explanation rather than a quantity requiring inflation. The
more favourable figures were available, were measured, and were declined, on
grounds that are documented and defensible.

---

## 17. References

1. P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School Student
   Performance." In A. Brito and J. Teixeira (eds.), *Proceedings of 5th FUture
   BUsiness TEChnology Conference (FUBUTEC 2008)*, pp. 5–12, Porto, Portugal,
   April 2008. ISBN 978-9077381-39-7.

2. Student Performance dataset. UCI Machine Learning Repository.
   <https://archive.ics.uci.edu/dataset/320/student+performance>.
   DOI: 10.24432/C5TG7T.

3. L. Breiman. "Random Forests." *Machine Learning*, 45(1):5–32, 2001.

4. L. Breiman. "Bagging Predictors." *Machine Learning*, 24(2):123–140, 1996.

5. C. Cortes and V. Vapnik. "Support-Vector Networks." *Machine Learning*,
   20(3):273–297, 1995.

6. F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel,
   M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos,
   D. Cournapeau, M. Brucher, M. Perrot and É. Duchesnay. "Scikit-learn: Machine
   Learning in Python." *Journal of Machine Learning Research*, 12:2825–2830,
   2011.

7. C. Strobl, A.-L. Boulesteix, A. Zeileis and T. Hothorn. "Bias in Random Forest
   Variable Importance Measures: Illustrations, Sources and a Solution." *BMC
   Bioinformatics*, 8:25, 2007.

8. A. Fisher, C. Rudin and F. Dominici. "All Models are Wrong, but Many are
   Useful: Learning a Variable's Importance by Studying an Entire Class of
   Prediction Models Simultaneously." *Journal of Machine Learning Research*,
   20(177):1–81, 2019.

9. J. Platt. "Probabilistic Outputs for Support Vector Machines and Comparisons
   to Regularized Likelihood Methods." In *Advances in Large Margin
   Classifiers*, MIT Press, pp. 61–74, 1999.

10. S. Kaufman, S. Rosset and C. Perlich. "Leakage in Data Mining: Formulation,
    Detection, and Avoidance." *ACM Transactions on Knowledge Discovery from
    Data*, 6(4):1–21, 2012.

---

**Disclaimer.** This report documents a university Machine Learning mini-project
undertaken for academic demonstration and assessment. The system described is
not fit for operational use in any educational institution. Its predictions
describe statistical associations within a small historical dataset; they do not
establish causation and are not of sufficient accuracy to inform decisions
concerning individual students.
