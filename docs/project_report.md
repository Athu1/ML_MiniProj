# Used Car Price & Premium Prediction System — Project Report

*Machine Learning mini-project. The formatted Word version for submission is
`Mini_Project_Report.docx`; the full technical write-up with every table is in
`README.md`. All figures are from `results/` (held-out test set of 1,386 cars).*

## Abstract

This project estimates the asking price of a used car from its listing
attributes (regression) and classifies it as Premium (≥ ₹6 lakh) or Standard,
using 8,128 public Indian used-car listings. Four algorithms are compared:
Multivariate Linear Regression (on log price), Logistic Regression, an RBF
Support Vector Machine and a Random Forest ensemble. 1,202 rows (14.8%) were
exact duplicates and were removed before splitting, leaving 6,926 cars. The
regression reaches R² 0.8589 and a median error of 15.97% (mean predictor:
−0.0020 and 46.24%). The classifiers reach F1 0.8512 / 0.8502 / 0.8411 (Random
Forest / SVM / Logistic Regression) and ROC-AUC 0.9736 / 0.9695 / 0.9611 against a
majority-class accuracy of 0.7056. Random Forest and SVM are a statistical tie.

## 1. Introduction

Used-car prices are hard to judge. The project asks how well the asking price can
be estimated from age, distance driven, engine, brand and seller details, and is
careful about evaluation: no price-derived column is an input, duplicates are
removed before the split, and differences between models are tested against
cross-validation noise.

**Problem statement.** Given 11 listing attributes (excluding the price and
anything derived from it), estimate the price and classify the car as Premium or
Standard.

**Objectives.** Validate a public dataset; explore and clean it; build a
leakage-free pipeline; implement the four required algorithms; evaluate against
baselines; analyse feature importance; deliver a web application; report
limitations honestly.

## 2. Methodology

| Step | Detail |
|---|---|
| Data | `cardekho.csv`, 8,128 × 12, SHA-256 checked; no official specification or confirmed licence |
| Cleaning | drop 1,202 duplicates; coerce `max_power`; `age = 2020 − year`; `brand` from `name` |
| Target | `selling_price` (modelled as log price); `is_premium = price ≥ 600,000` (29.47% premium) |
| Excluded inputs | `selling_price`, `is_premium`, `name`, `year` (assertion-enforced) |
| Split | stratified 80/20, seed 42: 5,540 train / 1,386 test |
| Preprocessing | in-pipeline: median impute + scale numerics; one-hot categoricals; brands under 30 training cars pooled |
| Tuning | `GridSearchCV`, 5 folds, training split only, F1 on Premium |
| Threshold | chosen on out-of-fold training predictions (1:1 error cost) |

## 3. Results

### Regression (lakh rupees)

| Model | MAE | RMSE | R² | MedAPE | CV R² |
|---|---|---|---|---|---|
| Linear Regression (log price) | 0.9201 | 1.7593 | 0.8589 | 15.97% | 0.8527 ± 0.0320 |
| Baseline: predict mean | 2.7531 | 4.6879 | −0.0020 | 46.24% | — |

### Classification

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 |
|---|---|---|---|---|---|---|
| Random Forest | 0.9127 | 0.8543 | 0.8480 | 0.8512 | 0.9736 | 0.8629 ± 0.0059 |
| SVM (RBF) | 0.9105 | 0.8381 | 0.8627 | 0.8502 | 0.9695 | 0.8550 ± 0.0047 |
| Logistic Regression | 0.9062 | 0.8390 | 0.8431 | 0.8411 | 0.9611 | 0.8320 ± 0.0084 |
| Majority class | 0.7056 | 0 | 0 | 0 | — | — |

Random Forest vs SVM: F1 gap 0.0009 against CV noise 0.0084 — a statistical tie.

### Duplicate leakage

With duplicates kept, 332 of 1,626 test cars (20.4%) have an exact twin in the
training set, and Random Forest F1 rises from 0.8512 to 0.8988 (SVM 0.8502 →
0.8836, Logistic Regression 0.8411 → 0.8639). The regression MAE moves the
opposite way to R² (0.92 → 1.21 lakh while R² improves), the two test sets are not
like-for-like, and the reason was not isolated.

### Feature importance

Permutation importance: `age` 0.2433, `max_power` 0.1271, `fuel` 0.0317,
`engine` 0.0307, `brand` 0.0230. Impurity importance agrees on the top two but
ranks `km_driven` (0.0811) and `mileage` (0.0804) 4th and 5th where permutation
importance puts them near zero — the known bias of impurity importance toward
continuous features.

### Threshold and calibration

Cost-tuned thresholds 0.4941 / 0.4771 / 0.5007 (Logistic / SVM / Random Forest)
change test F1 by at most 0.0014: a negative result. Brier scores 0.0705 / 0.0636 /
0.0588 against 0.2077 for a constant predictor.

## 4. Conclusion and future scope

Price is predictable from listing attributes and three classifiers are close;
age and engine power dominate; keeping duplicate rows would have inflated every
classifier. Limitations: scraped data of uncertain provenance and asking prices
only; the premium label is a deterministic function of price; rare categories are
uninterpretable; CV scores are slightly optimistic. Future work: gradient
boosting and a Random Forest regressor, prediction intervals, SHAP, verified and
dated data.

## References

1. L. Breiman, "Random Forests," *Machine Learning* 45(1):5–32, 2001.
2. L. Breiman, "Bagging Predictors," *Machine Learning* 24(2):123–140, 1996.
3. C. Cortes and V. Vapnik, "Support-Vector Networks," *Machine Learning* 20:273–297, 1995.
4. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR* 12:2825–2830, 2011.
5. C. Strobl, A.-L. Boulesteix, A. Zeileis, T. Hothorn, "Bias in random forest variable importance measures: Illustrations, sources and a solution," *BMC Bioinformatics* 8:25, 2007.
