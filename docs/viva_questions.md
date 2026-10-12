# Viva Questions & Answers

Used Car Price & Premium Prediction System. Every figure below is from
`results/` (test set of 1,386 cars unless stated). Answers are short on purpose:
say the one-sentence version first, then the number.

---

## Fundamentals

### 1. What is the project in one sentence?
It predicts a used car's selling price (regression) and whether it is Premium,
meaning priced at ₹6 lakh or more (classification), using 6,926 de-duplicated
Indian listings and four models: Linear Regression, Logistic Regression, SVM and
Random Forest.

### 2. Regression vs classification — which task is which here?
Regression predicts a number (price in rupees). Classification predicts a class
(Premium or Standard). The class is *derived* from the price: price ≥ 6 lakh.

### 3. Is that a weakness?
Yes, and I say so. The label is a deterministic function of the regression target
and ₹6 lakh is a round number I chose before fitting anything, not an industry
definition. The classification task adds little information beyond regression; it
is included because the syllabus needs all four algorithms, and a price band is a
legitimate filter.

### 4. Parameter vs hyper-parameter?
A parameter is learned from data (a regression coefficient). A hyper-parameter is
set before training (`C`, `gamma`, number of trees). I chose hyper-parameters with
`GridSearchCV` on the training split only.

---

## The four algorithms

### 5. Multivariate Linear Regression — what is it and why "multivariate"?
It fits `y = b0 + b1·x1 + … + bk·xk` by least squares; "multivariate" because
there are many inputs (11 here, 36 encoded columns). I fit it to **log price**, so a
coefficient reads as a percentage change in price.

### 6. Why log price?
Prices are right-skewed (skew 5.57; median ₹4.0 lakh, maximum ₹100 lakh). On the
log scale the skew is −0.16, errors become proportional, and predictions can never
be negative. Predictions are converted back with `exp` using
`TransformedTargetRegressor`.

### 7. What does a coefficient like `age = −0.452` mean?
Inputs are standardised, so one standard deviation more age multiplies the
predicted price by exp(−0.452) ≈ 0.64 with the other inputs fixed. It is an
association in listings, not a causal effect.

### 8. Logistic Regression vs Linear Regression?
Logistic Regression passes a linear score through a sigmoid to give P(Premium)
between 0 and 1 and is fitted by maximum likelihood, not least squares. It is
called "regression" because it regresses the log-odds on the inputs.

### 9. What is an SVM actually optimising?
The maximum-margin boundary between classes, with a penalty `C` for points on the
wrong side. Large `C` fits the training data harder; small `C` tolerates errors.
Tuned: `C = 1`, `gamma = 0.1`.

### 10. What does the RBF kernel do?
It measures similarity as `exp(−gamma·‖x−x′‖²)`, which lets a linear separator in
an implicit high-dimensional space become a curved boundary in the original one.
`gamma` is how local the influence of each training car is.

### 11. Why does SVM need scaling but Random Forest does not?
The RBF kernel uses squared distance, so unscaled `km_driven` (up to 2.4 million)
would drown `seats` (2–14). Trees split on one feature at a time and only care
about order, so scale is irrelevant. Both share the same preprocessor anyway.

---

## Ensemble learning

### 12. What is ensemble learning, and bagging?
Combining many models to beat one. **Bagging** trains each model on a bootstrap
sample (rows drawn with replacement) and averages the predictions, which reduces
variance.

### 13. What does Random Forest add to bagging?
Each split considers only a random subset of features (`sqrt` here), so trees are
decorrelated and averaging them helps more. I use 400 trees with
`min_samples_leaf = 3`.

### 14. Did the ensemble beat the simpler models?
Barely, and not provably. F1: Random Forest 0.8512, SVM 0.8502, Logistic
Regression 0.8411. The RF–SVM gap (0.0009) is far inside the CV noise (0.0084), so
they are a **statistical tie**. Logistic Regression is 0.0101 behind — just over
the noise. A simple linear model gets within a point of the ensemble.

### 15. Why keep Random Forest then?
As a tie-break: slightly higher ROC-AUC (0.9736 vs 0.9695), the best Brier score
(0.0588) and the lowest fold-to-fold variability. Not because it has been shown to
win.

---

## Evaluation

### 16. Why a train/test split, and why stratified?
The test set simulates unseen cars; fitting on it would inflate scores.
Stratification keeps the Premium share equal: 29.48% in train and 29.44% in test.
Split is 80/20, `random_state = 42`: 5,540 train, 1,386 test.

### 17. Why cross-validation when you have a test set?
The test set is for one final honest number. CV on the training split gives a
stability estimate and drives hyper-parameter selection without touching the test
set. Random Forest CV F1 is 0.8629 ± 0.0059 against a test F1 of 0.8512.

### 18. What is a confusion matrix? Show one.
Counts of true/false positives and negatives. Random Forest (positive = Premium):
TN 919, FP 59, FN 62, TP 346.

### 19. Precision, recall, F1 — which matters?
Precision: of cars called Premium, how many are. Recall: of real Premium cars, how
many were found. F1 is their harmonic mean. With symmetric costs, F1 is a fair
single number: Random Forest has precision 0.8543, recall 0.8480.

### 20. Why can accuracy mislead?
70.5% of cars are Standard, so always answering "Standard" scores 0.7056 accuracy
with recall 0. My models score 0.906–0.913, which is +0.20 over that floor — the
margin, not the raw number, is the result.

### 21. What is ROC-AUC?
The probability that a random Premium car is ranked above a random Standard one;
0.5 is guessing. Random Forest 0.9736, SVM 0.9695, Logistic Regression 0.9611.

### 22. MAE, RMSE, R², MedAPE?
MAE: average absolute error, 0.92 lakh. RMSE: penalises large errors, 1.76 lakh
(almost double MAE because a few expensive cars have big errors). R²: share of
variance explained, 0.859 (mean predictor: −0.002). MedAPE: median error as a % of
price, 15.97%. MedAPE matters because a 1-lakh error is huge on a 2-lakh car and
trivial on a 40-lakh one.

### 23. Is R² = 0.86 good?
Good relative to the baseline (−0.002), and the log-scale R² is 0.869. But a typical
error of 16% means it is an estimator, not a quote, and the CV R² varies from 0.825
to 0.909 across folds because a few luxury cars dominate the variance.

---

## Data leakage and preprocessing

### 24. What is data leakage? Give your example.
Information from outside the training data (or from the answer) reaching the
model. Two forms here: (1) target-derived columns — `selling_price`, `is_premium`,
`name`, `year` are excluded, enforced by an assertion with a test proving it fires;
(2) **duplicate rows** — 1,202 of 8,128 (14.8%) are exact copies.

### 25. How did duplicates matter? Is it measured?
Yes, measured. With duplicates kept, 332 of 1,626 test cars (20.4%) have an exact
twin in training, and Random Forest F1 rises from 0.8512 to 0.8988, SVM from 0.8502
to 0.8836, Logistic Regression from 0.8411 to 0.8639. Flexible models gain most
because they can memorise. I report the de-duplicated numbers.

### 26. Is anything odd in the duplicate results?
Yes: Linear Regression's MAE gets *worse* with duplicates kept (0.92 → 1.21 lakh)
while R² improves (0.859 → 0.894). The two test sets differ in size and
composition, so they are not a like-for-like pair, and I have not isolated why they
disagree. I would not over-interpret the regression side of that comparison.

### 27. What else prevents leakage?
The preprocessor (median imputer, scaler, encoder) lives inside each Pipeline, so
it is fitted on training rows only; a test confirms the scaler mean equals the
training mean, not the full-data mean. The decision threshold is chosen on
out-of-fold training predictions, never on the test set.

### 28. How are missing values handled?
208 cars lack mileage, engine and seats together and 206 lack `max_power` (stored
as text; one value is a blank). They are imputed with the median inside the
pipeline. Median, not mean, because ten-million-rupee cars would pull a mean.

### 29. Why one-hot encoding? What does `min_frequency=30` do?
Brands have no order, so one binary column per brand avoids implying one.
`min_frequency=30` pools any brand with fewer than 30 training cars — and any brand
never seen — into one "infrequent" column, preventing 32 mostly tiny columns from
overfitting and letting an unknown brand typed in the app degrade gracefully.

---

## Interpretation, calibration, limitations

### 30. What are the most important features?
Permutation importance (drop in F1): `age` 0.243, `max_power` 0.127, then `fuel`
0.032, `engine` 0.031, `brand` 0.023. Linear coefficients agree on age.

### 31. Do impurity and permutation importance agree?
On the top two, yes (`age` 0.255 and `max_power` 0.219 by impurity). They disagree
on `km_driven` (0.081 impurity, 0.0035 permutation) and `mileage` (0.080 vs 0.0054).
Impurity importance favours continuous features with many split points, so I trust
permutation importance more.

### 32. Where does the Random Forest overfit?
Train F1 is 0.9172 against test 0.8512. That generalisation gap is normal for a
forest and is why only held-out numbers are reported. Logistic Regression has none
(train 0.8373, test 0.8411).

### 33. What is the decision threshold, and did tuning help?
The cut-off on P(Premium) above which a car is called Premium. 0.5 is a default.
I choose it on out-of-fold training predictions to minimise expected cost. With a
1:1 cost ratio (nothing about car pricing justifies another) it lands at 0.4941 /
0.4771 / 0.5007 and the test F1 is unchanged within noise (e.g. Random Forest
0.8512 → 0.8498). It is a negative result and I report it as one.

### 34. Are the probabilities reliable?
Reasonably. Brier scores 0.0705 / 0.0636 / 0.0588 (Logistic / SVM / Random Forest)
against 0.2077 for a constant base-rate predictor, with small mean calibration
errors. The SVM's probabilities are Platt-scaled, so near the boundary its label
and probability can disagree; the app flags that.

### 35. What would you not claim?
That Random Forest is better than SVM; that brand coefficients are causal; that
`owner = Test Drive Car` (+0.678, only 5 cars) means anything; that predictions are
a valuation. The data is a scrape with no official specification or confirmed
licence, and prices are asking prices from before 2021.

### 36. What would you do next?
Gradient boosting and a Random Forest regressor to give the price task a
competitor; prediction intervals instead of a single number; SHAP for per-car
explanations; verified, dated and located data.

---

## Quick numbers to remember

| Item | Value |
|---|---|
| Raw rows / duplicates / cleaned | 8,128 / 1,202 (14.8%) / 6,926 |
| Premium share | 29.47% (2,041 cars) |
| Train / test | 5,540 / 1,386 |
| Majority-class accuracy | 0.7056 |
| Regression R² / MAE / MedAPE | 0.859 / 0.92 lakh / 15.97% |
| Mean-predictor R² | −0.002 |
| F1: RF / SVM / LR | 0.851 / 0.850 / 0.841 |
| ROC-AUC: RF / SVM / LR | 0.974 / 0.970 / 0.961 |
| Tie gap vs noise | 0.0009 vs 0.0084 |
| Training time | ≈ 196 s |
| Tests | 81 |
