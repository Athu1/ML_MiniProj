# Viva Questions & Answers

Thirty questions, grouped from the general to the specific, with answers drawn
from **this project's actual results**. The numbers here all come from
`results/model_results.csv`.

Start with ["Explain this project in 60 seconds"](#explain-this-project-in-60-seconds)
at the bottom, then [the five hardest questions](#the-five-questions-most-likely-to-catch-you-out).

---

## Fundamentals

### 1. What is supervised learning?

Learning a mapping from inputs to outputs using examples where the correct
output is already known. The model sees labelled pairs `(X, y)` during training
and learns a function that generalises to unseen `X`.

In this project both tasks are supervised: we have 395 students whose final
grades are already recorded, so every training example comes with its correct
answer. The contrast is unsupervised learning (clustering students into groups
with no labels) and reinforcement learning (learning from rewards).

### 2. What is regression, and what is classification?

Both are supervised, and the difference is the **type of the target**.

- **Regression** predicts a *continuous* number. Here: the final grade `G3`,
  anywhere from 0 to 20. A prediction of 11.3 is meaningful.
- **Classification** predicts a *discrete class*. Here: At Risk or Not At Risk.
  There is no "1.5th class".

This project does both on the same data, which is the clearest way to show the
distinction: the same 30 inputs, two different kinds of question.

### 3. Why does this project use two tasks instead of one?

They answer different questions and have different practical value. The
regression model gives a magnitude — *how well will this student do?* The
classifier gives an actionable decision — *does this student need help?*

It also exposes something honest: **the two tasks succeed to different degrees**.
The classification task is partly solvable (ROC-AUC 0.686–0.724 against 0.500
for guessing). The regression task essentially is not (cross-validated R² is
**−0.078**, i.e. no better than predicting the average). Predicting a rank or a
category is easier than predicting an exact number.

### 4. What is the difference between a parameter and a hyper-parameter?

A **parameter** is learned from the data during fitting: the coefficients of the
Linear Regression, the support vectors of the SVM, the split thresholds in each
tree.

A **hyper-parameter** is set *before* fitting and controls how learning happens:
`C` in Logistic Regression and the SVM, `gamma` in the RBF kernel,
`n_estimators` and `min_samples_leaf` in the Random Forest. These were chosen by
`GridSearchCV`, which is why selecting them on test data would be cheating —
they must be chosen using training data only.

---

## The four algorithms

### 5. What is Multivariate Linear Regression, and why "multivariate"?

It fits a straight-line relationship between several inputs and one continuous
output:

```
G3 = b0 + b1·age + b2·studytime + b3·failures + ... + b43·romantic_yes
```

**"Multivariate" refers to multiple input variables, not multiple outputs** —
30 student attributes (43 after encoding) predicting the single value `G3`. The
coefficients `b1...b43` are found by ordinary least squares: the values that
minimise the sum of squared differences between predicted and actual grades.

Simple linear regression would use one input. Multiple/multivariate regression
uses many, which is what lets the model weigh study time against past failures
against parental education simultaneously.

### 6. Why use Linear Regression at all, given it performed poorly here?

Three reasons, and the poor performance is one of them.

1. It is the natural **baseline** for a continuous target — the simplest thing
   that could work. Reaching for something complex before trying it would be
   unjustified.
2. Its coefficients are **directly interpretable**, which the project uses.
3. **Its failure is informative.** Test R² of 0.1415 with a *negative*
   cross-validated R² of −0.078 is strong evidence that the relationship between
   student background and exact final grade is not linear, not strong, or not
   present in this data. That is a genuine finding, not a wasted model — and it
   is why the project does not then pile on more regressors to disguise it.

### 7. What is Logistic Regression, and how does it differ from Linear Regression?

Logistic Regression predicts a **probability of class membership**, not a
quantity. It computes the same kind of linear combination, then passes it
through the **sigmoid** function:

```
z     = b0 + b1·x1 + ... + bn·xn
P(at risk) = 1 / (1 + e^(−z))
```

The sigmoid squashes any real number into (0, 1), so the output is a valid
probability. A default threshold of 0.5 converts it to a label.

Key differences: linear regression can output 23.7 or −4 (meaningless as a
probability); it minimises squared error while logistic regression maximises
likelihood (minimises log-loss); and despite the name, logistic regression is a
**classification** algorithm.

### 8. Why is it called "regression" if it classifies?

Historical naming. It *regresses* onto the **log-odds** of the outcome, which is
a continuous quantity:

```
log( P / (1 − P) ) = b0 + b1·x1 + ... + bn·xn
```

So there is a genuine linear regression happening — on the log-odds scale, not
on the class label. The classification comes from thresholding the resulting
probability.

### 9. What is an SVM, and what does it actually optimise?

A Support Vector Machine finds the decision boundary that separates the classes
with the **widest possible margin** — the greatest distance to the nearest
points of either class. Those nearest points are the **support vectors**, and
they alone determine the boundary; moving a far-away point changes nothing.

The intuition for why a wide margin helps: a boundary squeezed up against the
training points is fragile, and a small shift in a new point flips its
prediction. A wide margin is more robust to unseen data.

Because real data is rarely separable, a *soft margin* allows some
misclassification, and **`C`** controls that trade-off: small `C` tolerates more
errors for a wider margin (simpler model); large `C` insists on fewer training
errors at the cost of a narrower, more contorted margin. Grid search selected
**`C = 10.0`** here.

### 10. What is a kernel, and what does the RBF kernel do?

A **kernel** computes similarity between two points *as if* they had been mapped
into a higher-dimensional space, without ever performing that mapping. This is
the **kernel trick**: it gives a non-linear boundary at the cost of a linear one.

The **RBF (Radial Basis Function)** kernel is:

```
K(x, x') = exp( −gamma · ||x − x'||² )
```

It depends only on the *distance* between two points, so similarity decays with
distance. **`gamma`** sets how fast: large gamma means influence drops off
sharply (each point affects only its immediate neighbourhood → wiggly boundary,
overfitting risk); small gamma means far-reaching influence (smoother boundary).
Grid search selected **`gamma = 0.01`**, a relatively smooth boundary — which
makes sense on only 316 training rows.

RBF was chosen as a reasonable default that can capture non-linear structure. In
the event it did not help much: the SVM's F1 of 0.5614 is statistically tied
with the linear Logistic Regression's 0.5333, which suggests whatever structure
is in this data is not meaningfully non-linear.

### 11. Why does SVM need feature scaling when Random Forest does not?

Because the RBF kernel is built on **squared Euclidean distance**, which sums
squared differences across features. Unscaled, `absences` (range 0–75)
contributes differences up to 5625 while `studytime` (range 1–4) contributes at
most 9. The distance — and therefore every prediction — would be dictated by
`absences` purely because of its units, not its relevance.

Logistic Regression is affected too, through the `C` penalty on coefficient
magnitude: a feature on a large scale gets a small coefficient and is penalised
differently from an equivalent feature on a small scale.

**Random Forest is immune** because a tree asks threshold questions
(`absences > 7.5?`). Any monotonic rescaling preserves the ordering of values,
so it preserves every possible split and leaves the tree unchanged.

This project scales for all models anyway, so there is only one preprocessing
definition to audit — and the scaling genuinely does nothing for the forest.

---

## Ensemble learning

### 12. What is ensemble learning?

Combining several models so the group outperforms any individual member. The
necessary condition is that the members make **different** mistakes: if all
models err identically, averaging them gains nothing.

The two main families are **bagging** (train members in parallel on different
data samples, then vote — reduces *variance*) and **boosting** (train members
sequentially, each correcting its predecessor's errors — reduces *bias*).
Random Forest is bagging.

### 13. What is bagging?

**B**ootstrap **AGG**regat**ING**. Two steps:

1. **Bootstrap** — draw many random samples *with replacement* from the training
   set, each the same size as the original. Because sampling is with
   replacement, each sample omits about 37% of the rows and duplicates others.
2. **Aggregate** — train one model per sample and combine them by majority vote
   (classification) or averaging (regression).

The point is that each model sees a slightly different dataset and so learns
slightly different rules. High-variance models like deep decision trees benefit
most, because their individual errors are largely uncorrelated and therefore
cancel on averaging.

### 14. Why is Random Forest an ensemble, and what does it add to plain bagging?

It is an ensemble because the final prediction is a **majority vote across 400
separate decision trees**, each a complete model in its own right.

Random Forest = bagging **plus random feature selection at every split**. Each
tree is grown on its own bootstrap sample *and*, at each node, may only consider
a random subset of features — here `max_features="sqrt"`, about 6 of 43.

**That second ingredient is essential, and this project shows why concretely.**
`failures` is by far the strongest predictor (permutation importance 0.279
against 0.050 for the runner-up). With plain bagging, nearly every tree would
split on `failures` first and the trees would come out near-identical —
correlated trees, so averaging buys almost nothing. Restricting the features
forces most trees to find structure elsewhere, which decorrelates them and makes
the averaging worthwhile.

### 15. How do many trees produce one prediction?

For classification, each tree votes for a class and the majority wins. For a
probability, scikit-learn averages the per-tree class proportions: if 290 of 400
trees say At Risk, P(at risk) ≈ 0.72. That is where the percentage shown in the
app comes from.

### 16. Did the ensemble actually beat the simpler models here?

**No — and saying so is the correct answer.** On the test set:

| Model | F1 | CV F1 (mean ± sd) |
|---|---|---|
| SVM (RBF) | 0.5614 | 0.460 ± 0.038 |
| Random Forest | 0.5581 | 0.457 ± 0.083 |
| Logistic Regression | 0.5333 | 0.492 ± 0.075 |

The gap between first and second is **0.0033**, while the cross-validation
standard deviation is **0.0826** — twenty-five times larger. The three models
are **statistically tied**; the project detects this in code (`assess_tie()`)
and reports a tie rather than a winner.

Random Forest is retained as the primary model because it satisfies the ensemble
requirement and has the best accuracy (0.7595) and precision (0.7059) — *not*
because it won, because it did not.

The lesson: when the signal is weak and the sample is small (316 training rows),
model complexity buys nothing. A Logistic Regression with `C = 0.01` — heavily
regularised, the simplest thing in the project — matches a 400-tree forest.

---

## Evaluation

### 17. What is a train/test split, and why does it matter?

The data is divided so the model is fitted on one part and evaluated on another
it has never seen — here **80/20**, giving 316 training and 79 test students.

Without it, you measure memorisation rather than generalisation. This project's
own numbers make that vivid: Random Forest scores **F1 0.9763 on training data**
and **0.5581 on test data**. Judged on training data it would look excellent. It
is not.

`random_state=42` fixes the split so every result is reproducible.

### 18. What is stratification and why is it used here?

`stratify=y` makes the train and test sets preserve the original class
proportions: **32.9% at risk in both**, matching the full dataset.

Without it, a random 79-student test set could easily land at 25% or 42% at
risk, and the metrics would no longer be comparable to the training distribution
or to other models' metrics. With a minority class of only 130 students, that
variation is large enough to matter.

It is used for the classification split only. The regression target is
continuous, so there are no classes to stratify.

### 19. What is cross-validation, and why use it when you already have a test set?

**K-fold cross-validation** splits the training data into k parts (here 5), then
trains on k−1 and validates on the remaining one, rotating until every part has
served as validation once. The result is a mean score *and* a standard
deviation.

**Why here:** the test set is 79 students, so one student changing class moves F1
by roughly 0.02. A single split cannot tell apart models differing by 0.003. The
CV standard deviation **measures that noise directly**, and is what justifies
the tie verdict in question 16.

It also revealed that the test split is a flattering one: CV F1 (0.46–0.49) sits
*below* test F1 (0.53–0.56) for all three models. Reporting only test figures
would have overstated the results by about 0.07 F1.

**An important caveat:** the hyper-parameters were chosen on these same folds, so
the CV figures are optimistic. The held-out test set — never seen by the grid
search — remains the honest number. This project reports both.

### 20. What is a confusion matrix?

A table of predictions against actual classes. For Random Forest here:

|  | Predicted Not At Risk | Predicted At Risk |
|---|---|---|
| **Actually Not At Risk** | TN = 48 | FP = 5 |
| **Actually At Risk** | FN = 14 | TP = 12 |

- **TN (48)** correctly cleared
- **FP (5)** false alarm — flagged, but actually passed
- **FN (14)** **missed an at-risk student** — the costly error here
- **TP (12)** correctly flagged

It is more informative than accuracy because it shows *which* errors are being
made. Random Forest's accuracy of 0.7595 looks respectable until you see it
missed 14 of 26 at-risk students.

### 21. What are precision, recall and F1, and which matters most here?

With At Risk as the positive class:

- **Precision = TP/(TP+FP)** — of those flagged, how many really were at risk.
  Random Forest: 12/17 = **0.706**. Low precision wastes a tutor's time.
- **Recall = TP/(TP+FN)** — of those genuinely at risk, how many were found.
  Random Forest: 12/26 = **0.462**. Low recall means missing the students the
  system exists to help.
- **F1 = 2·(P·R)/(P+R)** — the harmonic mean, which punishes a model that is
  strong on one and weak on the other.

**Recall matters most for an early-warning system**, because the cost of missing
a struggling student exceeds the cost of an unnecessary conversation. That is
why all three classifiers use `class_weight="balanced"`.

The trade-off is visible in the project: Logistic Regression has recall 0.615
but precision 0.471 (34 alarms, 16 correct); Random Forest has recall 0.462 but
precision 0.706 (17 alarms, 12 correct). Neither is simply better — the choice
depends on how many conversations a school can afford.

### 22. Why can accuracy be misleading? Give the number from this project.

Because with imbalanced classes a useless model can score well. Here **67.1% of
students are Not At Risk**, so a model that predicts "Not At Risk" for every
single student achieves:

```
Accuracy  = 0.6709
Precision = 0.0000
Recall    = 0.0000
F1        = 0.0000
```

**67% accuracy while finding zero at-risk students.** That row appears in every
comparison table in this project for exactly this reason, and it is why the
ranking uses F1.

Note the consequence: Logistic Regression (0.6456) and the SVM (0.6835) score at
or *below* that baseline on accuracy while being far more useful — they were
deliberately trained to trade accuracy for recall.

### 23. What is ROC-AUC and why is it reported?

The ROC curve plots true positive rate against false positive rate at every
possible decision threshold; **AUC** is the area under it. Interpretation: the
probability that a randomly chosen at-risk student is ranked above a randomly
chosen not-at-risk student. 0.5 = random guessing, 1.0 = perfect.

Here: Logistic Regression **0.6981**, SVM **0.7242**, Random Forest **0.6858**.

It is reported because it is the metric **least distorted by the class imbalance
and independent of the arbitrary 50% cut-off**. Precision, recall and F1 all
depend on where the threshold is set; AUC assesses the underlying ranking. All
three models sitting clearly above 0.5 is the cleanest evidence that there is
real signal in the data, even though the F1 scores are modest.

### 24. What are MAE, MSE, RMSE and R²?

For the regression task (test set):

- **MAE = 3.3953** — mean absolute error. On average the prediction is off by
  about 3.4 grade points on a 0–20 scale. Most intuitive.
- **MSE = 17.6037** — mean squared error. Squaring punishes large errors more,
  but the units (grade points squared) are not interpretable.
- **RMSE = 4.1957** — the square root of MSE, back in grade points, still
  weighting large errors more heavily than MAE does. **RMSE > MAE always**, and
  the gap here (4.20 vs 3.40) reflects a few very large errors — mostly the 38
  students who recorded a grade of 0.
- **R² = 0.1415** — the fraction of variance in `G3` explained by the model.
  1.0 is perfect, 0 means no better than always predicting the mean, and
  negative means *worse* than that.

### 25. Your R² is 0.14. Is the model any good?

No, and the test figure is the flattering one.

On the test split R² = 0.1415 and the model does beat the mean predictor's RMSE
(4.1957 against 4.5502). But **5-fold cross-validation gives R² = −0.078 ±
0.140** — a *negative* mean, meaning on some folds the model is worse than
always answering "10.4". The positive test R² is largely the luck of one
79-student split.

The honest conclusion: predicting an exact final grade from background
information alone, with no prior marks, does not work on this dataset. The model
demonstrates the technique correctly; it is not a usable grade predictor. Saying
that is more defensible than quoting 0.14 as if it were a success.

---

## Data leakage and preprocessing

### 26. What is data leakage? Give the example from this project.

Leakage is when information that would not be available at prediction time
reaches the model during training, producing scores that cannot be reproduced in
practice.

**The example here is `G1` and `G2`** — the first- and second-period grades,
which correlate **+0.80** and **+0.90** with the final grade `G3`. They are not
background information about a student; they are earlier measurements of the very
outcome being predicted. Both are excluded.

The project measures the effect rather than asserting it. Refitting the same
models *with* `G1` and `G2`:

| Metric | Leakage-free | With leakage |
|---|---|---|
| Random Forest F1 | **0.5581** | 0.8679 |
| Random Forest ROC-AUC | **0.6858** | 0.9666 |
| Linear Regression R² | **0.1415** | 0.7241 |

Random Forest F1 rises from 0.558 to **0.868**. Many published write-ups of this
dataset report accuracies above 90% precisely because they include `G2`.

There are two reasons to refuse it: it is leakage, *and* it destroys the use
case. An early-warning system must raise the alarm while there is still time to
act. By the time the second-period grade exists, the warning is worthless.

### 27. Other than excluding G1/G2, how else does this project prevent leakage?

Three mechanisms:

1. **Preprocessing lives inside the Pipeline.** When `pipeline.fit(X_train)`
   runs, the scaler's mean and standard deviation and the encoder's category
   list are computed from **training rows only**. Fitting a scaler on the full
   dataset *before* splitting is the most common leakage bug in student
   projects, and the pipeline structure makes it impossible.
2. **An assertion enforces the exclusion.** `split_features_target()` raises an
   `AssertionError` if `G3`, `at_risk`, `G1` or `G2` reaches the feature matrix,
   so the run fails loudly rather than producing a quietly inflated score.
3. **Hyper-parameters are selected on training folds only.** `GridSearchCV` never
   sees the test set, so the reported test metrics are not contaminated by model
   selection.

A fourth, more subtle one: the Mathematics and Portuguese files are **not**
combined, because 382 students appear in both and stacking them would place the
same student in train and test.

### 28. Why one-hot encoding, and what does `drop="if_binary"` do?

Categorical values like `Mjob = "teacher"` are not numbers, and encoding them as
1, 2, 3 would invent an ordering ("teacher > health") that does not exist and
that distance-based models would act on. One-hot encoding creates one binary
column per level instead, with no implied order.

**`drop="if_binary"`** emits a single column for two-level features: `internet`
becomes `internet_yes` (1 or 0) rather than two perfectly collinear columns
`internet_yes` and `internet_no`. That avoids redundancy and keeps the Logistic
Regression coefficients readable. Multi-level features like `Mjob` keep all five
columns.

**`handle_unknown="ignore"`** matters for the web app: a category never seen in
training is encoded as all-zeros rather than raising an exception. The
application detects this and warns the user instead of crashing.

### 29. Why are the ordinal features treated as numeric rather than one-hot encoded?

Features like `studytime` (1 = under 2 hours ... 4 = over 10 hours) are **ordered**
scales. Treating them as numeric preserves that ordering, keeps the feature count
down (13 columns instead of ~45), and keeps coefficients interpretable as "per
one-step increase".

The cost, which should be acknowledged: it assumes the steps are **evenly
spaced** — that going from "under 2 hours" to "2–5 hours" is the same size of
change as "5–10" to "over 10". That is a simplification, not a free choice.
One-hot encoding them would drop the assumption but spend many more parameters
on 316 training rows.

---

## Interpretation and limitations

### 30. What is feature importance, why is it useful, and what is the trap?

Feature importance ranks inputs by how much the model relies on them. It is
useful for three things: understanding what the model learned, sanity-checking it
against domain knowledge, and communicating results to non-specialists.

This project computes **two** measures, and **they disagree** — which is the more
interesting answer:

| Rank | Impurity importance | Permutation importance |
|---|---|---|
| 1 | `absences` (0.0932) | **`failures` (0.2786)** |
| 2 | `failures` (0.0902) | `absences` (0.0502) |

**Impurity importance** (`feature_importances_`) sums how much each feature
reduced Gini impurity across all splits. It has a known bias: it **inflates
continuous and high-cardinality features**, because they offer more candidate
split points. `absences` takes dozens of distinct values; `studytime` takes four.

**Permutation importance** shuffles one column of the **held-out** data and
measures the actual drop in test F1. It answers the question directly and is not
inflated by cardinality. **Where they disagree, trust permutation importance** —
and here it puts `failures` 5.5× above the runner-up.

**The trap is causation.** Importance describes which columns *this* forest,
fitted to 316 students from two Portuguese schools in 2008, happened to rely on.
It is **not** evidence that reducing a student's absences would raise their
grade. The correct phrasing is "these features were influential in the model's
predictions", never "absences cause poor performance". Several features are
plausibly proxies for things the dataset never measures: household stability,
work outside school, health, prior schooling quality.

### 31. What is overfitting, and show it in your results.

Overfitting is when a model learns noise specific to the training data rather
than generalisable patterns — excellent training performance, poor test
performance.

This project's own train-to-test gaps:

| Model | train F1 | test F1 | gap |
|---|---|---|---|
| Logistic Regression | 0.5498 | 0.5333 | **+0.0164** |
| SVM (RBF) | 0.7719 | 0.5614 | **+0.2105** |
| Random Forest | 0.9763 | 0.5581 | **+0.4182** |

Random Forest is overfitting substantially — F1 of 0.98 on data it was trained
on, 0.56 on data it was not — **even with** 400 trees, `max_features="sqrt"` and
`min_samples_leaf=3`. The honest statement is that the ensemble *reduced*
overfitting, not that it eliminated it.

Logistic Regression's gap of +0.016 is strikingly small, which is what the grid
search's choice of `C = 0.01` (strong regularisation) buys: a simple model that
cannot memorise much. Counter-measures used here: train/test splitting,
cross-validation, `min_samples_leaf`, feature subsampling, and regularisation.

### 32. Why did the heavily-regularised Logistic Regression (`C = 0.01`) get *worse* on the test set after tuning?

An honest wrinkle worth being ready for. Before tuning, Logistic Regression
scored F1 0.5806; after the grid search selected `C = 0.01` it scored 0.5333.

The grid search picked the `C` that maximised F1 **across the cross-validation
folds**, and that choice did not transfer to this particular test split. With
316 training rows and 5 folds of ~63 students each, the fold-level estimates are
noisy enough that the selected hyper-parameter is partly fitted to fold noise.

It is the right *procedure* — selecting on test data would be leakage — applied
to a dataset too small for the procedure to be reliable. It is also further
evidence for the tie verdict: if re-tuning moves a model by 0.05 F1, a 0.003 gap
between models means nothing.

### 33. What are the main limitations of this project?

- **Small, old data.** 395 students, two Portuguese schools, 2008. The 79-student
  test set means one reclassified student moves F1 by ~0.02.
- **The models are weak.** F1 ≈ 0.56 means roughly half of flagged students are
  false alarms. Cross-validated regression R² is negative.
- **38 students have `G3` = 0** while recording zero absences *and* a non-zero
  `G1` — almost certainly dropout or an unrecorded mark rather than a real score.
  Keeping them is defensible but it distorts both tasks, and it suppresses the
  `absences` correlation in the aggregate.
- **The three classifiers are statistically indistinguishable**, so no claim that
  one algorithm is better is supportable.
- **Self-reported features** (study time, alcohol use, family relationships) carry
  questionnaire inaccuracy.
- **Association, never causation.**
- **The 50% threshold is an arbitrary default**, not derived from the relative
  cost of a missed student versus a false alarm.
- **Ordinal features are assumed evenly spaced.**
- **CV scores are optimistic**, since the same folds selected the
  hyper-parameters.
- **No fairness audit**, despite the model using sensitive attributes (sex,
  family structure, parental education).

### 34. If you had to improve the results, what would you do first?

**Not a different algorithm.** The three models are already tied, which says the
bottleneck is the data, not the learner.

In order:

1. **Tune the decision threshold** against an explicit cost ratio for missed
   students versus false alarms. Given the precision/recall spread (Random
   Forest 0.71/0.46, Logistic Regression 0.47/0.62), this would improve
   practical usefulness more than any model change — and it costs nothing.
2. **Get better features**: engagement over time — weekly submissions, LMS
   logins, assignment timeliness — rather than a single year-end absence count.
   The absence finding makes the case: the raw annual total correlates +0.034
   with the grade, but that is an artefact — all 38 zero-grade students record
   zero absences, and excluding them gives −0.213. An annual total is fragile to
   exactly this kind of record-keeping distortion; a within-year trend would be
   much harder to corrupt.
3. **More data**, from more institutions, to test generalisation at all.

### 35. Can this system be used in a real school?

No, and the project says so explicitly. Three reasons.

**It is not accurate enough.** At the reported precision, roughly one flagged
student in two is a false alarm, and it misses a substantial share of at-risk
students.

**It establishes no causation.** It cannot tell a school what to *change*, only
where patterns resembling past failure appear.

**It carries ethical risk.** A false "At Risk" label can stigmatise a student and
become self-fulfilling; a false "Not At Risk" label withholds help. Several
inputs — sex, family size, parental cohabitation and education — are sensitive
attributes, so a model allocating attention using them risks encoding existing
disadvantage as a prediction about an individual. A real deployment would need a
fairness audit this project does not perform.

The legitimate framing: a screening aid that helps decide **who a tutor talks to
first**, never a replacement for that conversation.

---

## The five questions most likely to catch you out

1. **"Random Forest is your main model — did it win?"**
   No. SVM had the highest F1 (0.5614 vs 0.5581), but the 0.0033 gap is 25×
   smaller than the 0.0826 CV standard deviation, so all three are statistically
   tied. Random Forest is primary because it is the required ensemble and has the
   best accuracy and precision — not because it won.

2. **"Your scores are low. Why?"**
   Because the scores that would look good require leakage. Including `G1`/`G2`
   lifts Random Forest F1 from 0.558 to 0.868 — I measured it, documented it, and
   refused it, because a model needing the second-period grade to predict the
   third has not solved the early-warning problem.

3. **"Your R² is 0.14 — is the regression model any good?"**
   No. And the cross-validated R² is **−0.078**, i.e. worse than predicting the
   average. The positive test R² is the luck of one 79-student split. Predicting
   an exact grade from background data alone does not work on this dataset.

4. **"Which feature matters most?"**
   `failures`, by permutation importance (0.279, 5.5× the runner-up). Note that
   the built-in impurity importance says `absences` instead — that is the
   well-documented high-cardinality bias, and the permutation measure on held-out
   data is the one to trust.

5. **"Your SVM's accuracy (0.6835) is barely above the 0.6709 baseline. Isn't it useless?"**
   On accuracy alone it looks that way, but the baseline has **recall 0 and F1 0** —
   it finds no at-risk students at all. The SVM finds 62% of them. All three
   classifiers use `class_weight="balanced"`, which deliberately trades accuracy
   for recall, because a missed at-risk student costs more than an unnecessary
   conversation.

---

## Explain this project in 60 seconds

> I built a system that predicts whether a secondary-school student is at risk of
> failing, using the UCI Student Performance dataset — 395 real students with 33
> attributes each.
>
> There are two tasks. **Multivariate Linear Regression** predicts the final grade
> on a 0–20 scale. Then **Logistic Regression, an RBF-kernel SVM, and a Random
> Forest ensemble** classify each student as At Risk or Not At Risk, where At Risk
> means a final grade below the pass mark of 10.
>
> The key design decision was **excluding the intermediate grades G1 and G2**. They
> correlate 0.80 and 0.90 with the final grade, so using them is target leakage —
> and useless for early warning, because by then it is too late to help. I measured
> what that cost: including them would have lifted Random Forest F1 from 0.56 to
> 0.87. I refused it and documented the trade-off.
>
> The honest results: all three classifiers reach ROC-AUC between 0.69 and 0.72
> against 0.50 for guessing, so there is real signal — but at F1 around 0.56 it is a
> screening aid, not a decision tool. The three models are **statistically tied**:
> the gap between best and second is 0.003 against a cross-validation spread of
> 0.083. The regression model is worse still — cross-validated R² is negative, no
> better than predicting the average.
>
> The strongest usable predictor is **prior failure**. And the most interesting
> technical finding is that the two feature-importance measures disagree: the
> built-in impurity importance ranks absences first, but permutation importance on
> held-out data ranks past failures first by 5.5×, which is the textbook
> high-cardinality bias showing up in my own results.
>
> Everything is in a Streamlit app with seven sections, including one dedicated to
> demonstrating the leakage I avoided.
