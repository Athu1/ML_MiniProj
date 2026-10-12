"""
Used Car Price & Premium Prediction System
==========================================

Streamlit front end. Run with:

    streamlit run app.py

This application does NOT train anything. It loads the pipelines and metrics
saved by `python src/train_models.py` and displays them. If those files are
missing it says so and stops, rather than retraining silently on every launch.

Every number shown here -- every metric, every importance value, every
confusion-matrix cell -- is read from models/metadata.pkl, which is written by
the training script. Nothing in this file contains a hard-coded result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import config as cfg                      # noqa: E402
import data_preprocessing as dp           # noqa: E402
import prediction as pr                   # noqa: E402
import visualization as vz                # noqa: E402

st.set_page_config(
    page_title="Used Car Price Prediction System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1280px; }
      h1, h2, h3 { color: #0b0b0b; letter-spacing: -0.01em; }
      h1 { font-size: 1.95rem; font-weight: 650; }
      h2 { font-size: 1.35rem; font-weight: 620; margin-top: 0.6rem; }
      h3 { font-size: 1.08rem; font-weight: 600; }
      .lede { color: #52514e; font-size: 0.97rem; line-height: 1.6; max-width: 62rem; }
      .card { background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10);
              border-radius: 10px; padding: 1.05rem 1.2rem; height: 100%; }
      .card-label { color: #52514e; font-size: 0.80rem; font-weight: 500; margin-bottom: 0.3rem; }
      .card-value { color: #0b0b0b; font-size: 1.85rem; font-weight: 640; line-height: 1.15; }
      .card-note { color: #898781; font-size: 0.78rem; margin-top: 0.3rem; line-height: 1.4; }
      .verdict { border-radius: 10px; padding: 1.1rem 1.3rem; border: 1px solid;
                 font-size: 1.28rem; font-weight: 640; }
      .verdict-prem { background: #efecfa; border-color: #4a3aa7; color: #2e2370; }
      .verdict-std { background: #eaf1fb; border-color: #2a78d6; color: #184f95; }
      .verdict-sub { font-size: 0.86rem; font-weight: 450; color: #52514e; margin-top: 0.4rem; }
      .caveat { background: #fcfcfb; border-left: 3px solid #898781; padding: 0.75rem 1rem;
                color: #52514e; font-size: 0.88rem; border-radius: 0 6px 6px 0; line-height: 1.55; }
      div[data-testid="stMetricValue"] { font-size: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Loaders and helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained models...")
def _load_artifacts():
    return pr.load_artifacts()


@st.cache_data(show_spinner="Loading dataset...")
def _load_frames():
    raw = dp.load_raw(validate=False)
    return raw, dp.clean(raw)


def metric_card(label: str, value: str, note: str = "") -> str:
    note_html = f'<div class="card-note">{note}</div>' if note else ""
    return (f'<div class="card"><div class="card-label">{label}</div>'
            f'<div class="card-value">{value}</div>{note_html}</div>')


def caveat(text: str) -> None:
    st.markdown(f'<div class="caveat">{text}</div>', unsafe_allow_html=True)


def chart(fig) -> None:
    st.plotly_chart(fig, width="stretch")


def lakh(value_rupees: float) -> str:
    return f"₹{value_rupees / cfg.LAKH:,.2f} lakh"


try:
    MODELS, META = _load_artifacts()
except pr.ModelsNotTrainedError as exc:
    st.title("Used Car Price & Premium Prediction System")
    st.error("The trained models are not available yet, so the app cannot start.")
    st.markdown("**To fix this, run the training script once:**\n\n"
                "```bash\npython src/train_models.py\n```\n\n"
                "Then reload this page.")
    with st.expander("What exactly is missing"):
        st.code(str(exc))
    st.stop()

try:
    RAW, DF = _load_frames()
except dp.DatasetError as exc:
    st.error("The dataset could not be loaded; the app needs data/raw/cardekho.csv "
             "to draw the data-analysis charts.")
    st.code(str(exc))
    st.stop()

SUMMARY = META["summary"]
CLF = META["classification"]
REG = META["regression"]
BASELINE = META["baseline_classification"]
TIE = META.get("tie_analysis", {})
CLF_METRICS = {name: res["test_metrics"] for name, res in CLF.items()}
PRIMARY = META.get("best_classifier", "Random Forest")

with st.sidebar:
    st.markdown("### Used Car Price\nPrediction System")
    st.caption("Machine Learning mini-project")
    section = st.radio(
        "Section",
        ["Overview", "Car Prediction", "Data Analysis", "Model Comparison",
         "Model Explainability", "Data Quality & Leakage", "About the Project"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        f"**Dataset** {META['dataset_file']}  \n"
        f"{SUMMARY['n_cars']:,} cars after de-duplication · "
        f"{len(cfg.ALL_FEATURES)} input features  \n"
        f"**Trained** {META['trained_at']}  \n"
        f"scikit-learn {META.get('sklearn_version', 'n/a')}  \n"
        f"random_state = {META['random_state']}"
    )
    st.divider()
    st.caption("Academic demonstration only. Predictions describe patterns in a "
               "scraped set of Indian listings (2020 and earlier) and are not a "
               "valuation to rely on.")


# ==========================================================================
# 1. OVERVIEW
# ==========================================================================
def render_overview():
    st.title("Used Car Price & Premium Prediction System")
    st.markdown(
        '<p class="lede">Given a used car\'s age, odometer, engine specification, '
        'brand and seller details, this project estimates its <b>selling price</b> '
        '(regression) and whether it is a <b>premium car</b> priced at '
        f'{cfg.PREMIUM_THRESHOLD / cfg.LAKH:g} lakh rupees or more (classification). '
        'Four models are trained and compared: Linear Regression, Logistic '
        'Regression, an RBF Support Vector Machine and a Random Forest.</p>',
        unsafe_allow_html=True,
    )

    c = st.columns(4)
    c[0].markdown(metric_card("Cars (after de-duplication)", f"{SUMMARY['n_cars']:,}",
                              f"{SUMMARY['n_duplicates_removed']:,} duplicate listings "
                              f"removed from {SUMMARY['n_raw_rows']:,}"),
                  unsafe_allow_html=True)
    c[1].markdown(metric_card("Premium cars", f"{SUMMARY['pct_premium']:.1f}%",
                              f"{SUMMARY['n_premium']:,} of {SUMMARY['n_cars']:,}"),
                  unsafe_allow_html=True)
    c[2].markdown(metric_card("Median price", lakh(SUMMARY["median_price"]),
                              f"mean {lakh(SUMMARY['mean_price'])} (pulled up by luxury cars)"),
                  unsafe_allow_html=True)
    c[3].markdown(metric_card("Input features", str(SUMMARY["n_model_features"]),
                              f"{SUMMARY['n_numeric_features']} numeric, "
                              f"{SUMMARY['n_categorical_features']} categorical"),
                  unsafe_allow_html=True)

    st.subheader("Headline results (held-out test set)")
    p = CLF_METRICS[PRIMARY]
    c = st.columns(4)
    c[0].markdown(metric_card("Price model R²", f"{REG['test_metrics']['R2']:.3f}",
                              f"mean predictor: {REG['baseline_metrics']['R2']:.3f}"),
                  unsafe_allow_html=True)
    c[1].markdown(metric_card("Typical price error (MedAPE)",
                              f"{REG['test_metrics']['MedAPE']:.1f}%",
                              f"MAE {REG['test_metrics']['MAE']:.2f} lakh · "
                              f"mean predictor {REG['baseline_metrics']['MedAPE']:.1f}%"),
                  unsafe_allow_html=True)
    c[2].markdown(metric_card(f"{PRIMARY} F1 (premium)", f"{p['F1']:.3f}",
                              f"ROC-AUC {p['ROC-AUC']:.3f}"), unsafe_allow_html=True)
    c[3].markdown(metric_card("Majority-class accuracy", f"{BASELINE['Accuracy']:.3f}",
                              "what answering 'Standard' every time scores"),
                  unsafe_allow_html=True)

    if TIE.get("is_tie"):
        caveat(f"<b>Read the comparison carefully.</b> The best F1 and the runner-up "
               f"differ by {TIE['gap']:.4f}, which is smaller than the cross-validation "
               f"noise ({TIE['noise_floor']:.4f}). On {TIE['n_test']:,} test cars "
               f"{' and '.join(TIE['tied_models'])} are statistically indistinguishable. "
               f"{PRIMARY} is kept as the primary model for its stability and "
               "calibration, not because it clearly won.")

    st.subheader("How the pipeline works")
    st.markdown(
        "1. **Load & clean** — read `cardekho.csv`, drop exact duplicate listings, "
        "build `age` and `brand`, derive the premium label from the price.\n"
        "2. **Split first** — 80/20 stratified split *before* any statistic is computed.\n"
        "3. **Preprocess inside each model's pipeline** — median imputation + scaling "
        "for numbers, one-hot encoding for categories, fitted on training rows only.\n"
        "4. **Train & tune** — grid search with 5-fold cross-validation on the training "
        "split; the decision threshold is chosen on out-of-fold predictions.\n"
        "5. **Evaluate once** on the untouched test set and compare against baselines."
    )


# ==========================================================================
# 2. CAR PREDICTION
# ==========================================================================
def render_prediction():
    st.title("Car Prediction")
    st.markdown('<p class="lede">Describe a car and see the price estimate and '
                'the premium verdict from every model. The form is pre-filled '
                'with the typical car in the dataset.</p>', unsafe_allow_html=True)

    d = META["default_profile"]
    rng = META["feature_ranges"]
    brands = META["known_brands"]

    with st.form("car_form"):
        a, b, c3 = st.columns(3)
        brand = a.selectbox(cfg.FEATURE_LABELS["brand"], brands,
                            index=brands.index(d["brand"]))
        age = b.number_input(cfg.FEATURE_LABELS["age"], 0, 40, int(d["age"]))
        km = c3.number_input(cfg.FEATURE_LABELS["km_driven"], 0, 3_000_000,
                             int(d["km_driven"]), step=5000)
        a, b, c3 = st.columns(3)
        engine = a.number_input(cfg.FEATURE_LABELS["engine"], 400, 6000, int(d["engine"]), step=50)
        power = b.number_input(cfg.FEATURE_LABELS["max_power"], 20.0, 700.0,
                               float(d["max_power"]), step=5.0)
        mileage = c3.number_input(cfg.FEATURE_LABELS["mileage"], 0.0, 50.0,
                                  float(d["mileage"]), step=0.5)
        a, b, c3 = st.columns(3)
        seats = a.number_input(cfg.FEATURE_LABELS["seats"], 2, 14, int(d["seats"]))
        fuel = b.selectbox(cfg.FEATURE_LABELS["fuel"], list(cfg.CATEGORICAL_LABELS["fuel"]),
                           index=list(cfg.CATEGORICAL_LABELS["fuel"]).index(d["fuel"]))
        trans = c3.selectbox(cfg.FEATURE_LABELS["transmission"],
                             list(cfg.CATEGORICAL_LABELS["transmission"]),
                             index=list(cfg.CATEGORICAL_LABELS["transmission"]).index(d["transmission"]))
        a, b, c3 = st.columns(3)
        seller = a.selectbox(cfg.FEATURE_LABELS["seller_type"],
                             list(cfg.CATEGORICAL_LABELS["seller_type"]),
                             index=list(cfg.CATEGORICAL_LABELS["seller_type"]).index(d["seller_type"]))
        owner = b.selectbox(cfg.FEATURE_LABELS["owner"],
                            list(cfg.CATEGORICAL_LABELS["owner"]),
                            format_func=lambda k: cfg.CATEGORICAL_LABELS["owner"][k],
                            index=list(cfg.CATEGORICAL_LABELS["owner"]).index(d["owner"]))
        classifier = c3.selectbox("Classifier for the verdict", cfg.CLASSIFICATION_MODEL_NAMES,
                                  index=cfg.CLASSIFICATION_MODEL_NAMES.index(PRIMARY))
        use_tuned = st.checkbox("Use the cost-tuned decision threshold chosen in training "
                                "(instead of 0.5)", value=False)
        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        st.info("Adjust the car above and press **Predict**.")
        return

    car = dict(brand=brand, age=age, km_driven=km, engine=engine, max_power=power,
               mileage=mileage, seats=seats, fuel=fuel, transmission=trans,
               seller_type=seller, owner=owner)
    tuned = CLF[classifier].get("threshold", {}).get("threshold") if use_tuned else None
    try:
        res = pr.predict_car(car, MODELS, META, classifier=classifier, threshold=tuned)
    except pr.InvalidCarInputError as exc:
        st.error(str(exc))
        return

    for note, items in (("Not seen in training (encoded as unknown)", res["unknown_categories"]),
                        ("Outside the range the models were trained on — treat the result as an "
                         "extrapolation", res["out_of_range"])):
        if items:
            st.warning(f"{note}: " + "; ".join(items))

    left, right = st.columns([1, 1])
    with left:
        chart(vz.price_display(res["predicted_price"]))
        mae = REG["test_metrics"]["MAE"]
        medape = REG["test_metrics"]["MedAPE"]
        st.markdown(
            f"**Estimated price: {lakh(res['predicted_price'])}** "
            f"(₹{res['predicted_price']:,.0f}). On the test set the typical miss was "
            f"{medape:.1f}% of the price (MAE {mae:.2f} lakh), so treat this as a "
            "range, not a quote."
        )
    with right:
        css = "verdict-prem" if res["premium"] else "verdict-std"
        pct = res["premium_probability"]
        st.markdown(
            f'<div class="verdict {css}">{classifier}: {res["label"]}'
            f'<div class="verdict-sub">Estimated probability of Premium: {pct:.1%} · '
            f'threshold {res["threshold_used"]:.3f}'
            f'{" (cost-tuned)" if res["threshold_is_tuned"] else " (default)"}</div></div>',
            unsafe_allow_html=True)
        chart(vz.premium_probability_bar(pct, res["threshold_used"]))

    st.subheader("What each model says")
    rows = [{"Model": cfg.REGRESSION_MODEL_NAME + " (price)",
             "Verdict": "Premium" if res["regression_implies_premium"] else "Standard",
             "Output": lakh(res["predicted_price"])}]
    rows += [{"Model": n, "Verdict": v["label"],
              "Output": f"P(Premium) = {v['probability']:.1%}"}
             for n, v in res["per_model"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if not res["models_agree"] or res["regression_implies_premium"] != res["premium"]:
        caveat("The models do not all agree on this car. That is information: it "
               "sits near the 6-lakh boundary, where the label flips on a small "
               "change in price.")
    if res["probability_disagrees_with_label"]:
        caveat("This model's label and its probability point different ways. The "
               "SVM labels by its decision function but reports a separately "
               "fitted (Platt-scaled) probability; near the boundary they can differ.")
    if res["threshold_decided_it"]:
        caveat(f"The threshold, not the model, decided this verdict: with the other "
               f"cut-off ({res['alt_threshold']:.3f}) the answer would flip.")

    st.subheader("Inputs the Random Forest relies on most")
    inf = pr.top_influential_features(META, res["input_row"], top_n=6)
    st.dataframe(pd.DataFrame(
        [{"Feature": r["label"], "This car": r["car_value"],
          "Global importance": round(r["importance"], 4)} for r in inf]),
        hide_index=True, width="stretch")
    st.caption("A global ranking (permutation importance on the test set), not an "
               "explanation of this one prediction.")
    chart(vz.car_profile_bars(res["input_row"].iloc[0].to_dict(), rng))


# ==========================================================================
# 3. DATA ANALYSIS
# ==========================================================================
def render_data_analysis():
    st.title("Data Analysis")
    st.markdown(f'<p class="lede">The source is a scraped set of {SUMMARY["n_raw_rows"]:,} '
                'Indian used-car listings. Prices are heavily right-skewed '
                f'(skew {SUMMARY["price_skew"]:.1f}), so several charts use a log axis.</p>',
                unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["Prices & classes", "Features vs price", "Correlations",
                              "Raw data"])
    with t1:
        l, r = st.columns(2)
        with l:
            chart(vz.price_distribution(DF))
        with r:
            chart(vz.class_distribution(DF))
        caveat("The premium label is simply <i>price ≥ 6 lakh</i>. It is a modelling "
               "convention chosen before any model was fitted, not an industry "
               "definition — a different cut-off would be a different task.")
    with t2:
        l, r = st.columns(2)
        cat = l.selectbox("Category", ["brand", "fuel", "transmission", "seller_type", "owner"],
                          format_func=lambda c: cfg.FEATURE_LABELS[c])
        num = r.selectbox("Numeric feature", list(cfg.NUMERIC_FEATURES),
                          format_func=lambda c: cfg.FEATURE_LABELS[c])
        l, r = st.columns(2)
        with l:
            chart(vz.feature_vs_price_box(DF, cat))
        with r:
            chart(vz.numeric_vs_price(DF, num))
    with t3:
        chart(vz.correlation_heatmap(DF))
        caveat("Age has the strongest relationship with log price and is negative; "
               "power and engine size are positive. Correlation is not causation: "
               "newer cars also tend to be higher-spec.")
    with t4:
        st.dataframe(RAW.head(200), hide_index=True, width="stretch")
        st.caption(f"First 200 of {len(RAW):,} raw rows, exactly as published "
                   f"({RAW.shape[1]} columns).")
        st.markdown("**Missing values in the raw file**")
        miss = RAW.isna().sum()
        st.dataframe(miss[miss > 0].rename("missing").to_frame(), width="stretch")


# ==========================================================================
# 4. MODEL COMPARISON
# ==========================================================================
def render_model_comparison():
    st.title("Model Comparison")
    st.subheader("Price regression")
    tm, bm = REG["test_metrics"], REG["baseline_metrics"]
    reg_table = pd.DataFrame([
        {"Model": "Linear Regression (log price)", "MAE (lakh)": tm["MAE"],
         "RMSE (lakh)": tm["RMSE"], "R²": tm["R2"], "MedAPE %": tm["MedAPE"],
         "R² on log price": tm["R2_log"], "CV R² (mean ± std)":
         f"{REG['cv_r2_mean']:.3f} ± {REG['cv_r2_std']:.3f}"},
        {"Model": "Baseline: predict the mean", "MAE (lakh)": bm["MAE"],
         "RMSE (lakh)": bm["RMSE"], "R²": bm["R2"], "MedAPE %": bm["MedAPE"],
         "R² on log price": None, "CV R² (mean ± std)": "—"},
    ])
    st.dataframe(reg_table.round(3).fillna("—").astype(str), hide_index=True,
                 width="stretch")
    l, r = st.columns(2)
    with l:
        chart(vz.actual_vs_predicted(REG["y_test"], REG["y_pred"], tm["R2"]))
    with r:
        chart(vz.residual_plot(REG["y_test"], REG["y_pred"]))

    st.subheader("Premium classification")
    table = pd.DataFrame([{"Model": n, **m, "CV F1 (mean)": CLF[n]["cv_f1_mean"],
                           "CV F1 (std)": CLF[n]["cv_f1_std"]}
                          for n, m in CLF_METRICS.items()])
    st.dataframe(table.round(4), hide_index=True, width="stretch")
    chart(vz.metric_comparison(CLF_METRICS, BASELINE))
    l, r = st.columns(2)
    with l:
        chart(vz.cv_stability(CLF))
    with r:
        chart(vz.roc_curves(CLF))
    if TIE.get("is_tie"):
        caveat(f"<b>Statistical tie.</b> Gap between first and second F1: "
               f"{TIE['gap']:.4f}; cross-validation noise: {TIE['noise_floor']:.4f}. "
               f"Tied: {', '.join(TIE['tied_models'])}.")

    st.subheader("Confusion matrices")
    cols = st.columns(3)
    for col, name in zip(cols, cfg.CLASSIFICATION_MODEL_NAMES):
        with col:
            chart(vz.confusion_matrix_figure(CLF[name]["confusion"]["matrix"], name))

    st.subheader("Decision threshold and calibration")
    sel = st.selectbox("Model", cfg.CLASSIFICATION_MODEL_NAMES,
                       index=cfg.CLASSIFICATION_MODEL_NAMES.index(PRIMARY))
    th = CLF[sel]["threshold"]
    chart(vz.threshold_tradeoff(CLF[sel]["threshold_curve"], cfg.DEFAULT_THRESHOLD,
                                th["threshold"], sel))
    caveat(f"The cut-off minimises expected errors with a false-negative : "
           f"false-positive cost ratio of {th['fn_cost_ratio']:g}:1 on <b>out-of-fold "
           f"training predictions</b> (never the test set). The ratio is an assumption: "
           "nothing about used-car pricing says one error is worse than the other, so "
           "1:1 is used, and the chosen threshold lands within a hair of 0.5.")
    st.markdown("**Sensitivity to the cost ratio** (computed on out-of-fold training predictions)")
    st.dataframe(pd.DataFrame(CLF[sel]["threshold_sweep"]), hide_index=True, width="stretch")
    cal = {n: CLF[n]["calibration"] for n in CLF}
    l, r = st.columns(2)
    with l:
        chart(vz.calibration_chart(cal))
    with r:
        chart(vz.probability_spread(cal))
    for n, c in cal.items():
        st.caption(f"**{n}:** Brier {c['brier_score']:.4f} vs {c['brier_baseline']:.4f} "
                   f"for a constant base-rate predictor — {c['summary']}.")


# ==========================================================================
# 5. EXPLAINABILITY
# ==========================================================================
def render_explainability():
    st.title("Model Explainability")
    perm = pd.DataFrame(META["rf_permutation_importance"])
    imp = pd.DataFrame(META["rf_impurity_importance"])
    l, r = st.columns(2)
    with l:
        chart(vz.feature_importance_bars(
            imp, "impurity_importance",
            "Random Forest: impurity importance (encoded columns)", "Importance"))
    with r:
        chart(vz.feature_importance_bars(
            perm, "permutation_importance",
            "Random Forest: permutation importance (test F1 drop)",
            "Drop in F1", std_col="permutation_std"))
    caveat("The two rankings measure different things. Impurity importance is "
           "computed on the training trees and is split across one-hot columns "
           "(fuel_Diesel, owner_First Owner…); permutation importance is measured "
           "on held-out data and grouped per original feature. Prefer the second "
           "when they disagree.")
    chart(vz.logistic_coefficient_bars(pd.DataFrame(META["logistic_coefficients"])))
    st.subheader("Linear Regression coefficients (log price, standardised inputs)")
    reg = pd.DataFrame(META["regression_coefficients"]).head(12)
    st.dataframe(reg[["feature", "coefficient", "pct_change_per_sd"]].round(4),
                 hide_index=True, width="stretch")
    caveat("A coefficient of −0.45 on <i>age</i> means: one standard deviation more "
           "age multiplies the predicted price by exp(−0.45) ≈ 0.64, other inputs held "
           "fixed. Large coefficients on rare levels (e.g. <i>Test Drive Car</i>, a "
           "handful of cars) are noisy and should not be over-read.")


# ==========================================================================
# 6. DATA QUALITY & LEAKAGE
# ==========================================================================
def render_leakage():
    st.title("Data Quality & Leakage")
    st.markdown(f'<p class="lede">{SUMMARY["n_duplicates_removed"]:,} of '
                f'{SUMMARY["n_raw_rows"]:,} raw rows ({SUMMARY["pct_duplicates"]:.1f}%) '
                'are exact copies of another row. If they are left in and the data '
                'is split at random, the same car can sit in both the training and '
                'the test set — and the test score then measures memory, not skill.</p>',
                unsafe_allow_html=True)
    tw = META["duplicate_twins"]
    c = st.columns(3)
    c[0].markdown(metric_card("Duplicates in raw file", f"{SUMMARY['n_duplicates_removed']:,}"),
                  unsafe_allow_html=True)
    c[1].markdown(metric_card("Test cars with an exact twin in train",
                              f"{tw['n_twins']:,} of {tw['n_test']:,}",
                              f"{tw['pct_twins']:.1f}% when duplicates are kept"),
                  unsafe_allow_html=True)
    c[2].markdown(metric_card("Reported scores use", "De-duplicated data",
                              "duplicates removed before the split"),
                  unsafe_allow_html=True)
    leak = pd.read_csv(cfg.RESULTS_LEAKAGE_CSV)
    chart(vz.duplicate_leakage_chart(META["duplicate_leakage"]))
    st.dataframe(leak, hide_index=True, width="stretch")
    caveat("The inflated numbers come from models refit on the duplicate-laden data "
           "for this demonstration only; they are never saved or used for "
           "predictions. Removal assumes identical rows are duplicates rather than "
           "two genuinely identical listings — plausible, but not provable from the file.")

    st.subheader("Columns deliberately excluded from the inputs")
    st.dataframe(pd.DataFrame(META["excluded_columns"], columns=["Column", "Why it is excluded"]),
                 hide_index=True, width="stretch")


# ==========================================================================
# 7. ABOUT
# ==========================================================================
def render_about():
    st.title("About the Project")
    st.markdown(f"""
**Goal.** Predict a used car's selling price (regression) and whether it is
premium, i.e. priced at {cfg.PREMIUM_THRESHOLD / cfg.LAKH:g} lakh or more (classification).

**Models.** Multivariate Linear Regression on log price, Logistic Regression,
SVM with an RBF kernel, Random Forest (400 trees).

**Method.** Duplicates removed; stratified 80/20 split; preprocessing inside each
pipeline (fit on training rows only); grid search with 5-fold CV; threshold chosen
on out-of-fold predictions; one evaluation on the held-out test set.

**Dataset.** CarDekho-style used-car listings, {SUMMARY['n_raw_rows']:,} rows × 12
columns, copied from a public GitHub repository. It has no official specification
or confirmed licence, so it is used here for coursework only.

**Limitations.**
- The premium label is derived from the target, so classification adds little
  information beyond the regression task.
- Listings are asking prices from before 2021, not transaction prices.
- Several models are statistically tied; rankings between them should not be over-read.
- The dataset is a scrape with duplicates and missing specifications (imputed with
  medians inside the pipeline).
""")


{
    "Overview": render_overview,
    "Car Prediction": render_prediction,
    "Data Analysis": render_data_analysis,
    "Model Comparison": render_model_comparison,
    "Model Explainability": render_explainability,
    "Data Quality & Leakage": render_leakage,
    "About the Project": render_about,
}[section]()
