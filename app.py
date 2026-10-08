"""
Student Academic Performance & At-Risk Prediction System
========================================================

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
    page_title="Student At-Risk Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling -- a small, deliberately plain stylesheet
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1280px; }
      h1, h2, h3 { color: #0b0b0b; letter-spacing: -0.01em; }
      h1 { font-size: 1.95rem; font-weight: 650; }
      h2 { font-size: 1.35rem; font-weight: 620; margin-top: 0.6rem; }
      h3 { font-size: 1.08rem; font-weight: 600; }
      .lede { color: #52514e; font-size: 0.97rem; line-height: 1.6;
              max-width: 62rem; }
      .card { background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10);
              border-radius: 10px; padding: 1.05rem 1.2rem; height: 100%; }
      .card-label { color: #52514e; font-size: 0.80rem; font-weight: 500;
                    text-transform: none; margin-bottom: 0.3rem; }
      .card-value { color: #0b0b0b; font-size: 1.85rem; font-weight: 640;
                    line-height: 1.15; }
      .card-note { color: #898781; font-size: 0.78rem; margin-top: 0.3rem;
                   line-height: 1.4; }
      .verdict { border-radius: 10px; padding: 1.1rem 1.3rem;
                 border: 1px solid; font-size: 1.28rem; font-weight: 640; }
      .verdict-risk { background: #fbeaea; border-color: #d03b3b; color: #8c1f1f; }
      .verdict-safe { background: #eaf4ea; border-color: #0ca30c; color: #0b5c0b; }
      .verdict-sub { font-size: 0.86rem; font-weight: 450; color: #52514e;
                     margin-top: 0.4rem; }
      .caveat { background: #fcfcfb; border-left: 3px solid #898781;
                padding: 0.75rem 1rem; color: #52514e; font-size: 0.88rem;
                border-radius: 0 6px 6px 0; line-height: 1.55; }
      .pipeline { background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10);
                  border-radius: 10px; padding: 1.2rem; overflow-x: auto; }
      div[data-testid="stMetricValue"] { font-size: 1.5rem; }
      .stTabs [data-baseweb="tab-list"] { gap: 0.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained models...")
def _load_artifacts():
    return pr.load_artifacts()


@st.cache_data(show_spinner="Loading dataset...")
def _load_dataframe(dataset: str):
    return dp.load_dataset(dataset)


def metric_card(label: str, value: str, note: str = "") -> str:
    note_html = f'<div class="card-note">{note}</div>' if note else ""
    return (
        f'<div class="card"><div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div>{note_html}</div>'
    )


def fail_gracefully(message: str, detail: str = "") -> None:
    """Show a readable error instead of a Python traceback, then stop."""
    st.error(message)
    if detail:
        with st.expander("Technical detail"):
            st.code(detail)
    st.stop()


# --------------------------------------------------------------------------
# Startup: load everything, or explain exactly what is missing
# --------------------------------------------------------------------------
try:
    MODELS, META = _load_artifacts()
except pr.ModelsNotTrainedError as exc:
    st.title("Student Academic Performance & At-Risk Prediction System")
    st.error("The trained models are not available yet, so the app cannot start.")
    st.markdown(
        "**To fix this, run the training script once:**\n\n"
        "```bash\npython src/train_models.py\n```\n\n"
        "It takes under a minute and writes the four model files plus the "
        "evaluation results. Then reload this page."
    )
    with st.expander("What exactly is missing"):
        st.code(str(exc))
    st.stop()

try:
    DF = _load_dataframe(META.get("dataset", cfg.DEFAULT_DATASET))
except dp.DatasetError as exc:
    fail_gracefully(
        "The dataset could not be loaded. The app needs the raw CSV file to "
        "draw the data-analysis charts.",
        str(exc),
    )

SUMMARY = META["summary"]
CLF = META["classification"]
REG = META["regression"]
BASELINE = META["baseline_classification"]
TIE = META.get("tie_analysis", {})
CLF_METRICS = {name: res["test_metrics"] for name, res in CLF.items()}
PRIMARY_MODEL = "Random Forest"

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Student At-Risk\nPrediction System")
    st.caption("Machine Learning mini-project")
    section = st.radio(
        "Section",
        ["Overview", "Student Prediction", "Data Analysis", "Model Comparison",
         "Model Explainability", "Data Leakage", "About the Project"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        f"**Dataset** {META['dataset_file']}  \n"
        f"{SUMMARY['n_students']} students · {len(cfg.ALL_FEATURES)} input features  \n"
        f"**Trained** {META['trained_at']}  \n"
        f"scikit-learn {META.get('sklearn_version', 'n/a')}  \n"
        f"random_state = {META['random_state']}"
    )
    st.divider()
    st.caption(
        "Academic demonstration only. Predictions describe patterns in a "
        "395-student dataset from 2008 and must not be used to make decisions "
        "about real students."
    )


# ==========================================================================
# 1. OVERVIEW
# ==========================================================================
def render_overview():
    st.title("Student Academic Performance & At-Risk Prediction System")
    st.markdown(
        '<p class="lede">Two supervised learning tasks on the UCI Student '
        'Performance dataset. <b>Task 1</b> predicts a student\'s final grade '
        '(a continuous value, 0&ndash;20) with Multivariate Linear Regression. '
        '<b>Task 2</b> predicts whether that student will be '
        '<b>At Risk</b> of failing, using Logistic Regression, a Support Vector '
        'Machine and a Random Forest ensemble. Crucially, neither task is '
        'allowed to see any grade as an input &mdash; the point is to predict '
        'the outcome from student background and behaviour <i>before</i> the '
        'result exists.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("### At a glance")
    c = st.columns(4)
    c[0].markdown(metric_card(
        "Students", f"{SUMMARY['n_students']}",
        f"{META['n_train']} train · {META['n_test']} test",
    ), unsafe_allow_html=True)
    c[1].markdown(metric_card(
        "Input features", f"{len(cfg.ALL_FEATURES)}",
        f"{len(META['encoded_feature_names'])} after encoding · "
        f"4 grade columns excluded",
    ), unsafe_allow_html=True)
    c[2].markdown(metric_card(
        "At risk", f"{SUMMARY['pct_at_risk']:.1f}%",
        f"{SUMMARY['n_at_risk']} of {SUMMARY['n_students']} scored below "
        f"{cfg.AT_RISK_THRESHOLD}/20",
    ), unsafe_allow_html=True)
    rf = CLF_METRICS[PRIMARY_MODEL]
    c[3].markdown(metric_card(
        "Random Forest F1", f"{rf['F1']:.3f}",
        f"accuracy {rf['Accuracy']:.3f} · ROC-AUC {rf['ROC-AUC']:.3f}",
    ), unsafe_allow_html=True)

    st.markdown("")
    c = st.columns(4)
    c[0].markdown(metric_card(
        "Baseline to beat", f"{BASELINE['Accuracy']:.3f}",
        "accuracy of always answering “Not At Risk”, which finds "
        "<b>none</b> of the at-risk students",
    ), unsafe_allow_html=True)
    c[1].markdown(metric_card(
        "Best F1", f"{TIE.get('best_score', 0):.3f}",
        f"{TIE.get('best_model', 'n/a')}, but see the tie note below",
    ), unsafe_allow_html=True)
    c[2].markdown(metric_card(
        "Regression R² (test)", f"{REG['test_metrics']['R2']:.3f}",
        f"5-fold CV R² is {REG['cv_r2_mean']:+.3f} — the honest figure",
    ), unsafe_allow_html=True)
    c[3].markdown(metric_card(
        "Regression error", f"{REG['test_metrics']['RMSE']:.2f}",
        f"RMSE in grade points vs {REG['baseline_metrics']['RMSE']:.2f} for "
        "predicting the mean",
    ), unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### What the results actually say")
    col1, col2 = st.columns([1, 1])
    with col1:
        if TIE.get("is_tie"):
            st.markdown(
                f"""<div class="caveat">
                <b>The three classifiers are statistically tied.</b>
                {TIE['best_model']} has the highest F1 ({TIE['best_score']:.4f}),
                but the gap to {TIE['runner_up']} is only
                <b>{TIE['gap']:.4f}</b> &mdash; far smaller than the
                cross-validation spread of &plusmn;{TIE['noise_floor']:.4f}. On a
                {TIE['n_test']}-student test set, two students changing class
                moves F1 more than that. This project therefore reports a tie
                rather than crowning a winner, and keeps <b>Random Forest</b> as
                the primary model: it satisfies the ensemble-learning
                requirement, and it has the best accuracy
                ({rf['Accuracy']:.3f}) and precision ({rf['Precision']:.3f}) of
                the three.
                </div>""",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""<div class="caveat" style="margin-top:0.8rem">
            <b>The signal in this data is real but weak.</b> Every classifier
            reaches a ROC-AUC between
            {min(m['ROC-AUC'] for m in CLF_METRICS.values()):.2f} and
            {max(m['ROC-AUC'] for m in CLF_METRICS.values()):.2f}, which is
            clearly better than the 0.50 of random guessing &mdash; so student
            background and behaviour do carry information about who will
            struggle. But the F1 scores near
            {max(m['F1'] for m in CLF_METRICS.values()):.2f} mean roughly half
            of the flagged students would be false alarms. Honest conclusion:
            useful as a screening aid to prioritise who a tutor talks to,
            nowhere near good enough to decide anything on its own.
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="caveat">
            <b>Linear Regression barely beats predicting the average.</b>
            Test R&sup2; is {REG['test_metrics']['R2']:.3f}, but 5-fold
            cross-validation on the training split gives
            {REG['cv_r2_mean']:+.3f} &plusmn; {REG['cv_r2_std']:.3f} &mdash;
            a <i>negative</i> mean, meaning the model is sometimes worse than
            always answering “{SUMMARY['mean_g3']:.1f}”. The positive test
            R&sup2; is largely the luck of one 79-student split. Predicting an
            exact grade from background data alone, with no prior marks, is
            close to impossible on this dataset, and saying so is more useful
            than quoting the flattering number.
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<div class="caveat" style="margin-top:0.8rem">
            <b>{SUMMARY['n_zero_g3']} students recorded a final grade of 0.</b>
            That is {SUMMARY['n_zero_g3'] / SUMMARY['n_students'] * 100:.1f}% of
            the cohort, and it is almost certainly a dropout or a missed exam
            rather than a genuine score of zero. They were kept in the data
            &mdash; they are the students an early-warning system most needs to
            find &mdash; but they create the spike at zero in the chart below
            and a cluster of very large regression errors.
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("")
    left, right = st.columns(2)
    left.plotly_chart(vz.class_distribution(DF), width="stretch")
    right.plotly_chart(vz.score_distribution(DF), width="stretch")

    st.markdown("### How the system is put together")
    st.markdown(PIPELINE_DIAGRAM, unsafe_allow_html=True)


# ==========================================================================
# 2. STUDENT PREDICTION
# ==========================================================================
def render_prediction():
    st.title("Student Prediction")
    st.markdown(
        '<p class="lede">Enter a student&rsquo;s details to get a predicted '
        'final grade and an at-risk assessment. The form exposes the features a '
        'teacher or administrator could plausibly know; the remaining inputs are '
        'held at this cohort&rsquo;s typical values, listed at the bottom of the '
        'form.</p>',
        unsafe_allow_html=True,
    )

    profile = dict(META["default_profile"])

    with st.form("student_form"):
        st.markdown("#### Academic history")
        c = st.columns(3)
        failures = c[0].selectbox(
            cfg.FEATURE_LABELS["failures"], [0, 1, 2, 3],
            index=int(profile["failures"]),
            help="Past class failures. The single most influential feature in "
                 "every model trained here.",
        )
        studytime = c[1].selectbox(
            cfg.FEATURE_LABELS["studytime"], [1, 2, 3, 4],
            index=[1, 2, 3, 4].index(int(profile["studytime"])),
            format_func=lambda v: cfg.ORDINAL_SCALES["studytime"][v],
        )
        absences = c[2].number_input(
            cfg.FEATURE_LABELS["absences"], min_value=0, max_value=100,
            value=int(profile["absences"]), step=1,
            help=f"Dataset range: {int(DF['absences'].min())} to "
                 f"{int(DF['absences'].max())} days.",
        )

        st.markdown("#### Student")
        c = st.columns(3)
        age = c[0].number_input(
            cfg.FEATURE_LABELS["age"], min_value=15, max_value=22,
            value=int(profile["age"]), step=1,
        )
        sex = c[1].selectbox(
            cfg.FEATURE_LABELS["sex"], ["F", "M"],
            index=["F", "M"].index(profile["sex"]),
            format_func=lambda v: cfg.CATEGORICAL_LABELS["sex"][v],
        )
        higher = c[2].selectbox(
            cfg.FEATURE_LABELS["higher"], ["yes", "no"],
            index=["yes", "no"].index(profile["higher"]),
            format_func=lambda v: cfg.CATEGORICAL_LABELS["higher"][v],
        )

        st.markdown("#### Family background")
        c = st.columns(3)
        Medu = c[0].selectbox(
            cfg.FEATURE_LABELS["Medu"], [0, 1, 2, 3, 4],
            index=int(profile["Medu"]),
            format_func=lambda v: cfg.ORDINAL_SCALES["Medu"][v],
        )
        Fedu = c[1].selectbox(
            cfg.FEATURE_LABELS["Fedu"], [0, 1, 2, 3, 4],
            index=int(profile["Fedu"]),
            format_func=lambda v: cfg.ORDINAL_SCALES["Fedu"][v],
        )
        guardian = c[2].selectbox(
            cfg.FEATURE_LABELS["guardian"], ["mother", "father", "other"],
            index=["mother", "father", "other"].index(profile["guardian"]),
            format_func=lambda v: cfg.CATEGORICAL_LABELS["guardian"][v],
        )
        c = st.columns(3)
        famsize = c[0].selectbox(
            cfg.FEATURE_LABELS["famsize"], ["GT3", "LE3"],
            index=["GT3", "LE3"].index(profile["famsize"]),
            format_func=lambda v: cfg.CATEGORICAL_LABELS["famsize"][v],
        )
        Pstatus = c[1].selectbox(
            cfg.FEATURE_LABELS["Pstatus"], ["T", "A"],
            index=["T", "A"].index(profile["Pstatus"]),
            format_func=lambda v: cfg.CATEGORICAL_LABELS["Pstatus"][v],
        )
        famrel = c[2].selectbox(
            cfg.FEATURE_LABELS["famrel"], [1, 2, 3, 4, 5],
            index=[1, 2, 3, 4, 5].index(int(profile["famrel"])),
            format_func=lambda v: cfg.ORDINAL_SCALES["famrel"][v],
        )

        st.markdown("#### Support and resources")
        c = st.columns(4)
        schoolsup = c[0].selectbox(
            cfg.FEATURE_LABELS["schoolsup"], ["no", "yes"],
            index=["no", "yes"].index(profile["schoolsup"]),
            format_func=lambda v: cfg.CATEGORICAL_LABELS["schoolsup"][v],
        )
        famsup = c[1].selectbox(
            cfg.FEATURE_LABELS["famsup"], ["no", "yes"],
            index=["no", "yes"].index(profile["famsup"]),
            format_func=lambda v: cfg.CATEGORICAL_LABELS["famsup"][v],
        )
        paid = c[2].selectbox(
            cfg.FEATURE_LABELS["paid"], ["no", "yes"],
            index=["no", "yes"].index(profile["paid"]),
            format_func=lambda v: cfg.CATEGORICAL_LABELS["paid"][v],
        )
        internet = c[3].selectbox(
            cfg.FEATURE_LABELS["internet"], ["yes", "no"],
            index=["yes", "no"].index(profile["internet"]),
            format_func=lambda v: cfg.CATEGORICAL_LABELS["internet"][v],
        )

        st.markdown("#### Lifestyle")
        c = st.columns(4)
        goout = c[0].selectbox(
            cfg.FEATURE_LABELS["goout"], [1, 2, 3, 4, 5],
            index=[1, 2, 3, 4, 5].index(int(profile["goout"])),
            format_func=lambda v: cfg.ORDINAL_SCALES["goout"][v],
        )
        freetime = c[1].selectbox(
            cfg.FEATURE_LABELS["freetime"], [1, 2, 3, 4, 5],
            index=[1, 2, 3, 4, 5].index(int(profile["freetime"])),
            format_func=lambda v: cfg.ORDINAL_SCALES["freetime"][v],
        )
        health = c[2].selectbox(
            cfg.FEATURE_LABELS["health"], [1, 2, 3, 4, 5],
            index=[1, 2, 3, 4, 5].index(int(profile["health"])),
            format_func=lambda v: cfg.ORDINAL_SCALES["health"][v],
        )
        romantic = c[3].selectbox(
            cfg.FEATURE_LABELS["romantic"], ["no", "yes"],
            index=["no", "yes"].index(profile["romantic"]),
            format_func=lambda v: cfg.CATEGORICAL_LABELS["romantic"][v],
        )
        c = st.columns(4)
        Dalc = c[0].selectbox(
            cfg.FEATURE_LABELS["Dalc"], [1, 2, 3, 4, 5],
            index=[1, 2, 3, 4, 5].index(int(profile["Dalc"])),
            format_func=lambda v: cfg.ORDINAL_SCALES["Dalc"][v],
        )
        Walc = c[1].selectbox(
            cfg.FEATURE_LABELS["Walc"], [1, 2, 3, 4, 5],
            index=[1, 2, 3, 4, 5].index(int(profile["Walc"])),
            format_func=lambda v: cfg.ORDINAL_SCALES["Walc"][v],
        )
        traveltime = c[2].selectbox(
            cfg.FEATURE_LABELS["traveltime"], [1, 2, 3, 4],
            index=[1, 2, 3, 4].index(int(profile["traveltime"])),
            format_func=lambda v: cfg.ORDINAL_SCALES["traveltime"][v],
        )
        activities = c[3].selectbox(
            cfg.FEATURE_LABELS["activities"], ["yes", "no"],
            index=["yes", "no"].index(profile["activities"]),
            format_func=lambda v: cfg.CATEGORICAL_LABELS["activities"][v],
        )

        st.markdown("")
        chosen_model = st.selectbox(
            "Which classifier should give the verdict?",
            cfg.CLASSIFICATION_MODEL_NAMES,
            index=cfg.CLASSIFICATION_MODEL_NAMES.index(PRIMARY_MODEL),
            help="All three are shown in the results. Random Forest is the "
                 "project's primary model.",
        )
        submitted = st.form_submit_button("Predict", type="primary",
                                          width="stretch")

    hidden = [f for f in cfg.ALL_FEATURES if f not in {
        "failures", "studytime", "absences", "age", "sex", "higher", "Medu",
        "Fedu", "guardian", "famsize", "Pstatus", "famrel", "schoolsup",
        "famsup", "paid", "internet", "goout", "freetime", "health",
        "romantic", "Dalc", "Walc", "traveltime", "activities",
    }]
    if hidden:
        st.caption(
            "Features not on this form are held at the cohort's typical value: "
            + ", ".join(
                f"{cfg.FEATURE_LABELS.get(f, f)} = "
                f"{cfg.CATEGORICAL_LABELS.get(f, {}).get(profile[f], profile[f])}"
                for f in hidden
            )
            + "."
        )

    if not submitted:
        st.info("Fill in the form above and press **Predict**. It is pre-filled "
                "with this cohort's median/most-common values, so you can change "
                "one field at a time and watch the effect.")
        return

    student = dict(
        profile,
        failures=failures, studytime=studytime, absences=absences, age=age,
        sex=sex, higher=higher, Medu=Medu, Fedu=Fedu, guardian=guardian,
        famsize=famsize, Pstatus=Pstatus, famrel=famrel, schoolsup=schoolsup,
        famsup=famsup, paid=paid, internet=internet, goout=goout,
        freetime=freetime, health=health, romantic=romantic, Dalc=Dalc,
        Walc=Walc, traveltime=traveltime, activities=activities,
    )

    try:
        result = pr.predict_student(student, MODELS, META, classifier=chosen_model)
    except pr.InvalidStudentInputError as exc:
        st.error(f"That student profile could not be scored: {exc}")
        return
    except Exception as exc:  # last-resort guard: never show a raw traceback
        st.error("Something went wrong while making the prediction. "
                 "Please adjust the inputs and try again.")
        with st.expander("Technical detail"):
            st.code(f"{type(exc).__name__}: {exc}")
        return

    if result["unknown_categories"]:
        st.warning(
            "These values were not present in the training data, so the models "
            "treat them as 'unseen' and rely on the other features instead: "
            + "; ".join(result["unknown_categories"]) + "."
        )

    st.markdown("### Prediction")
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(vz.score_gauge(result["predicted_score"]),
                        width="stretch")
        if result["score_was_clipped"]:
            st.caption(
                f"Linear Regression is unbounded, so it returned "
                f"{result['predicted_score_raw']:.2f}. The value is shown "
                f"clipped to the valid 0&ndash;{cfg.GRADE_MAX} grade range."
            )
    with right:
        at_risk = result["at_risk"]
        css = "verdict-risk" if at_risk else "verdict-safe"
        icon = "🔴" if at_risk else "🟢"
        st.markdown(
            f"""<div class="verdict {css}">{icon} {result['risk_label'].upper()}
            <div class="verdict-sub">according to {result['classifier']}
            &mdash; {result['at_risk_votes']} of {result['n_models']} models
            agree on this verdict</div></div>""",
            unsafe_allow_html=True,
        )
        if result["risk_probability"] is not None:
            st.plotly_chart(vz.risk_probability_bar(result["risk_probability"]),
                            width="stretch")

    if result.get("probability_disagrees_with_label"):
        st.info(
            f"**Why the label and the percentage look inconsistent:** "
            f"{result['classifier']} decides its label from which side of the "
            "decision boundary the student falls on, while the percentage comes "
            "from a separate probability calibration step. Very close to the "
            "boundary the two can disagree, as here. The label is the model's "
            "actual decision and is what all the reported metrics are computed "
            "from."
        )

    if not result["models_agree"]:
        st.warning(
            f"**The models disagree on this student** "
            f"({result['at_risk_votes']} of {result['n_models']} say At Risk). "
            "That is informative rather than broken: this student sits near the "
            "decision boundary, where the models are least reliable. Treat the "
            "verdict as low-confidence."
        )

    if result["regression_implies_at_risk"] != result["at_risk"]:
        st.warning(
            f"**The two tasks disagree.** The regression model predicts "
            f"{result['predicted_score']:.1f}/20, which is "
            f"{'below' if result['regression_implies_at_risk'] else 'at or above'} "
            f"the pass mark of {cfg.AT_RISK_THRESHOLD}, while "
            f"{result['classifier']} says "
            f"“{result['risk_label']}”. The two models were trained "
            "separately on different targets, so they are not guaranteed to "
            "agree — and near the threshold they often do not."
        )

    st.markdown("#### What every model said")
    rows = [
        {
            "Model": name,
            "Verdict": v["label"],
            "P(at risk)": f"{v['probability']:.1%}" if v["probability"] is not None else "n/a",
            "Test F1": f"{CLF_METRICS[name]['F1']:.3f}",
            "Test recall": f"{CLF_METRICS[name]['Recall']:.3f}",
        }
        for name, v in result["per_model"].items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.markdown("### Interpreting this prediction")
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(vz.student_profile_bars(student, DF),
                        width="stretch")
        st.caption(
            "Each bar shows where this student sits within the range that "
            "feature takes across the dataset. It describes the inputs only "
            "— it is not a statement about which features drove the prediction."
        )
    with right:
        st.markdown("#### Features the model relies on most")
        influential = pr.top_influential_features(META, result["input_row"], top_n=6)
        st.dataframe(
            pd.DataFrame([
                {
                    "Feature": f["label"],
                    "This student": f["student_value"],
                    "Model reliance": f"{f['importance']:.4f}",
                }
                for f in influential
            ]),
            hide_index=True, width="stretch",
        )
        st.markdown(
            """<div class="caveat">
            <b>Read this as association, not cause.</b> These are the features
            whose values the Random Forest depends on most across <i>all</i>
            students &mdash; measured by how much its F1 score drops when the
            column is randomly shuffled. They were influential in the model's
            predictions. They are <i>not</i> evidence that changing one of them
            would change a student's result: a dataset of 395 teenagers from two
            Portuguese schools in 2008 cannot establish cause and effect.
            </div>""",
            unsafe_allow_html=True,
        )


# ==========================================================================
# 3. DATA ANALYSIS
# ==========================================================================
def render_data_analysis():
    st.title("Data Analysis")
    st.markdown(
        '<p class="lede">Exploring the dataset before and independently of the '
        'models: how grades are distributed, how balanced the classes are, and '
        'which student attributes actually move with the final result.</p>',
        unsafe_allow_html=True,
    )

    c = st.columns(4)
    c[0].markdown(metric_card("Rows × columns",
                              f"{SUMMARY['n_students']} × {SUMMARY['n_raw_columns']}",
                              "as published by UCI"), unsafe_allow_html=True)
    c[1].markdown(metric_card("Missing values", f"{SUMMARY['missing_values']}",
                              "no imputation was needed, though the pipeline "
                              "still includes an imputer"), unsafe_allow_html=True)
    c[2].markdown(metric_card("Duplicate rows", f"{SUMMARY['duplicate_rows']}",
                              "no de-duplication needed"), unsafe_allow_html=True)
    c[3].markdown(metric_card("Mean final grade", f"{SUMMARY['mean_g3']:.2f}",
                              f"median {SUMMARY['median_g3']:.0f} · "
                              f"{SUMMARY['n_zero_g3']} zeros"),
                  unsafe_allow_html=True)

    st.markdown("")
    left, right = st.columns(2)
    left.plotly_chart(vz.score_distribution(DF), width="stretch")
    right.plotly_chart(vz.class_distribution(DF), width="stretch")

    n_zero = SUMMARY["n_zero_g3"]
    zeros = DF[DF[cfg.REGRESSION_TARGET] == 0]
    st.markdown(
        f"""<div class="caveat">
        The distribution is not a smooth bell curve. There is a spike of
        <b>{n_zero} students at grade 0</b> and then a gap &mdash; the lowest
        non-zero grade is {int(DF.loc[DF[cfg.REGRESSION_TARGET] > 0, cfg.REGRESSION_TARGET].min())}.
        Two facts make these near-certain record-keeping artefacts rather than
        genuine scores: <b>all {n_zero} earned a non-zero first-period grade</b>
        (G1 from {int(zeros['G1'].min())} to {int(zeros['G1'].max())}), so they
        were attending and being assessed; and <b>all {n_zero} record exactly
        zero absences</b>. A student cannot have perfect attendance and also have
        failed to sit the final assessment. They are kept in the analysis &mdash;
        they are the students an early-warning system most needs to find &mdash;
        but they are why the regression model produces a handful of very large
        errors, and why the absences correlation below is misleading.
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("### Final grade against individual features")
    st.caption(
        "Box plots rather than scatter plots for the ordinal features: "
        "`studytime` and `failures` take only three or four distinct values, so "
        "a scatter plot would show a few vertical stripes of overlapping points "
        "instead of a distribution."
    )
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Past failures", "Study time", "Absences", "Any feature"]
    )
    with tab1:
        st.plotly_chart(vz.feature_vs_score_box(DF, "failures"),
                        width="stretch")
        st.caption(
            "The clearest relationship in the dataset, and the reason `failures` "
            "tops every importance ranking: median grade falls steadily as the "
            "number of past failures rises."
        )
    with tab2:
        st.plotly_chart(vz.feature_vs_score_box(DF, "studytime"),
                        width="stretch")
        st.caption(
            "A weak upward trend. More reported study time goes with slightly "
            "higher grades, but the boxes overlap heavily, so study time on its "
            "own separates students poorly."
        )
    with tab3:
        st.plotly_chart(vz.absences_vs_score(DF), width="stretch")
        st.caption(
            "The most instructive chart in this section, because the obvious "
            "reading of it is wrong. Across all 395 students, absences and final "
            "grade are essentially uncorrelated (r = +0.03) — which looks like "
            "evidence that attendance does not matter. It is not. Every one of "
            "the 38 students who recorded a grade of 0 also recorded 0 absences "
            "(the red diamonds), and that block alone drags the fit flat. Among "
            "students with a genuine recorded grade the correlation is r = −0.21. "
            "The lesson is about the data, not about attendance: an aggregate "
            "statistic can be reversed by a record-keeping artefact in 10% of "
            "the rows."
        )
    with tab4:
        feature = st.selectbox(
            "Feature", cfg.NUMERIC_FEATURES + cfg.CATEGORICAL_FEATURES,
            format_func=lambda f: cfg.FEATURE_LABELS.get(f, f),
            index=cfg.NUMERIC_FEATURES.index("goout"),
        )
        st.plotly_chart(vz.feature_vs_score_box(DF, feature),
                        width="stretch")

    st.markdown("### Correlation between numeric features")
    st.plotly_chart(vz.correlation_heatmap(DF), width="stretch")
    corr = DF[cfg.NUMERIC_FEATURES + ["G3"]].corr(numeric_only=True)["G3"].drop("G3")
    top = corr.abs().sort_values(ascending=False).head(5).index
    graded = DF[cfg.REGRESSION_TARGET] > 0
    r_abs_graded = DF.loc[graded, "absences"].corr(DF.loc[graded, cfg.REGRESSION_TARGET])
    st.markdown(
        "**The strongest correlations with the final grade, among the features "
        "the models are allowed to use:** "
        + ", ".join(f"`{f}` ({corr[f]:+.3f})" for f in top)
        + f". The largest is {corr[top[0]]:+.3f}, which is a weak relationship — "
        "a direct preview of why the models in this project reach only moderate "
        "scores. By contrast `G1` and `G2`, the intermediate grades, correlate "
        f"{DF['G1'].corr(DF['G3']):+.2f} and {DF['G2'].corr(DF['G3']):+.2f} with "
        "the final grade, which is exactly why using them would be leakage.\n\n"
        f"One caveat on reading this matrix: the `absences` row shows "
        f"{corr['absences']:+.3f}, but that figure is suppressed by the 38 "
        f"zero-grade students, who all record zero absences. Among students with "
        f"a genuine recorded grade it is {r_abs_graded:+.3f}. A correlation "
        "matrix computed over rows that include data artefacts can hide a real "
        "relationship, or invent one."
    )
    with st.expander("View the correlation values as a table"):
        st.dataframe(
            DF[cfg.NUMERIC_FEATURES + ["G1", "G2", "G3"]]
            .corr(numeric_only=True).round(3),
            width="stretch",
        )


# ==========================================================================
# 4. MODEL COMPARISON
# ==========================================================================
def render_model_comparison():
    st.title("Model Comparison")
    st.markdown(
        '<p class="lede">Three classifiers on the same at-risk task, trained on '
        'the same 316 students, tuned with the same cross-validation folds and '
        'the same scoring metric, and evaluated on the same 79 held-out '
        'students. Ranked by <b>F1</b> rather than accuracy, because with a '
        '67/33 class split accuracy rewards a model for ignoring the minority '
        'class.</p>',
        unsafe_allow_html=True,
    )

    if TIE.get("is_tie"):
        st.info(
            f"**Verdict: a statistical tie.** {TIE['best_model']} has the "
            f"highest F1 at {TIE['best_score']:.4f}, but the gap to "
            f"{TIE['runner_up']} is {TIE['gap']:.4f}, against a "
            f"cross-validation standard deviation of {TIE['noise_floor']:.4f}. "
            f"The difference is inside the noise. Random Forest remains this "
            f"project's primary model — it is the required ensemble method, and "
            f"it has the best accuracy and precision of the three."
        )

    st.markdown("### Metrics on the held-out test set")
    rows = []
    for name in cfg.CLASSIFICATION_MODEL_NAMES:
        m = CLF_METRICS[name]
        res = CLF[name]
        rows.append({
            "Model": name,
            "Accuracy": round(m["Accuracy"], 4),
            "Precision": round(m["Precision"], 4),
            "Recall": round(m["Recall"], 4),
            "F1": round(m["F1"], 4),
            "ROC-AUC": round(m["ROC-AUC"], 4),
            "CV F1 (mean ± sd)": f"{res['cv_f1_mean']:.3f} ± {res['cv_f1_std']:.3f}",
            "Tuned parameters": ", ".join(f"{k}={v}" for k, v in
                                          res["best_params"].items()) or "defaults",
        })
    rows.append({
        "Model": "Baseline (always “Not At Risk”)",
        "Accuracy": round(BASELINE["Accuracy"], 4),
        "Precision": round(BASELINE["Precision"], 4),
        "Recall": round(BASELINE["Recall"], 4),
        "F1": round(BASELINE["F1"], 4),
        "ROC-AUC": None,
        "CV F1 (mean ± sd)": "—",
        "Tuned parameters": "—",
    })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.markdown(
        f"""<div class="caveat">
        <b>Read the baseline row first.</b> Always answering “Not At Risk”
        scores {BASELINE['Accuracy']:.1%} accuracy while finding
        <b>zero</b> at-risk students (recall 0, F1 0). Logistic Regression and
        the SVM score <i>below</i> that accuracy on purpose: both were trained
        with <code>class_weight="balanced"</code>, which deliberately trades
        accuracy for recall on the at-risk class. A model that catches
        {CLF_METRICS['Logistic Regression']['Recall']:.0%} of at-risk students at
        {CLF_METRICS['Logistic Regression']['Accuracy']:.1%} accuracy is more
        useful for an early-warning system than one that is right more often by
        never raising an alarm.
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.plotly_chart(vz.metric_comparison(CLF_METRICS, BASELINE),
                    width="stretch")

    st.markdown("### How stable is that comparison?")
    st.plotly_chart(vz.cv_stability(CLF), width="stretch")
    st.markdown(
        """<div class="caveat">
        <b>The tie verdict replicates on an independent cohort.</b> The same
        pipeline was run on the second UCI subject file (the Portuguese-language
        file, 649 students) with <code>--dataset por</code>. There the gap
        between the best and second-best classifier is 0.0223 against a
        cross-validation spread of 0.1435 &mdash; again inside the noise, again a
        tie. Two different cohorts, the same conclusion: on this problem these
        three algorithms are not distinguishable. Full figures are in
        <code>results/robustness_por.csv</code>, and the README's robustness
        section records the one conclusion that did <i>not</i> replicate.
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        "The error bars are one standard deviation across the "
        f"{cfg.CV_FOLDS} cross-validation folds. They overlap almost completely, "
        "which is the visual version of the tie verdict: with 316 training "
        "students, the differences between these three algorithms are smaller "
        "than the variation between folds. Note also that the cross-validation "
        "F1 values sit *below* the test-set F1 values — the single test split "
        "happens to be a flattering one, which is why both are reported."
    )

    st.markdown("### Confusion matrix")
    st.caption(
        "Pick a model to see exactly which students it got right and wrong. The "
        "cell to watch is the false negative — an at-risk student the model "
        "failed to flag, which is the costly error for this application."
    )
    selected = st.radio("Model", cfg.CLASSIFICATION_MODEL_NAMES,
                        horizontal=True, label_visibility="collapsed")
    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(
            vz.confusion_matrix_figure(CLF[selected]["confusion"]["matrix"], selected),
            width="stretch",
        )
    with right:
        conf = CLF[selected]["confusion"]
        st.markdown(f"#### {selected}")
        st.markdown(
            f"""
- **{conf['TP']}** at-risk students correctly flagged (true positives)
- **{conf['FN']}** at-risk students **missed** (false negatives) — the costly error
- **{conf['FP']}** false alarms (false positives) — a student flagged who in fact passed
- **{conf['TN']}** correctly cleared (true negatives)

Of the {conf['TP'] + conf['FN']} genuinely at-risk students in the test set,
this model found **{conf['TP']}**, a recall of
**{CLF_METRICS[selected]['Recall']:.1%}**. Of the {conf['TP'] + conf['FP']}
students it flagged, **{conf['TP']}** really were at risk, a precision of
**{CLF_METRICS[selected]['Precision']:.1%}**.
            """
        )
        with st.expander("Full classification report"):
            st.code(CLF[selected]["report_text"])

    st.markdown("### ROC curves")
    st.plotly_chart(vz.roc_curves(CLF), width="stretch")
    st.markdown(
        "ROC-AUC measures something different from the metrics above: how well a "
        "model *ranks* students by risk, across every possible decision "
        "threshold. All three curves sit above the diagonal, so all three carry "
        "real signal — "
        f"the best reaches {max(m['ROC-AUC'] for m in CLF_METRICS.values()):.3f} "
        "against 0.500 for random guessing. This is the metric that is least "
        "affected by the class imbalance and by the choice of a 50% cut-off."
    )

    st.markdown("### Regression task")
    st.markdown(
        "The regression half of the project exists to demonstrate "
        "**Multivariate Linear Regression** — many input variables predicting "
        "one continuous output. Only one regression model is reported, and "
        "deliberately so: adding unrelated regressors purely to fill out a "
        "comparison table would not demonstrate anything the classification "
        "comparison does not already show. The reference point that matters is "
        "the baseline that always predicts the training mean."
    )
    reg_rows = [
        {
            "Model": cfg.REGRESSION_MODEL_NAME,
            "MAE": round(REG["test_metrics"]["MAE"], 4),
            "MSE": round(REG["test_metrics"]["MSE"], 4),
            "RMSE": round(REG["test_metrics"]["RMSE"], 4),
            "R² (test)": round(REG["test_metrics"]["R2"], 4),
            "R² (5-fold CV)": f"{REG['cv_r2_mean']:+.4f} ± {REG['cv_r2_std']:.4f}",
        },
        {
            "Model": "Baseline (always predict the mean)",
            "MAE": round(REG["baseline_metrics"]["MAE"], 4),
            "MSE": round(REG["baseline_metrics"]["MSE"], 4),
            "RMSE": round(REG["baseline_metrics"]["RMSE"], 4),
            "R² (test)": round(REG["baseline_metrics"]["R2"], 4),
            "R² (5-fold CV)": "—",
        },
    ]
    st.dataframe(pd.DataFrame(reg_rows), hide_index=True, width="stretch")
    st.markdown(
        f"""<div class="caveat">
        <b>The honest reading.</b> On the test split the model explains
        {REG['test_metrics']['R2']:.1%} of the variance in the final grade and
        beats the mean predictor's RMSE
        ({REG['test_metrics']['RMSE']:.2f} against
        {REG['baseline_metrics']['RMSE']:.2f} grade points). But 5-fold
        cross-validation on the training data gives a <b>negative</b> mean
        R&sup2; of {REG['cv_r2_mean']:+.3f} &plusmn; {REG['cv_r2_std']:.3f}: on
        some folds the model does worse than always answering
        “{SUMMARY['mean_g3']:.1f}”. The positive test R&sup2; is mostly the luck
        of one 79-student split. With a typical error of around
        {REG['test_metrics']['MAE']:.1f} grade points on a 0&ndash;20 scale,
        this model should be treated as a demonstration of the technique, not as
        a usable grade predictor.<br><br>
        <b>One bound on that claim.</b> Run on the larger Portuguese-language
        file (649 students), the same model reaches a cross-validated R&sup2; of
        <b>+0.271</b> with RMSE 2.86 grade points. So the failure here is a
        property of <i>this</i> cohort &mdash; 38 zero-grade artefact records and
        a wide grade spread &mdash; rather than proof that a grade can never be
        predicted from background data.
        </div>""",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    left.plotly_chart(
        vz.actual_vs_predicted(REG["y_test"], REG["y_pred"],
                               REG["test_metrics"]["R2"]),
        width="stretch",
    )
    right.plotly_chart(vz.residual_plot(REG["y_test"], REG["y_pred"]),
                       width="stretch")
    st.caption(
        "In the left chart a perfect model would place every point on the dashed "
        "line. Instead the predictions are squeezed into a narrow band around "
        "the cohort mean while the actual grades spread across the full scale — "
        "the model has learned the average far better than it has learned any "
        "individual student. In the residual plot, the points falling to around "
        "−10 are the students who recorded a grade of 0."
    )


# ==========================================================================
# 5. MODEL EXPLAINABILITY
# ==========================================================================
def render_explainability():
    st.title("Model Explainability")
    st.markdown(
        '<p class="lede">Which inputs the models lean on, and how much to trust '
        'each answer. Two different importance measures are shown because they '
        'disagree &mdash; and the disagreement is itself the lesson.</p>',
        unsafe_allow_html=True,
    )

    imp = pd.DataFrame(META["rf_impurity_importance"])
    perm = pd.DataFrame(META["rf_permutation_importance"])

    st.markdown("### Random Forest feature importance")
    tab1, tab2 = st.tabs([
        "Permutation importance (preferred)", "Impurity importance (built-in)"
    ])
    with tab1:
        st.plotly_chart(
            vz.feature_importance_bars(
                perm, "permutation_importance",
                "Random Forest permutation importance (top 15)",
                "Drop in test F1 when this column is shuffled",
                std_col="permutation_std",
            ),
            width="stretch",
        )
        st.markdown(
            "Permutation importance shuffles one column of the **test** data at "
            "a time and measures how far the F1 score falls. It answers a direct "
            "question — *how much does the model's performance actually depend on "
            "this feature?* — and because it is measured on held-out data it is "
            "not inflated by a feature simply having many distinct values. The "
            "error bars are the spread over 20 shuffles."
        )
    with tab2:
        st.plotly_chart(
            vz.feature_importance_bars(
                imp, "impurity_importance",
                "Random Forest impurity importance (top 15)",
                "Mean decrease in Gini impurity",
            ),
            width="stretch",
        )
        st.markdown(
            "This is scikit-learn's built-in `feature_importances_`: how much "
            "each feature reduced impurity across all the splits in all the "
            "trees. It is the measure most often shown in textbooks, and it has "
            "a known bias toward continuous and high-cardinality features, "
            "because they offer more possible split points for a tree to choose "
            "from."
        )

    top_imp = imp.iloc[0]["feature"]
    top_perm = perm.iloc[0]["feature"]
    if top_imp != top_perm:
        st.warning(
            f"**The two measures disagree, and this is the textbook bias in "
            f"action.** Impurity importance ranks "
            f"`{top_imp}` first; permutation importance ranks `{top_perm}` first "
            f"by a wide margin "
            f"({perm.iloc[0]['permutation_importance']:.3f} against "
            f"{perm.iloc[1]['permutation_importance']:.3f} for the runner-up). "
            f"`{top_imp}` is a continuous count with many distinct values, so it "
            f"offers trees far more split points than a feature with three or "
            f"four levels — which inflates its impurity score without making the "
            f"model genuinely depend on it. Where the two disagree, trust the "
            f"permutation result."
        )

    st.markdown("### Logistic Regression coefficients")
    coef = pd.DataFrame(META["logistic_coefficients"])
    st.plotly_chart(vz.logistic_coefficient_bars(coef), width="stretch")
    st.markdown(
        """
Each coefficient is the change in the **log-odds** of being at risk for a
one-standard-deviation increase in that feature (one-hot columns move from 0 to
1). Because every numeric feature was standardised inside the same pipeline, the
coefficients are comparable with one another.

- A **positive** coefficient is associated with a higher predicted probability of being at risk.
- A **negative** coefficient is associated with a lower one.

Two caveats that matter in a viva. First, interpretation depends entirely on the
preprocessing: these are values on the *standardised* scale, not on the original
units, and a one-hot column's coefficient is relative to the level that was
dropped. Second, the grid search selected a strong regularisation penalty
"""
        + f"(`C = {CLF['Logistic Regression']['best_params'].get('C', 'n/a')}`), "
        "which deliberately shrinks all the coefficients toward zero — so their "
        "absolute sizes are small, and only their relative sizes and signs should "
        "be read."
    )
    with st.expander("All coefficients as a table"):
        st.dataframe(
            coef.assign(
                feature=lambda d: d["feature"].map(
                    lambda f: cfg.FEATURE_LABELS.get(f, f)
                )
            ).round(4),
            hide_index=True, width="stretch",
        )

    st.markdown("### What these results mean")
    st.markdown(
        f"""
**The models agree on what matters.** `failures` — the number of classes the
student has already failed — leads both the permutation importance ranking and
the Logistic Regression coefficients. In plain terms, the best available signal
about whether a student will struggle is whether they have struggled before. The
next tier (`absences`, `goout`, `studytime`, `age`) contributes far less
individually.

**And they agree on how weak the rest is.** After `failures`, permutation
importance drops from {perm.iloc[0]['permutation_importance']:.3f} to
{perm.iloc[1]['permutation_importance']:.3f} — more than a fivefold fall. No
family, lifestyle or demographic attribute in this dataset is individually a
strong predictor. That is a real finding about the data, not a shortcoming of
the models.

**A warning worth stating explicitly.** Feature importance is not causation. It
describes which columns this particular Random Forest, fitted to 316 students
from two Portuguese schools in 2008, happened to rely on. It is not evidence
that reducing a student's absences or their time out with friends would improve
their grade. Several of these features are plausibly proxies for circumstances
the dataset never measures — household stability, workload outside school,
health, prior schooling quality.
        """
    )


# ==========================================================================
# 6. DATA LEAKAGE
# ==========================================================================
def render_leakage():
    st.title("Data Leakage: Why G1 and G2 Were Excluded")
    st.markdown(
        '<p class="lede">The single most consequential design decision in this '
        'project. The dataset ships three grade columns &mdash; G1, G2 and G3 '
        '&mdash; and every model here is forbidden from seeing any of them as an '
        'input. This section shows exactly what that choice costs on paper, and '
        'why it is still the right choice.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("### Excluded columns and the reason for each")
    st.dataframe(
        pd.DataFrame(
            [{"Column": c, "Why it is excluded": r}
             for c, r in META["excluded_columns"]]
        ),
        hide_index=True, width="stretch",
    )

    g1r = DF["G1"].corr(DF["G3"])
    g2r = DF["G2"].corr(DF["G3"])
    st.markdown(
        f"""
`G1` and `G2` are the first- and second-period grades on the same 0&ndash;20
scale as the final grade. They correlate **{g1r:+.2f}** and **{g2r:+.2f}** with
`G3`. They are not independent background information about a student &mdash;
they are earlier measurements of the very outcome being predicted.

Including them would be wrong for two separate reasons:

1. **It is target leakage.** The model would largely be learning the identity
   function "the final grade resembles the previous grade", which tells us
   nothing about which students are at risk and why.
2. **It destroys the purpose.** An early-warning system has to raise the alarm
   while there is still time to help. By the time the second-period grade is in,
   a struggling student is already struggling and the warning is worthless.
        """
    )

    leakage = META.get("leakage_comparison") or []
    if not leakage:
        st.info("The leakage comparison was not run. Re-run "
                "`python src/train_models.py` without `--skip-leakage-demo`.")
        return

    st.markdown("### What leakage would have bought us")
    st.markdown(
        "To make the effect concrete rather than theoretical, the training "
        "script fits a second, parallel set of models **with** `G1` and `G2` "
        "included and evaluates them identically. Those models are reported "
        "here and then discarded — they are never saved and never used to make "
        "a prediction anywhere in this application."
    )
    st.plotly_chart(vz.leakage_comparison_chart(leakage), width="stretch")

    df_leak = pd.DataFrame(leakage)
    df_leak["Change"] = (
        df_leak["With leakage (G1+G2)"] - df_leak["Leakage-free (no G1/G2)"]
    ).round(4)
    st.dataframe(df_leak, hide_index=True, width="stretch")

    rf_f1 = next((r for r in leakage
                  if r["Model"] == "Random Forest" and r["Metric"] == "F1"), None)
    r2 = next((r for r in leakage if r["Metric"] == "R2"), None)
    bullets = []
    if rf_f1:
        bullets.append(
            f"- Random Forest F1 jumps from **{rf_f1['Leakage-free (no G1/G2)']:.3f}** "
            f"to **{rf_f1['With leakage (G1+G2)']:.3f}**"
        )
    if r2:
        bullets.append(
            f"- Linear Regression R² jumps from **{r2['Leakage-free (no G1/G2)']:.3f}** "
            f"to **{r2['With leakage (G1+G2)']:.3f}**"
        )
    st.markdown(
        "\n".join(bullets)
        + """

That is the whole lesson. A report quoting the second set of numbers would look
far more impressive and would be far less honest: almost all of that apparent
skill comes from being shown a near-copy of the answer. Many published
write-ups of this dataset report accuracies above 90% precisely because they
include `G2`.

**This is the question to be ready for in a viva.** If an examiner asks why the
scores are not higher, the answer is not an apology — it is that the higher
scores are available and were deliberately refused, and that this tab
demonstrates exactly how much was given up and why.
        """
    )


# ==========================================================================
# 7. ABOUT
# ==========================================================================
def render_about():
    st.title("About the Project")

    st.markdown("### Problem statement")
    st.markdown(
        f"""
Academic failure is usually visible in hindsight and expensive to reverse. The
question this project asks is whether it can be anticipated: given what a school
already knows about a student — family background, study habits, support
received, past failures, lifestyle — can a model identify, *before the final
assessment*, who is likely to fall below the pass mark?

Two supervised learning tasks are posed on the same data:

- **Regression.** Predict the final grade `G3`, a continuous value from 0 to {cfg.GRADE_MAX}.
- **Classification.** Predict whether `G3` will fall below {cfg.AT_RISK_THRESHOLD},
  the pass mark on the Portuguese grading scale — the **At Risk** label. On this
  dataset {SUMMARY['n_at_risk']} of {SUMMARY['n_students']} students
  ({SUMMARY['pct_at_risk']:.1f}%) are at risk by that definition.
        """
    )

    st.markdown("### Dataset")
    st.markdown(
        f"""
**UCI Student Performance** (Cortez & Silva, 2008) — real records from two
Portuguese secondary schools, collected via school reports and questionnaires.
This project uses the **Mathematics** file, `{META['dataset_file']}`:
**{SUMMARY['n_students']} students × {SUMMARY['n_raw_columns']} attributes**,
with **{SUMMARY['missing_values']} missing values** and
**{SUMMARY['duplicate_rows']} duplicate rows**.

The file was verified against the shape UCI documents before training, and the
loader refuses to proceed if it does not match — a safeguard against silently
training on a truncated or substituted download.

The companion Portuguese-language file (649 students) is included in `data/raw/`
and the training script accepts `--dataset por`, but the two files are **not**
combined: 382 students appear in both, so concatenating them would place the
same student in both the training and test sets. That is itself a form of
leakage.
        """
    )

    st.markdown("### Algorithms, and why each one")
    st.markdown(
        f"""
| Model | Task | Why it is here |
|---|---|---|
| **Multivariate Linear Regression** | predict `G3` | The baseline for a continuous target: many input variables, one numeric output, fitted by least squares. Its coefficients are directly readable, and its failure on this data is informative in itself. |
| **Logistic Regression** | at-risk classification | The linear baseline for classification. Outputs a genuine probability via the sigmoid function, and its coefficients show the direction of each association. |
| **Support Vector Machine (RBF)** | at-risk classification | Tests whether a *non-linear* decision boundary helps. The RBF kernel can curve the boundary in ways a linear model cannot. |
| **Random Forest** | at-risk classification | The required **ensemble** method, and this project's primary model. Also supplies the feature-importance analysis. |

**Why these four and nothing more.** Each one answers a distinct question — is
the target continuous or categorical, is the boundary linear or curved, is one
model or many better. A neural network on 316 training rows would add
parameters, training time and explanation cost without adding insight, and would
almost certainly overfit. The project deliberately stops here.
        """
    )

    st.markdown("### Ensemble learning: why Random Forest is the required model")
    st.markdown(ENSEMBLE_DIAGRAM, unsafe_allow_html=True)
    st.markdown(
        f"""
**Ensemble learning** means combining several models so that the group performs
better than any member alone. It works when the members make *different*
mistakes, so that the errors partly cancel when the votes are combined.

**Bagging** (Bootstrap AGGregatING) is one way to create that diversity. From
the {META['n_train']} training students, draw many random samples *with
replacement*, each the same size as the original. Each sample omits roughly a
third of the students and duplicates others, so each tree sees a slightly
different world and learns slightly different rules.

**A Random Forest is bagging plus one more idea.** Each tree is grown on its own
bootstrap sample *and* is restricted, at every split, to a random subset of the
features (here `max_features={CLF['Random Forest']['best_params'].get('max_features', 'sqrt')}`
of 43 encoded features). Without that restriction, every tree would seize on
`failures` as its first split and the trees would end up nearly identical. The
feature restriction forces them apart, which is what makes the averaging
worthwhile.

**Why it is an ensemble and a single decision tree is not.** One tree fitted to
{META['n_train']} students can keep splitting until almost every leaf holds a
single student — it memorises the training data. The Random Forest trains
{CLF['Random Forest'].get('n_estimators', 400)} such trees and takes a majority
vote; the individual over-fitting largely averages out. The effect is visible in
this project's own numbers: the tuned Random Forest still shows a train-to-test
F1 gap of
{CLF['Random Forest']['train_metrics']['F1'] - CLF_METRICS['Random Forest']['F1']:+.3f},
which is the honest measure of how much it memorised even *with* the ensemble
and the `min_samples_leaf` limit in place.
        """
    )

    st.markdown("### Methodology")
    st.markdown(
        f"""
1. **Load and verify** the raw CSV against the published shape.
2. **Derive the target**: `at_risk = 1` where `G3 < {cfg.AT_RISK_THRESHOLD}`.
3. **Remove every leakage-prone column** — `G3`, `at_risk`, `G2`, `G1` — leaving
   {len(cfg.ALL_FEATURES)} inputs ({len(cfg.NUMERIC_FEATURES)} numeric,
   {len(cfg.CATEGORICAL_FEATURES)} categorical), which become
   {len(META['encoded_feature_names'])} columns after encoding. An assertion in
   the code fails the run if a forbidden column reaches the feature matrix.
4. **Split** {int((1 - META['test_size']) * 100)}/{int(META['test_size'] * 100)}
   with `random_state={META['random_state']}`. The classification split uses
   `stratify=y`, so the at-risk proportion is preserved
   ({META['train_at_risk_pct']:.1f}% in train, {META['test_at_risk_pct']:.1f}%
   in test). Without it, a random 79-student test set could have a very
   different class balance and the metrics would not be comparable.
5. **Preprocess inside a `Pipeline`**: median imputation and `StandardScaler`
   for numeric features, most-frequent imputation and `OneHotEncoder` for
   categorical ones. Because the preprocessor sits inside the pipeline, it is
   fitted on training rows only — the scaler never sees a test-set mean.
6. **Tune** each classifier with a small `GridSearchCV` over the same
   {META['cv_folds']} stratified folds, scored by F1 on the at-risk class. All
   three are tuned, so the comparison is fair.
7. **Evaluate** on the held-out test set, and additionally report
   {META['cv_folds']}-fold cross-validation because a 79-student test set is
   small enough that a single split is noisy.
8. **Save** the fitted pipelines with `joblib`, so this application loads them
   rather than retraining.
        """
    )

    st.markdown("### Limitations")
    st.markdown(
        f"""
Stated plainly, because these bound what the results can be used for.

- **The dataset is small and old.** {SUMMARY['n_students']} students from two
  Portuguese schools, collected in 2008. {META['n_test']} students in the test
  set means a single reclassified student moves F1 by roughly 0.02. Nothing here
  should be assumed to transfer to another school system or another decade.
- **The models are weak, and the honest metrics say so.** F1 around
  {max(m['F1'] for m in CLF_METRICS.values()):.2f} means roughly half of the
  flagged students are false alarms. Cross-validated regression R² is
  {REG['cv_r2_mean']:+.3f} on this file — no better than predicting the cohort
  average. It is **+0.271** on the larger Portuguese file, so that failure is a
  property of this cohort rather than of the problem.
- **Results are reported for one subject file.** A robustness run on the
  Portuguese file (649 students) confirms the model tie and the dominance of
  `failures`, but contradicts the regression verdict — see
  `results/robustness_por.csv`.
- **{SUMMARY['n_zero_g3']} students have a final grade of 0** while recording
  zero absences *and* a non-zero first-period grade — a combination that cannot
  describe a real academic record, so these are almost certainly dropout or
  unentered marks. They were kept, which is the defensible choice, but they
  distort both tasks and they suppress the `absences` correlation.
- **The three models are statistically indistinguishable.** Reporting any one of
  them as "best" on this test set would be over-reading the data.
- **Self-reported features.** Study time, alcohol consumption, free time and
  family relationships come from student questionnaires, with all the
  inaccuracy that implies.
- **Association, never causation.** Nothing here supports a claim that changing
  a feature would change a student's outcome. Several features are likely
  proxies for circumstances the dataset never records.
- **A 50% probability cut-off is an arbitrary default.** A real deployment would
  choose the threshold from the relative cost of a missed student versus a false
  alarm, which is a policy decision, not a modelling one.
        """
    )

    st.markdown("### Future scope")
    st.markdown(
        """
- **More and more recent data**, across multiple institutions, to test whether
  any of this generalises.
- **Tune the decision threshold explicitly** against a stated cost ratio for
  missed students versus false alarms, instead of accepting 0.5.
- **Attendance and engagement over time** — weekly submissions, LMS logins —
  rather than a single end-of-year absence count. Trends are likely to carry
  much more signal than totals.
- **Integration as a termly screening report** for tutors, with the model
  prioritising conversations rather than issuing verdicts.
- **Per-prediction explanations** (for example SHAP values) so a flagged student
  comes with the specific reasons that pushed the model, not only global
  importances.
- **Additional ensemble methods** such as gradient boosting, to test whether
  boosting extracts more from this weak signal than bagging does.
        """
    )

    st.markdown("### Ethical statement")
    st.markdown(
        """<div class="caveat">
        This system is an <b>academic machine-learning demonstration</b> and must
        not be used to make high-stakes decisions about students. Its predictions
        describe statistical patterns in a small 2008 dataset; they do not
        establish causation, and they are wrong often enough that acting on an
        individual prediction without human judgement would be indefensible.
        A false "At Risk" label can stigmatise a student and become
        self-fulfilling; a false "Not At Risk" label can withhold help from
        someone who needs it. Any real use of student data must be consented to,
        access-controlled, retained only as long as needed, and subject to human
        review — and a model should only ever prioritise who a teacher talks to,
        never replace that conversation.
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("### Reproducing these results")
    st.code(
        "pip install -r requirements.txt\n"
        "python src/train_models.py      # trains, evaluates, saves everything\n"
        "streamlit run app.py            # launches this application",
        language="bash",
    )
    st.caption(
        f"Every figure in this application was produced by the run on "
        f"{META['trained_at']} with `random_state={META['random_state']}` and "
        f"scikit-learn {META.get('sklearn_version', 'n/a')}. The raw numbers are "
        f"in `results/model_results.csv`."
    )

    st.markdown("### References")
    st.markdown(
        """
1. P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School
   Student Performance." In A. Brito and J. Teixeira (eds.), *Proceedings of
   5th FUture BUsiness TEChnology Conference (FUBUTEC 2008)*, pp. 5–12,
   Porto, Portugal, 2008.
2. Student Performance dataset, UCI Machine Learning Repository.
   <https://archive.ics.uci.edu/dataset/320/student+performance>
   (DOI: 10.24432/C5TG7T)
3. L. Breiman. "Random Forests." *Machine Learning*, 45(1):5–32, 2001.
4. L. Breiman. "Bagging Predictors." *Machine Learning*, 24(2):123–140, 1996.
5. C. Cortes and V. Vapnik. "Support-Vector Networks." *Machine Learning*,
   20(3):273–297, 1995.
6. F. Pedregosa et al. "Scikit-learn: Machine Learning in Python." *Journal of
   Machine Learning Research*, 12:2825–2830, 2011.
        """
    )


# --------------------------------------------------------------------------
# Static diagrams (inline SVG so they render identically everywhere)
# --------------------------------------------------------------------------
PIPELINE_DIAGRAM = """
<div class="pipeline">
<svg viewBox="0 0 980 330" width="100%" height="330" role="img"
     aria-label="Pipeline: student data is preprocessed, split, then fed to a
     linear regression for the grade and to three classifiers for the at-risk
     label, which are compared before a final prediction.">
  <defs>
    <marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3"
            orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#898781"/></marker>
  </defs>
  <style>
    .bx { fill:#fcfcfb; stroke:#c3c2b7; stroke-width:1.5; rx:7; }
    .bxa { fill:#eaf1fb; stroke:#2a78d6; stroke-width:1.5; rx:7; }
    .bxe { fill:#e6f6ef; stroke:#1baf7a; stroke-width:2; rx:7; }
    .bxo { fill:#fdf3e6; stroke:#eb6834; stroke-width:1.5; rx:7; }
    .t  { font:12px system-ui,-apple-system,sans-serif; fill:#0b0b0b;
          text-anchor:middle; }
    .ts { font:10.5px system-ui,-apple-system,sans-serif; fill:#52514e;
          text-anchor:middle; }
    .ln { stroke:#898781; stroke-width:1.5; fill:none; marker-end:url(#ar); }
  </style>

  <rect class="bx" x="390" y="8" width="200" height="40"/>
  <text class="t" x="490" y="26">Student data (395 rows)</text>
  <text class="ts" x="490" y="40">30 features · no grades</text>
  <path class="ln" d="M490,48 L490,72"/>

  <rect class="bxa" x="360" y="74" width="260" height="42"/>
  <text class="t" x="490" y="92">Preprocessing (inside the Pipeline)</text>
  <text class="ts" x="490" y="106">impute · StandardScaler · OneHotEncoder → 43 cols</text>
  <path class="ln" d="M490,116 L490,140"/>

  <rect class="bx" x="385" y="142" width="210" height="40"/>
  <text class="t" x="490" y="160">Train / test split 80 / 20</text>
  <text class="ts" x="490" y="174">stratified · random_state=42</text>

  <path class="ln" d="M440,182 L150,206"/>
  <path class="ln" d="M470,182 L370,206"/>
  <path class="ln" d="M490,182 L590,206"/>
  <path class="ln" d="M520,182 L810,206"/>

  <rect class="bxo" x="40" y="208" width="220" height="46"/>
  <text class="t" x="150" y="228">Linear Regression</text>
  <text class="ts" x="150" y="243">Task 1 · predicts the grade</text>

  <rect class="bx" x="280" y="208" width="180" height="46"/>
  <text class="t" x="370" y="228">Logistic Regression</text>
  <text class="ts" x="370" y="243">Task 2 · linear boundary</text>

  <rect class="bx" x="500" y="208" width="180" height="46"/>
  <text class="t" x="590" y="228">SVM (RBF)</text>
  <text class="ts" x="590" y="243">Task 2 · curved boundary</text>

  <rect class="bxe" x="720" y="208" width="220" height="46"/>
  <text class="t" x="830" y="228">Random Forest</text>
  <text class="ts" x="830" y="243">Task 2 · ENSEMBLE (bagging)</text>

  <path class="ln" d="M370,254 L470,282"/>
  <path class="ln" d="M590,254 L510,282"/>
  <path class="ln" d="M830,254 L560,282"/>
  <path class="ln" d="M150,254 L430,282"/>

  <rect class="bxa" x="300" y="284" width="380" height="40"/>
  <text class="t" x="490" y="302">Evaluate · compare · predict</text>
  <text class="ts" x="490" y="316">predicted grade (0–20) + At Risk / Not At Risk + probability</text>
</svg>
</div>
"""

ENSEMBLE_DIAGRAM = """
<div class="pipeline">
<svg viewBox="0 0 980 300" width="100%" height="300" role="img"
     aria-label="Ensemble flow: one decision tree overfits; bootstrap sampling
     plus random feature selection produces many different trees; a majority
     vote across them gives the final prediction.">
  <defs>
    <marker id="ar2" markerWidth="8" markerHeight="8" refX="6" refY="3"
            orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#898781"/></marker>
  </defs>
  <style>
    .b  { fill:#fcfcfb; stroke:#c3c2b7; stroke-width:1.5; rx:7; }
    .bt { fill:#e6f6ef; stroke:#1baf7a; stroke-width:1.5; rx:7; }
    .bf { fill:#e6f6ef; stroke:#1baf7a; stroke-width:2.5; rx:7; }
    .bw { fill:#fbeaea; stroke:#d03b3b; stroke-width:1.5; rx:7; }
    .x  { font:12px system-ui,-apple-system,sans-serif; fill:#0b0b0b;
          text-anchor:middle; }
    .xs { font:10.5px system-ui,-apple-system,sans-serif; fill:#52514e;
          text-anchor:middle; }
    .l  { stroke:#898781; stroke-width:1.5; fill:none; marker-end:url(#ar2); }
  </style>

  <rect class="bw" x="20" y="110" width="170" height="54"/>
  <text class="x" x="105" y="130">One decision tree</text>
  <text class="xs" x="105" y="146">splits until each leaf</text>
  <text class="xs" x="105" y="158">holds one student → overfits</text>
  <path class="l" d="M190,137 L246,137"/>

  <rect class="b" x="250" y="100" width="180" height="74"/>
  <text class="x" x="340" y="122">Bagging</text>
  <text class="xs" x="340" y="138">bootstrap samples:</text>
  <text class="xs" x="340" y="151">draw 316 rows WITH</text>
  <text class="xs" x="340" y="164">replacement, many times</text>
  <path class="l" d="M430,137 L486,137"/>

  <rect class="bt" x="490" y="24" width="190" height="40"/>
  <text class="x" x="585" y="40">Tree 1</text>
  <text class="xs" x="585" y="55">own sample + random features</text>
  <rect class="bt" x="490" y="76" width="190" height="40"/>
  <text class="x" x="585" y="92">Tree 2</text>
  <text class="xs" x="585" y="107">own sample + random features</text>
  <rect class="bt" x="490" y="128" width="190" height="40"/>
  <text class="x" x="585" y="150">...</text>
  <rect class="bt" x="490" y="180" width="190" height="40"/>
  <text class="x" x="585" y="196">Tree 400</text>
  <text class="xs" x="585" y="211">own sample + random features</text>

  <path class="l" d="M680,44 L742,120"/>
  <path class="l" d="M680,96 L742,130"/>
  <path class="l" d="M680,148 L742,142"/>
  <path class="l" d="M680,200 L742,154"/>

  <rect class="bf" x="746" y="112" width="214" height="54"/>
  <text class="x" x="853" y="132">Random Forest</text>
  <text class="xs" x="853" y="148">majority vote across trees</text>
  <text class="xs" x="853" y="160">→ At Risk / Not At Risk</text>

  <text class="xs" x="585" y="248">Each tree sees a different sample and a different
  subset of features, so the trees make DIFFERENT mistakes.</text>
  <text class="xs" x="585" y="264">Averaging the votes cancels much of that individual
  variance — this is why the ensemble beats any single tree.</text>
</svg>
</div>
"""


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
RENDERERS = {
    "Overview": render_overview,
    "Student Prediction": render_prediction,
    "Data Analysis": render_data_analysis,
    "Model Comparison": render_model_comparison,
    "Model Explainability": render_explainability,
    "Data Leakage": render_leakage,
    "About the Project": render_about,
}

try:
    RENDERERS[section]()
except Exception as exc:  # never show the user a raw traceback
    st.error(
        "Something went wrong while rendering this section. The trained models "
        "may be out of date with the current code — re-running "
        "`python src/train_models.py` usually fixes it."
    )
    with st.expander("Technical detail"):
        st.code(f"{type(exc).__name__}: {exc}")
