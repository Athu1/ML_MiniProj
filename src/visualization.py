"""
Plotly figure builders for the Streamlit application.

Every chart in the app is created here so that colour, typography and axis
treatment are defined once. No figure in this module invents data: each takes
values that came out of a trained model or out of the dataset itself.

Colour rules followed throughout
--------------------------------
* "At Risk" / "Not At Risk" are *states*, so they use a reserved status pair
  (green / red) and always carry a text label as well -- colour never carries
  the meaning on its own.
* Magnitude charts (histogram, confusion matrix, feature importance) use a
  single blue hue, light to dark. One series gets one colour; bars are never
  shaded darker-where-bigger, because that would encode bar length twice and
  waste the colour channel.
* The correlation heatmap is the one diverging scale: blue and red poles for
  opposite signs with a neutral grey at zero, so "no correlation" reads as
  nothing.
* Multi-model charts use the first three categorical slots, assigned to a fixed
  model order so a model keeps its colour across every chart in the app.
* Axis lines, gridlines and tick labels are deliberately recessive; text never
  takes the colour of the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config as cfg

# --------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXISLINE = "#c3c2b7"

# Categorical slots 1-3 (validated for all-pairs colour-vision separation).
# Pinned per model so Random Forest is the same colour in every chart.
MODEL_COLORS = {
    "Logistic Regression": "#2a78d6",   # blue
    "SVM (RBF)": "#eb6834",             # orange
    "Random Forest": "#1baf7a",         # aqua
}
SERIES_BLUE = "#2a78d6"
SERIES_VIOLET = "#4a3aa7"

# Reserved status colours -- used only for the at-risk state, never as a series.
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
STATUS_WARNING = "#fab219"

# Single-hue sequential ramp (blue, light to dark) for magnitude scales.
SEQ_BLUE = [
    [0.00, "#cde2fb"], [0.25, "#86b6ef"], [0.50, "#3987e5"],
    [0.75, "#256abf"], [1.00, "#0d366b"],
]
# Diverging: blue <-> red with a neutral grey midpoint.
DIVERGING = [
    [0.0, "#184f95"], [0.25, "#86b6ef"], [0.5, "#f0efec"],
    [0.75, "#e88c8c"], [1.0, "#a52222"],
]

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _base_layout(title: str, height: int = 380, **kwargs) -> dict:
    """Shared layout: recessive chrome, readable titles, hover enabled."""
    layout = dict(
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left"),
        height=height,
        margin=dict(l=60, r=24, t=52, b=52),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK_SECONDARY),
        hoverlabel=dict(font=dict(family=FONT, size=12), bgcolor="white",
                        bordercolor=AXISLINE),
        showlegend=False,
    )
    layout.update(kwargs)
    return layout


def _axis(title: str = "", **kwargs) -> dict:
    axis = dict(
        title=dict(text=title, font=dict(size=12, color=INK_SECONDARY)),
        showgrid=True, gridcolor=GRIDLINE, gridwidth=1,
        zeroline=False,
        linecolor=AXISLINE, linewidth=1, showline=True, ticks="outside",
        tickcolor=AXISLINE, tickfont=dict(size=11, color=INK_MUTED),
    )
    axis.update(kwargs)
    return axis


def _no_grid_axis(title: str = "", **kwargs) -> dict:
    return _axis(title, showgrid=False, **kwargs)


# ==========================================================================
# SECTION: dataset analysis
# ==========================================================================
def score_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of final grades, with the at-risk threshold marked.

    One series, so one colour and no legend. The annotation at grade 0 is the
    point of the chart: that bar is not a smooth tail, it is a separate group of
    students who recorded no final grade at all.
    """
    grades = df[cfg.REGRESSION_TARGET]
    counts = grades.value_counts().reindex(range(0, 21), fill_value=0)
    zero_n = int(counts.loc[0])

    fig = go.Figure(
        go.Bar(
            x=list(counts.index), y=counts.values,
            marker=dict(color=SERIES_BLUE, line=dict(width=2, color=SURFACE)),
            hovertemplate="Grade %{x}/20<br>%{y} students<extra></extra>",
        )
    )
    fig.add_vline(
        x=cfg.AT_RISK_THRESHOLD - 0.5, line=dict(color=STATUS_CRITICAL, width=2, dash="dash"),
        annotation_text=f"At-risk threshold (below {cfg.AT_RISK_THRESHOLD})",
        annotation_position="top right",
        annotation_font=dict(size=11, color=STATUS_CRITICAL),
    )
    if zero_n:
        fig.add_annotation(
            x=0, y=zero_n, text=f"{zero_n} students<br>recorded 0",
            showarrow=True, arrowhead=0, arrowcolor=INK_MUTED, ax=38, ay=-32,
            font=dict(size=11, color=INK_SECONDARY), align="left",
        )
    fig.update_layout(**_base_layout("Distribution of final grades (G3)"))
    fig.update_xaxes(**_axis("Final grade (0-20)", dtick=2, showgrid=False))
    fig.update_yaxes(**_axis("Number of students"))
    return fig


def class_distribution(df: pd.DataFrame) -> go.Figure:
    """At-risk vs not-at-risk counts.

    These are two states rather than two series, so they use the reserved
    status colours and are labelled directly on the bars -- the reader never has
    to decode a colour.
    """
    counts = df[cfg.CLASSIFICATION_TARGET].value_counts().sort_index()
    n = int(counts.sum())
    labels = [cfg.NEGATIVE_CLASS_LABEL, cfg.POSITIVE_CLASS_LABEL]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]

    fig = go.Figure(
        go.Bar(
            x=labels, y=values,
            marker=dict(color=[STATUS_GOOD, STATUS_CRITICAL],
                        line=dict(width=2, color=SURFACE)),
            text=[f"{v}  ({v / n * 100:.1f}%)" for v in values],
            textposition="outside",
            textfont=dict(size=12, color=INK_SECONDARY),
            hovertemplate="%{x}<br>%{y} students<extra></extra>",
            width=0.5,
        )
    )
    fig.update_layout(**_base_layout(
        f"Class balance: {values[1]} of {n} students are at risk", height=360
    ))
    fig.update_xaxes(**_no_grid_axis(""))
    fig.update_yaxes(**_axis("Number of students", range=[0, max(values) * 1.18]))
    return fig


def feature_vs_score_box(df: pd.DataFrame, feature: str) -> go.Figure:
    """Box plot of final grade across the levels of one ordinal/categorical feature.

    A box plot rather than a scatter plot: features such as `studytime` and
    `failures` take only three to five distinct values, so a scatter would draw
    a few vertical stripes of heavily overlapping points. The box shows the
    median and spread per level, which is the comparison that matters.
    """
    label = cfg.FEATURE_LABELS.get(feature, feature)
    scale = cfg.ORDINAL_SCALES.get(feature, {})
    cat_labels = cfg.CATEGORICAL_LABELS.get(feature, {})

    levels = sorted(df[feature].unique())
    fig = go.Figure()
    for lvl in levels:
        sub = df.loc[df[feature] == lvl, cfg.REGRESSION_TARGET]
        tick = scale.get(lvl, cat_labels.get(lvl, str(lvl)))
        fig.add_trace(
            go.Box(
                y=sub, name=str(tick), boxmean=True,
                marker=dict(color=SERIES_BLUE, size=5,
                            line=dict(width=1, color=SURFACE)),
                line=dict(color=SERIES_BLUE, width=2),
                fillcolor="rgba(42,120,214,0.14)",
                hovertemplate=(f"{label}: {tick}<br>"
                               "Final grade: %{y}/20<extra></extra>"),
            )
        )
    fig.add_hline(
        y=cfg.AT_RISK_THRESHOLD, line=dict(color=STATUS_CRITICAL, width=2, dash="dash"),
        annotation_text="Pass mark", annotation_position="bottom right",
        annotation_font=dict(size=11, color=STATUS_CRITICAL),
    )
    fig.update_layout(**_base_layout(f"Final grade by {label.lower()}", height=400))
    fig.update_xaxes(**_no_grid_axis(label))
    fig.update_yaxes(**_axis("Final grade (0-20)", range=[-1, 21]))
    return fig


def absences_vs_score(df: pd.DataFrame) -> go.Figure:
    """Scatter of absences against final grade, with the fitted trend line.

    `absences` is a genuine count with a long tail (0 to 75), so a scatter is
    the right form here.

    The chart exists to show why the aggregate correlation (r = +0.03) must not
    be quoted on its own. Every student who recorded a final grade of 0 also
    recorded 0 absences -- they sit as a block in the bottom-left corner -- and
    that block drags the overall fit flat. Among students with a genuine
    recorded grade the correlation is r = -0.21, a sign reversal. Both are shown.
    """
    x = df["absences"].to_numpy(dtype=float)
    y = df[cfg.REGRESSION_TARGET].to_numpy(dtype=float)
    r = float(np.corrcoef(x, y)[0, 1])

    # The headline correlation is suppressed by the zero-grade group: every one
    # of those students records zero absences, which pulls the fit flat. The
    # correlation among students with a recorded grade is computed and shown
    # too, because quoting the aggregate alone would be misleading.
    graded = df[cfg.REGRESSION_TARGET] > 0
    r_graded = float(df.loc[graded, "absences"].corr(df.loc[graded, cfg.REGRESSION_TARGET]))
    n_zero = int((~graded).sum())

    slope, intercept = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])

    fig = go.Figure()
    # The zero-grade students are drawn in the reserved "critical" colour and
    # labelled, because they are the reason the two correlations differ.
    fig.add_trace(
        go.Scatter(
            x=x[graded], y=y[graded], mode="markers",
            name=f"Students with a recorded grade (r = {r_graded:+.3f})",
            marker=dict(size=8, color=SERIES_BLUE, opacity=0.6,
                        line=dict(width=2, color=SURFACE)),
            hovertemplate="%{x} absences<br>Final grade %{y}/20<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x[~graded], y=y[~graded], mode="markers",
            name=f"{n_zero} students recorded grade 0 AND 0 absences",
            marker=dict(size=9, color=STATUS_CRITICAL, opacity=0.75,
                        symbol="diamond", line=dict(width=2, color=SURFACE)),
            hovertemplate="0 absences<br>Final grade 0/20<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs, y=slope * xs + intercept, mode="lines",
            name=f"Fit over all students (r = {r:+.3f})",
            line=dict(color=INK_SECONDARY, width=2, dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.update_layout(**_base_layout(
        "Absences vs final grade: why the headline correlation is misleading",
        height=470, showlegend=True,
        margin=dict(l=60, r=24, t=52, b=104),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0,
                    font=dict(size=11, color=INK_SECONDARY)),
    ))
    fig.update_xaxes(**_axis("School absences (days)"))
    fig.update_yaxes(**_axis("Final grade (0-20)", range=[-1, 21]))
    return fig


def correlation_heatmap(df: pd.DataFrame, include_grades: bool = True) -> go.Figure:
    """Pearson correlation between the numeric features (and optionally grades).

    Diverging scale, because the sign of a correlation matters: blue for
    negative, red for positive, neutral grey at zero so "unrelated" reads as
    nothing. Identifier-like columns are not in NUMERIC_FEATURES, so nothing
    meaningless is plotted.
    """
    cols = list(cfg.NUMERIC_FEATURES)
    if include_grades:
        cols = cols + ["G1", "G2", "G3"]
    corr = df[cols].corr(numeric_only=True)

    fig = go.Figure(
        go.Heatmap(
            z=corr.values, x=cols, y=cols,
            colorscale=DIVERGING, zmid=0, zmin=-1, zmax=1,
            xgap=2, ygap=2,
            colorbar=dict(title=dict(text="Pearson r", side="right"),
                          thickness=14, tickfont=dict(size=10, color=INK_MUTED)),
            hovertemplate="%{y} vs %{x}<br>r = %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(
        "Correlation between numeric features and grades", height=560,
        margin=dict(l=96, r=24, t=52, b=96),
    ))
    fig.update_xaxes(**_no_grid_axis("", tickangle=-45,
                                     tickfont=dict(size=10, color=INK_MUTED)))
    fig.update_yaxes(**_no_grid_axis("", autorange="reversed",
                                     tickfont=dict(size=10, color=INK_MUTED)))
    return fig


# ==========================================================================
# SECTION: model comparison
# ==========================================================================
def metric_comparison(clf_metrics: dict, baseline: dict | None = None) -> go.Figure:
    """Grouped bars: one group per metric, one bar per model.

    Metrics on the x-axis rather than models, so the eye compares models within
    a metric -- which is the comparison the reader actually wants. The dashed
    line is the majority-class baseline accuracy: any bar below it in the
    Accuracy group is doing worse than a model that always answers
    "Not At Risk".
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    models = [m for m in cfg.CLASSIFICATION_MODEL_NAMES if m in clf_metrics]

    fig = go.Figure()
    for model in models:
        vals = [clf_metrics[model].get(m, np.nan) for m in metrics]
        fig.add_trace(
            go.Bar(
                name=model, x=metrics, y=vals,
                marker=dict(color=MODEL_COLORS[model],
                            line=dict(width=2, color=SURFACE)),
                text=[f"{v:.3f}" if v == v else "n/a" for v in vals],
                textposition="outside",
                textfont=dict(size=10, color=INK_SECONDARY),
                hovertemplate=f"{model}<br>%{{x}} = %{{y:.4f}}<extra></extra>",
            )
        )

    if baseline and baseline.get("Accuracy") is not None:
        # Drawn as a real trace rather than an annotated hline so it appears in
        # the legend. A floating annotation in the plot area collided with the
        # bar value labels.
        acc = float(baseline["Accuracy"])
        fig.add_trace(
            go.Scatter(
                x=metrics, y=[acc] * len(metrics), mode="lines",
                name=f"Majority-class baseline accuracy ({acc:.3f})",
                line=dict(color=INK_MUTED, width=2, dash="dot"),
                hovertemplate=f"Majority-class baseline = {acc:.4f}<extra></extra>",
            )
        )

    # The legend sits BELOW the plot: a horizontal legend above the plot area
    # overlapped the chart title.
    fig.update_layout(**_base_layout(
        "Classification performance on the held-out test set", height=480,
        barmode="group", bargap=0.26, bargroupgap=0.06,
        showlegend=True,
        margin=dict(l=60, r=24, t=52, b=96),
        legend=dict(orientation="h", yanchor="top", y=-0.14, x=0,
                    font=dict(size=11, color=INK_SECONDARY)),
    ))
    fig.update_xaxes(**_no_grid_axis(""))
    fig.update_yaxes(**_axis("Score (0 to 1)", range=[0, 1.12], dtick=0.2))
    return fig


def cv_stability(clf_results: dict) -> go.Figure:
    """Cross-validation F1 mean with the standard deviation as an error bar.

    This chart exists to stop the comparison table being over-read. The error
    bars overlap heavily, which is the visual form of the project's conclusion:
    on 316 training students the three classifiers are not reliably
    distinguishable.
    """
    models = [m for m in cfg.CLASSIFICATION_MODEL_NAMES if m in clf_results]
    means = [clf_results[m]["cv_f1_mean"] for m in models]
    stds = [clf_results[m]["cv_f1_std"] for m in models]

    fig = go.Figure(
        go.Bar(
            x=models, y=means,
            marker=dict(color=[MODEL_COLORS[m] for m in models],
                        line=dict(width=2, color=SURFACE)),
            error_y=dict(type="data", array=stds, visible=True,
                         color=INK_SECONDARY, thickness=2, width=8),
            text=[f"{m:.3f} ± {s:.3f}" for m, s in zip(means, stds)],
            textposition="outside",
            textfont=dict(size=11, color=INK_SECONDARY),
            hovertemplate="%{x}<br>CV F1 = %{y:.4f}<extra></extra>",
            width=0.5,
        )
    )
    fig.update_layout(**_base_layout(
        f"Cross-validation stability ({cfg.CV_FOLDS}-fold, training split only)",
        height=400,
    ))
    fig.update_xaxes(**_no_grid_axis(""))
    fig.update_yaxes(**_axis("F1 on the at-risk class", range=[0, 0.8], dtick=0.1))
    return fig


def confusion_matrix_figure(matrix, model_name: str) -> go.Figure:
    """Confusion matrix as a heatmap with every cell named in words.

    Each cell carries its count, its share of the test set, and the plain-English
    name of that outcome, because "false negative" is the number that matters
    here and a bare 2x2 grid of digits does not make that obvious.
    """
    m = np.asarray(matrix, dtype=int)
    total = int(m.sum())
    cell_names = [["True negative", "False positive"],
                  ["False negative", "True positive"]]
    cell_notes = [["correctly cleared", "false alarm"],
                  ["MISSED at-risk student", "correctly flagged"]]

    text = [
        [f"<b>{m[i][j]}</b><br>{cell_names[i][j]}<br>"
         f"<span style='font-size:10px'>{cell_notes[i][j]}</span>"
         for j in range(2)]
        for i in range(2)
    ]
    hover = [
        [f"Actual: {[cfg.NEGATIVE_CLASS_LABEL, cfg.POSITIVE_CLASS_LABEL][i]}<br>"
         f"Predicted: {[cfg.NEGATIVE_CLASS_LABEL, cfg.POSITIVE_CLASS_LABEL][j]}<br>"
         f"{m[i][j]} students ({m[i][j] / total * 100:.1f}% of the test set)<br>"
         f"{cell_names[i][j]}"
         for j in range(2)]
        for i in range(2)
    ]

    fig = go.Figure(
        go.Heatmap(
            z=m, x=[f"Predicted<br>{cfg.NEGATIVE_CLASS_LABEL}",
                    f"Predicted<br>{cfg.POSITIVE_CLASS_LABEL}"],
            y=[f"Actual<br>{cfg.NEGATIVE_CLASS_LABEL}",
               f"Actual<br>{cfg.POSITIVE_CLASS_LABEL}"],
            colorscale=SEQ_BLUE, showscale=False, xgap=2, ygap=2,
            text=text, texttemplate="%{text}",
            textfont=dict(size=13, color=INK),
            customdata=hover, hovertemplate="%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(
        f"{model_name} -- confusion matrix ({total} test students)", height=400,
        margin=dict(l=110, r=24, t=52, b=60),
    ))
    fig.update_xaxes(**_no_grid_axis("", side="bottom"))
    fig.update_yaxes(**_no_grid_axis("", autorange="reversed"))
    return fig


def roc_curves(clf_results: dict) -> go.Figure:
    """ROC curves for every classifier that can produce probabilities."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random guess (0.500)",
            line=dict(color=INK_MUTED, width=2, dash="dot"), hoverinfo="skip",
        )
    )
    for model in cfg.CLASSIFICATION_MODEL_NAMES:
        res = clf_results.get(model)
        if not res or not res.get("roc"):
            continue
        roc = res["roc"]
        fig.add_trace(
            go.Scatter(
                x=roc["fpr"], y=roc["tpr"], mode="lines",
                name=f"{model} (AUC {roc['auc']:.3f})",
                line=dict(color=MODEL_COLORS[model], width=2),
                hovertemplate=(f"{model}<br>False positive rate %{{x:.3f}}"
                               "<br>True positive rate %{y:.3f}<extra></extra>"),
            )
        )
    fig.update_layout(**_base_layout(
        "ROC curves -- ability to rank at-risk students above the rest", height=440,
        showlegend=True,
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98,
                    font=dict(size=11, color=INK_SECONDARY),
                    bgcolor="rgba(252,252,251,0.85)", bordercolor=GRIDLINE,
                    borderwidth=1),
    ))
    fig.update_xaxes(**_axis("False positive rate", range=[-0.02, 1.02], dtick=0.2))
    fig.update_yaxes(**_axis("True positive rate (recall)", range=[-0.02, 1.02],
                             dtick=0.2, scaleanchor="x", scaleratio=1))
    return fig


# ==========================================================================
# SECTION: regression diagnostics
# ==========================================================================
def actual_vs_predicted(y_true, y_pred, r2: float) -> go.Figure:
    """Actual against predicted grade, with the ideal y = x reference line.

    A model with no error would put every point on the dashed line. The visible
    pattern here -- predictions clustered in a narrow band around 10 while the
    actual grades spread across the whole 0-20 range -- is what a low R-squared
    looks like.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 20], y=[0, 20], mode="lines", name="Perfect prediction",
            line=dict(color=INK_MUTED, width=2, dash="dash"), hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_true, y=y_pred, mode="markers", name="Test students",
            marker=dict(size=9, color=SERIES_BLUE, opacity=0.68,
                        line=dict(width=2, color=SURFACE)),
            hovertemplate=("Actual %{x:.0f}/20<br>Predicted %{y:.2f}/20"
                           "<br>Error %{customdata:+.2f}<extra></extra>"),
            customdata=y_pred - y_true,
        )
    )
    fig.update_layout(**_base_layout(
        f"Linear Regression: actual vs predicted grade (test R² = {r2:.3f})",
        height=440, showlegend=True,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                    font=dict(size=11, color=INK_SECONDARY),
                    bgcolor="rgba(252,252,251,0.85)", bordercolor=GRIDLINE,
                    borderwidth=1),
    ))
    fig.update_xaxes(**_axis("Actual final grade (0-20)", range=[-1, 21], dtick=5))
    fig.update_yaxes(**_axis("Predicted final grade (0-20)", range=[-1, 21], dtick=5,
                             scaleanchor="x", scaleratio=1))
    return fig


def residual_plot(y_true, y_pred) -> go.Figure:
    """Residuals (actual - predicted) against the predicted value.

    Residuals should scatter randomly around zero. The downward band of points
    at the bottom belongs to the students who recorded a final grade of 0: the
    model predicts a normal pass for them and is wrong by ten points or more.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred

    fig = go.Figure(
        go.Scatter(
            x=y_pred, y=resid, mode="markers",
            marker=dict(size=9, color=SERIES_VIOLET, opacity=0.68,
                        line=dict(width=2, color=SURFACE)),
            hovertemplate=("Predicted %{x:.2f}/20<br>Residual %{y:+.2f}"
                           "<extra></extra>"),
        )
    )
    fig.add_hline(y=0, line=dict(color=INK_SECONDARY, width=2, dash="dash"))
    fig.update_layout(**_base_layout(
        "Residual plot -- residual = actual grade minus predicted grade", height=400
    ))
    fig.update_xaxes(**_axis("Predicted final grade (0-20)"))
    fig.update_yaxes(**_axis("Residual (grade points)"))
    return fig


# ==========================================================================
# SECTION: explainability
# ==========================================================================
def feature_importance_bars(
    importance_df: pd.DataFrame, value_col: str, title: str,
    x_title: str, top_n: int = 15, std_col: str | None = None,
) -> go.Figure:
    """Horizontal bars, sorted by importance, largest at the top.

    One series, so one colour for every bar. The bars are deliberately *not*
    shaded darker-where-larger: bar length already encodes the magnitude, and
    re-encoding it as colour would waste the channel and mislead.
    """
    top = importance_df.nlargest(top_n, value_col).iloc[::-1]
    labels = [cfg.FEATURE_LABELS.get(f, f) for f in top["feature"]]

    err = None
    if std_col and std_col in top.columns:
        err = dict(type="data", array=top[std_col].to_numpy(), visible=True,
                   color=INK_SECONDARY, thickness=1.5, width=4)

    fig = go.Figure(
        go.Bar(
            x=top[value_col], y=labels, orientation="h",
            marker=dict(color=SERIES_BLUE, line=dict(width=2, color=SURFACE)),
            error_x=err,
            hovertemplate="%{y}<br>" + x_title + " = %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(
        title, height=max(360, 26 * len(top) + 110),
        margin=dict(l=210, r=48, t=52, b=52),
    ))
    fig.update_xaxes(**_axis(x_title))
    fig.update_yaxes(**_no_grid_axis("", tickfont=dict(size=11, color=INK_SECONDARY)))
    return fig


def logistic_coefficient_bars(coef_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Signed Logistic Regression coefficients.

    A diverging colour scale is correct here because the *sign* is the message:
    red for coefficients associated with higher predicted risk, blue for lower,
    and the zero line reads as "no association". Values are on the standardised
    feature scale, so they are comparable with each other.
    """
    top = coef_df.nlargest(top_n, "abs_coefficient").iloc[::-1]
    labels = [cfg.FEATURE_LABELS.get(f, f) for f in top["feature"]]
    colors = [STATUS_CRITICAL if c >= 0 else SERIES_BLUE for c in top["coefficient"]]

    fig = go.Figure(
        go.Bar(
            x=top["coefficient"], y=labels, orientation="h",
            marker=dict(color=colors, line=dict(width=2, color=SURFACE)),
            customdata=np.stack([top["odds_ratio"], top["direction"]], axis=-1),
            hovertemplate=("%{y}<br>coefficient %{x:+.4f}"
                           "<br>odds ratio %{customdata[0]:.3f}"
                           "<br>%{customdata[1]}<extra></extra>"),
        )
    )
    fig.add_vline(x=0, line=dict(color=AXISLINE, width=2))
    fig.update_layout(**_base_layout(
        "Logistic Regression coefficients (standardised scale)",
        height=max(360, 26 * len(top) + 110),
        margin=dict(l=210, r=48, t=52, b=52),
    ))
    fig.update_xaxes(**_axis("Change in log-odds of being at risk  "
                            "(right = higher risk)"))
    fig.update_yaxes(**_no_grid_axis("", tickfont=dict(size=11, color=INK_SECONDARY)))
    return fig


# ==========================================================================
# SECTION: single-student prediction
# ==========================================================================
def score_gauge(predicted_score: float) -> go.Figure:
    """Predicted grade as a gauge against the 0-20 scale.

    The red band below the pass mark and the threshold needle let the number be
    read in context: 9.4/20 means little until you can see it sits under 10.
    """
    score = float(np.clip(predicted_score, 0, cfg.GRADE_MAX))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(suffix=f" / {cfg.GRADE_MAX}", font=dict(size=40, color=INK)),
            gauge=dict(
                axis=dict(range=[0, cfg.GRADE_MAX], tickwidth=1, tickcolor=AXISLINE,
                          tickfont=dict(size=10, color=INK_MUTED), dtick=5),
                bar=dict(color=SERIES_BLUE, thickness=0.62),
                bgcolor=SURFACE, borderwidth=1, bordercolor=GRIDLINE,
                steps=[
                    dict(range=[0, cfg.AT_RISK_THRESHOLD], color="#fbeaea"),
                    dict(range=[cfg.AT_RISK_THRESHOLD, cfg.GRADE_MAX], color="#eaf4ea"),
                ],
                threshold=dict(line=dict(color=STATUS_CRITICAL, width=3),
                               thickness=0.85, value=cfg.AT_RISK_THRESHOLD),
            ),
        )
    )
    fig.update_layout(**_base_layout("Predicted final grade", height=300,
                                     margin=dict(l=36, r=36, t=52, b=16)))
    return fig


def risk_probability_bar(probability: float, threshold: float = 0.5) -> go.Figure:
    """Risk probability as a labelled meter with the decision threshold marked.

    The meter's fill colour carries the severity and the percentage is printed,
    so the state is never communicated by colour alone.
    """
    pct = float(np.clip(probability, 0, 1)) * 100
    color = STATUS_CRITICAL if probability >= threshold else STATUS_GOOD

    fig = go.Figure()
    # The bar is deliberately less than full height so the threshold label has
    # clear space above it instead of printing over the fill.
    fig.add_trace(
        go.Bar(
            x=[100], y=["risk"], orientation="h", width=0.42,
            marker=dict(color="#f0efec"), hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Bar(
            x=[pct], y=["risk"], orientation="h", width=0.42,
            marker=dict(color=color, line=dict(width=2, color=SURFACE)),
            hovertemplate=f"Estimated probability of being at risk: {pct:.1f}%"
                          "<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(
        x=threshold * 100, line=dict(color=INK_SECONDARY, width=2, dash="dash"),
    )
    fig.add_annotation(
        x=threshold * 100, y=1.0, yref="paper", yanchor="bottom",
        text=f"Decision threshold ({threshold:.0%})", showarrow=False,
        font=dict(size=11, color=INK_SECONDARY), xanchor="center", yshift=2,
    )
    fig.add_annotation(
        x=1, y="risk", text=f"<b>{pct:.1f}%</b>", showarrow=False,
        xanchor="left", font=dict(size=14, color=INK), xshift=6,
    )
    fig.update_layout(**_base_layout(
        "Estimated probability of being at risk", height=230, barmode="overlay",
        margin=dict(l=12, r=70, t=74, b=46),
    ))
    fig.update_xaxes(**_no_grid_axis("Probability (%)", range=[0, 108], dtick=20))
    fig.update_yaxes(**_no_grid_axis("", showticklabels=False, showline=False,
                                     ticks=""))
    return fig


def student_profile_bars(profile: dict, df: pd.DataFrame) -> go.Figure:
    """Where this student sits on each ordinal scale, as a percentile-free bar.

    Each bar is the student's value rescaled to 0-100% of that feature's own
    observed range, so the chart answers "high or low for this cohort?". It is
    NOT a statement about which features mattered to the prediction -- the
    importance charts are the place for that, and the axis title says so.
    """
    features = ["studytime", "failures", "absences", "goout", "famrel",
                "freetime", "health", "Dalc", "Walc", "traveltime"]
    features = [f for f in features if f in profile]

    labels, pcts, raws, ranges = [], [], [], []
    for f in features:
        lo, hi = float(df[f].min()), float(df[f].max())
        val = float(profile[f])
        pct = 0.0 if hi == lo else (val - lo) / (hi - lo) * 100
        labels.append(cfg.FEATURE_LABELS.get(f, f))
        pcts.append(pct)
        raws.append(val)
        ranges.append(f"{lo:g} to {hi:g}")

    order = np.argsort(pcts)
    fig = go.Figure(
        go.Bar(
            x=[pcts[i] for i in order],
            y=[labels[i] for i in order],
            orientation="h",
            marker=dict(color=SERIES_BLUE, line=dict(width=2, color=SURFACE)),
            customdata=[[raws[i], ranges[i]] for i in order],
            text=[f"{raws[i]:g}" for i in order],
            textposition="outside",
            textfont=dict(size=11, color=INK_SECONDARY),
            hovertemplate=("%{y}<br>This student: %{customdata[0]:g}"
                           "<br>Dataset range: %{customdata[1]}"
                           "<br>%{x:.0f}% of that range<extra></extra>"),
        )
    )
    fig.update_layout(**_base_layout(
        "Student input profile", height=max(340, 28 * len(features) + 110),
        margin=dict(l=210, r=56, t=52, b=56),
    ))
    fig.update_xaxes(**_axis("Position within this feature's range in the dataset (%)",
                            range=[0, 112], dtick=25))
    fig.update_yaxes(**_no_grid_axis("", tickfont=dict(size=11, color=INK_SECONDARY)))
    return fig


def leakage_comparison_chart(leakage_rows: list[dict]) -> go.Figure:
    """Side-by-side metrics with and without the intermediate grades G1/G2.

    Two conditions rather than two series, so they are labelled directly. The
    gap between the pairs is the whole point of the chart: it is what target
    leakage buys you on paper and why the project refuses to use it.
    """
    rows = [r for r in leakage_rows if r["Metric"] in ("F1", "ROC-AUC", "R2")]
    names = [f"{r['Model'].replace(' (RBF)', '')}<br>"
             f"{r['Metric'].replace('R2', 'R²')}" for r in rows]
    clean = [r["Leakage-free (no G1/G2)"] for r in rows]
    leaky = [r["With leakage (G1+G2)"] for r in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Leakage-free (G1/G2 excluded) -- what this project reports",
            x=names, y=clean,
            marker=dict(color=SERIES_BLUE, line=dict(width=2, color=SURFACE)),
            text=[f"{v:.3f}" for v in clean], textposition="outside",
            textfont=dict(size=10, color=INK_SECONDARY),
            hovertemplate="%{x}<br>Leakage-free: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="With leakage (G1/G2 included) -- misleadingly good",
            x=names, y=leaky,
            marker=dict(color=STATUS_WARNING, line=dict(width=2, color=SURFACE)),
            text=[f"{v:.3f}" for v in leaky], textposition="outside",
            textfont=dict(size=10, color=INK_SECONDARY),
            hovertemplate="%{x}<br>With leakage: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(
        "What target leakage does to the reported scores", height=520,
        barmode="group", bargap=0.3, bargroupgap=0.08, showlegend=True,
        margin=dict(l=60, r=24, t=52, b=132),
        legend=dict(orientation="h", yanchor="top", y=-0.20, x=0,
                    font=dict(size=11, color=INK_SECONDARY)),
    ))
    fig.update_xaxes(**_no_grid_axis("", tickfont=dict(size=10, color=INK_MUTED)))
    fig.update_yaxes(**_axis("Score", range=[0, 1.14], dtick=0.2))
    return fig


# ==========================================================================
# SECTION: decision threshold and calibration
# ==========================================================================
def threshold_tradeoff(curve: dict, default_threshold: float,
                       tuned_threshold: float, model_name: str) -> go.Figure:
    """Precision, recall and F1 across every possible decision threshold.

    The point of the chart is that 0.5 is not a special place on this axis. Two
    vertical rules mark the default cut-off and the cost-tuned one, so the
    reader can see exactly what moving it buys and costs.

    Three series, so the first three categorical slots are used and every series
    is named in the legend -- identity never rests on colour alone.
    """
    fig = go.Figure()
    for key, label, color in (
        ("precision", "Precision", "#2a78d6"),
        ("recall", "Recall (at-risk students found)", "#eb6834"),
        ("f1", "F1", "#1baf7a"),
    ):
        fig.add_trace(
            go.Scatter(
                x=curve["threshold"], y=curve[key], mode="lines", name=label,
                line=dict(color=color, width=2),
                hovertemplate=f"{label}<br>threshold %{{x:.2f}} → "
                              "%{y:.3f}<extra></extra>",
            )
        )

    for x, text, dash in (
        (default_threshold, f"default {default_threshold:.2f}", "dot"),
        (tuned_threshold, f"cost-tuned {tuned_threshold:.2f}", "dash"),
    ):
        fig.add_vline(x=x, line=dict(color=INK_SECONDARY, width=2, dash=dash))
        fig.add_annotation(
            x=x, y=1.0, yref="paper", yanchor="bottom", text=text,
            showarrow=False, font=dict(size=11, color=INK_SECONDARY),
            xanchor="center", yshift=2,
        )

    fig.update_layout(**_base_layout(
        f"{model_name}: what the decision threshold costs and buys", height=460,
        showlegend=True, margin=dict(l=60, r=24, t=78, b=96),
        legend=dict(orientation="h", yanchor="top", y=-0.16, x=0,
                    font=dict(size=11, color=INK_SECONDARY)),
    ))
    fig.update_xaxes(**_axis("Decision threshold — P(at risk) above which a "
                            "student is flagged", range=[0, 1], dtick=0.1))
    fig.update_yaxes(**_axis("Score (0 to 1)", range=[0, 1.02], dtick=0.2))
    return fig


def calibration_chart(calibrations: dict) -> go.Figure:
    """Predicted probability against observed failure rate, per model.

    A perfectly calibrated model sits on the diagonal: of the students it calls
    40% risk, 40% actually fail. Points below the diagonal mean the model is
    over-confident (it claims more risk than materialises); above means
    under-confident.

    Bins hold an equal NUMBER of students rather than an equal slice of the
    probability axis, because these probabilities are compressed and an evenly
    sliced axis would leave most bins empty.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
            line=dict(color=INK_MUTED, width=2, dash="dot"), hoverinfo="skip",
        )
    )
    for model in cfg.CLASSIFICATION_MODEL_NAMES:
        cal = calibrations.get(model)
        if not cal:
            continue
        fig.add_trace(
            go.Scatter(
                x=cal["mean_predicted"], y=cal["observed_frequency"],
                mode="lines+markers",
                name=f"{model} (Brier {cal['brier_score']:.3f})",
                line=dict(color=MODEL_COLORS[model], width=2),
                marker=dict(size=9, line=dict(width=2, color=SURFACE)),
                hovertemplate=(f"{model}<br>predicted %{{x:.3f}}"
                               "<br>actually failed %{y:.3f}<extra></extra>"),
            )
        )
    fig.update_layout(**_base_layout(
        "Are the predicted probabilities real? (test set, equal-count bins)",
        height=470, showlegend=True, margin=dict(l=60, r=24, t=52, b=104),
        legend=dict(orientation="h", yanchor="top", y=-0.17, x=0,
                    font=dict(size=11, color=INK_SECONDARY)),
    ))
    # No `scaleanchor` here: forcing a 1:1 aspect in a container wider than it
    # is tall made Plotly extend the axes past [0, 1], which is nonsense on a
    # probability scale. The dashed reference line carries the comparison
    # instead, and both axes stay honestly bounded.
    fig.update_xaxes(**_axis("Mean predicted P(at risk)", range=[0, 1], dtick=0.2))
    fig.update_yaxes(**_axis("Observed share who actually failed", range=[0, 1],
                             dtick=0.2))
    return fig


def probability_spread(calibrations: dict) -> go.Figure:
    """The observed range of each model's predicted probabilities.

    This chart exists because the application displays these numbers as
    percentages, which implies the full 0-100% scale is available. It is not:
    each bar shows the span a model actually produces, against the full scale in
    grey. A short bar means the model never expresses confidence, so its output
    is a relative ranking rather than a probability.
    """
    models = [m for m in cfg.CLASSIFICATION_MODEL_NAMES if m in calibrations]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[100] * len(models), y=models, orientation="h", width=0.46,
            marker=dict(color="#f0efec"), hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Bar(
            x=[(calibrations[m]["proba_range"]) * 100 for m in models],
            base=[calibrations[m]["proba_min"] * 100 for m in models],
            y=models, orientation="h", width=0.46,
            marker=dict(color=[MODEL_COLORS[m] for m in models],
                        line=dict(width=2, color=SURFACE)),
            text=[f"{calibrations[m]['proba_min']:.0%} – "
                  f"{calibrations[m]['proba_max']:.0%}" for m in models],
            textposition="outside",
            textfont=dict(size=11, color=INK_SECONDARY),
            hovertemplate="%{y}<br>observed range %{text}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(x=cfg.DEFAULT_THRESHOLD * 100,
                  line=dict(color=INK_SECONDARY, width=2, dash="dash"))
    fig.add_annotation(
        x=cfg.DEFAULT_THRESHOLD * 100, y=1.0, yref="paper", yanchor="bottom",
        text="default cut-off (50%)", showarrow=False, xanchor="center",
        font=dict(size=11, color=INK_SECONDARY), yshift=2,
    )
    fig.update_layout(**_base_layout(
        "The range of risk percentages each model actually produces",
        height=320, barmode="overlay", margin=dict(l=150, r=96, t=74, b=52),
    ))
    fig.update_xaxes(**_no_grid_axis("P(at risk) (%)", range=[0, 112], dtick=20))
    fig.update_yaxes(**_no_grid_axis("", tickfont=dict(size=11, color=INK_SECONDARY)))
    return fig
