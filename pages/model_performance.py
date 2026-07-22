import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


st.set_page_config(
    page_title="Model Performance",
    layout="wide"
)


# ==========================================
# PAGE TITLE
# ==========================================

st.title("Model Performance Evaluation")

st.markdown(
    """
Evaluate the reliability of the historical SARIMAX forecasts using:

- Mean Absolute Error
- Root Mean Squared Error
- Mean Absolute Percentage Error
- Forecast bias
- Residual behaviour
- Ward-level performance comparison
"""
)


# ==========================================
# VALIDATE SESSION STATE
# ==========================================

required_items = [
    "sarimax_short",
    "sarimax_long",
    "hospital",
    "ward"
]

missing_items = [
    item
    for item in required_items
    if item not in st.session_state
]

if missing_items:

    st.error(
        "Required model-evaluation data is unavailable. "
        "Please open the main dashboard first."
    )

    st.stop()


# ==========================================
# LOAD DATA
# ==========================================

sarimax_short = (
    st.session_state["sarimax_short"]
    .copy()
)

sarimax_long = (
    st.session_state["sarimax_long"]
    .copy()
)

hospital = st.session_state["hospital"]

ward = st.session_state["ward"]


# ==========================================
# PREPARE DATASETS
# ==========================================

evaluation_datasets = {
    "SARIMAX 24–72 Hours": sarimax_short,
    "SARIMAX 7–30 Days": sarimax_long
}


for dataset in evaluation_datasets.values():

    if "date" in dataset.columns:

        dataset["date"] = pd.to_datetime(
            dataset["date"],
            errors="coerce"
        )


    numeric_columns = [
        "actual_occupied_beds",
        "predicted_occupied_beds"
    ]


    for column in numeric_columns:

        if column in dataset.columns:

            dataset[column] = pd.to_numeric(
                dataset[column],
                errors="coerce"
            )


# ==========================================
# SIDEBAR CONTROLS
# ==========================================

st.sidebar.header("Evaluation Controls")

model_choice = st.sidebar.selectbox(
    "Backtesting Dataset",
    list(evaluation_datasets.keys())
)

st.sidebar.caption(
    """
Only historical backtesting datasets are evaluated here because
future forecasts do not yet have actual occupancy values.
"""
)


analysis_scope = st.sidebar.radio(
    "Analysis Scope",
    [
        "Selected Ward",
        "All Wards"
    ]
)


selected_dataset = (
    evaluation_datasets[
        model_choice
    ].copy()
)


# ==========================================
# FILTER SELECTED WARD
# ==========================================

ward_df = selected_dataset[
    (
        selected_dataset["hospital_id"]
        == hospital
    )
    &
    (
        selected_dataset["ward"]
        == ward
    )
].copy()


required_columns = [
    "actual_occupied_beds",
    "predicted_occupied_beds"
]


missing_columns = [
    column
    for column in required_columns
    if column not in selected_dataset.columns
]


if missing_columns:

    st.error(
        "The evaluation dataset is missing: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ==========================================
# METRIC FUNCTIONS
# ==========================================

def calculate_safe_mape(
    actual_values: pd.Series,
    predicted_values: pd.Series
) -> float:
    """
    Calculate MAPE while excluding actual values
    equal to or close to zero.
    """

    valid_mask = (
        actual_values.abs() > 1e-8
    )

    if not valid_mask.any():

        return np.nan

    percentage_errors = (
        (
            actual_values[valid_mask]
            -
            predicted_values[valid_mask]
        ).abs()
        /
        actual_values[valid_mask].abs()
    )

    return float(
        percentage_errors.mean() * 100
    )


def calculate_metrics(
    dataframe: pd.DataFrame
) -> dict | None:
    """
    Calculate forecast performance metrics for
    a prepared evaluation dataframe.
    """

    valid_df = dataframe.dropna(
        subset=[
            "actual_occupied_beds",
            "predicted_occupied_beds"
        ]
    ).copy()


    if valid_df.empty:

        return None


    actual_values = (
        valid_df["actual_occupied_beds"]
    )

    predicted_values = (
        valid_df["predicted_occupied_beds"]
    )


    residuals = (
        predicted_values
        -
        actual_values
    )


    mae_value = mean_absolute_error(
        actual_values,
        predicted_values
    )


    rmse_value = np.sqrt(
        mean_squared_error(
            actual_values,
            predicted_values
        )
    )


    mape_value = calculate_safe_mape(
        actual_values,
        predicted_values
    )


    bias_value = residuals.mean()


    maximum_error = (
        residuals.abs().max()
    )


    return {
        "MAE": float(mae_value),
        "RMSE": float(rmse_value),
        "MAPE": float(mape_value)
        if pd.notna(mape_value)
        else np.nan,
        "Bias": float(bias_value),
        "Maximum Error": float(maximum_error),
        "Records": len(valid_df)
    }


# ==========================================
# BUILD WARD-LEVEL LEADERBOARD
# ==========================================

leaderboard_records = []


for (
    hospital_id,
    ward_name
), group in selected_dataset.groupby(
    [
        "hospital_id",
        "ward"
    ]
):

    group_metrics = calculate_metrics(
        group
    )


    if group_metrics is None:

        continue


    leaderboard_records.append(
        {
            "hospital_id": hospital_id,
            "ward": ward_name,
            **group_metrics
        }
    )


leaderboard_df = pd.DataFrame(
    leaderboard_records
)


if not leaderboard_df.empty:

    leaderboard_df = (
        leaderboard_df
        .sort_values(
            [
                "MAE",
                "RMSE"
            ],
            ascending=True
        )
        .reset_index(drop=True)
    )


    leaderboard_df["Rank"] = (
        leaderboard_df.index + 1
    )


# ==========================================
# SELECT ANALYSIS DATA
# ==========================================

if analysis_scope == "Selected Ward":

    analysis_df = ward_df.copy()

    analysis_title = (
        f"{hospital} — {ward}"
    )

else:

    analysis_df = selected_dataset.copy()

    analysis_title = "All Hospitals and Wards"


analysis_df = (
    analysis_df
    .dropna(
        subset=[
            "actual_occupied_beds",
            "predicted_occupied_beds"
        ]
    )
    .sort_values("date")
    .reset_index(drop=True)
)


if analysis_df.empty:

    st.warning(
        "No valid model-evaluation records are available "
        "for this selection."
    )

    st.stop()


analysis_df["Residual"] = (
    analysis_df["predicted_occupied_beds"]
    -
    analysis_df["actual_occupied_beds"]
)


analysis_df["Absolute Error"] = (
    analysis_df["Residual"].abs()
)


analysis_df["Squared Error"] = (
    analysis_df["Residual"] ** 2
)


# ==========================================
# CALCULATE SELECTED METRICS
# ==========================================

metrics_result = calculate_metrics(
    analysis_df
)


if metrics_result is None:

    st.warning(
        "Unable to calculate model metrics "
        "for the selected data."
    )

    st.stop()


mae = metrics_result["MAE"]

rmse = metrics_result["RMSE"]

mape = metrics_result["MAPE"]

bias = metrics_result["Bias"]

max_error = metrics_result[
    "Maximum Error"
]


# ==========================================
# CONTEXT
# ==========================================

st.caption(
    f"Dataset: **{model_choice}** | "
    f"Scope: **{analysis_title}**"
)


# ==========================================
# PERFORMANCE SCORECARDS
# ==========================================

st.subheader("Performance Summary")


c1, c2, c3, c4, c5, c6 = (
    st.columns(6)
)


c1.metric(
    "MAE",
    f"{mae:.2f}",
    help=(
        "Average absolute difference between "
        "actual and predicted occupied beds."
    )
)


c2.metric(
    "RMSE",
    f"{rmse:.2f}",
    help=(
        "Penalises larger forecasting errors "
        "more strongly than MAE."
    )
)


if pd.notna(mape):

    c3.metric(
        "MAPE",
        f"{mape:.2f}%"
    )

else:

    c3.metric(
        "MAPE",
        "Unavailable",
        help=(
            "MAPE cannot be calculated when all "
            "actual values are zero."
        )
    )


c4.metric(
    "Bias",
    f"{bias:.2f}",
    help=(
        "Positive bias means overprediction. "
        "Negative bias means underprediction."
    )
)


c5.metric(
    "Maximum Error",
    f"{max_error:.2f}"
)


c6.metric(
    "Evaluation Records",
    len(analysis_df)
)


# ==========================================
# PERFORMANCE INTERPRETATION
# ==========================================

st.subheader("Model Interpretation")


if mae <= 1:

    accuracy_message = (
        "🟢 Strong accuracy: the average forecast "
        "error is approximately one occupied bed or less."
    )

elif mae <= 3:

    accuracy_message = (
        "🟡 Moderate accuracy: the model is generally "
        "close to actual demand but may require local refinement."
    )

else:

    accuracy_message = (
        "🔴 Material error: the model should be reviewed "
        "before operational decisions rely heavily on it."
    )


st.info(accuracy_message)


if bias > 0.5:

    st.warning(
        "The model has an overprediction tendency. "
        "Forecast demand is generally higher than actual demand."
    )

elif bias < -0.5:

    st.warning(
        "The model has an underprediction tendency. "
        "Actual demand is generally higher than forecast demand."
    )

else:

    st.success(
        "The model shows limited systematic bias."
    )


# ==========================================
# ACTUAL VS FORECAST
# ==========================================

st.subheader("Actual vs Forecast")


actual_fig = px.line(
    analysis_df,
    x="date",
    y=[
        "actual_occupied_beds",
        "predicted_occupied_beds"
    ],
    markers=True,
    title="Actual and Predicted Occupied Beds"
)


actual_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Occupied Beds",
    legend_title_text=""
)


st.plotly_chart(
    actual_fig,
    use_container_width=True
)


# ==========================================
# RESIDUAL TREND
# ==========================================

st.subheader("Residual Trend")


residual_fig = px.line(
    analysis_df,
    x="date",
    y="Residual",
    markers=True,
    title="Forecast Residuals Over Time"
)


residual_fig.add_hline(
    y=0,
    line_dash="dash",
    annotation_text="No Error"
)


residual_fig.update_layout(
    xaxis_title="Date",
    yaxis_title=(
        "Residual "
        "(Predicted − Actual)"
    )
)


st.plotly_chart(
    residual_fig,
    use_container_width=True
)


# ==========================================
# RESIDUAL DISTRIBUTION
# ==========================================

st.subheader("Residual Distribution")


histogram_fig = px.histogram(
    analysis_df,
    x="Residual",
    nbins=20,
    title="Distribution of Forecast Errors"
)


histogram_fig.update_layout(
    xaxis_title="Residual",
    yaxis_title="Frequency"
)


st.plotly_chart(
    histogram_fig,
    use_container_width=True
)


# ==========================================
# ACTUAL VS PREDICTED SCATTER
# ==========================================

st.subheader(
    "Actual vs Predicted Relationship"
)


scatter_fig = px.scatter(
    analysis_df,
    x="actual_occupied_beds",
    y="predicted_occupied_beds",
    trendline="ols",
    title="Actual vs Predicted Occupied Beds"
)


minimum_value = min(
    analysis_df[
        "actual_occupied_beds"
    ].min(),
    analysis_df[
        "predicted_occupied_beds"
    ].min()
)


maximum_value = max(
    analysis_df[
        "actual_occupied_beds"
    ].max(),
    analysis_df[
        "predicted_occupied_beds"
    ].max()
)


scatter_fig.add_shape(
    type="line",
    x0=minimum_value,
    y0=minimum_value,
    x1=maximum_value,
    y1=maximum_value,
    line=dict(
        dash="dash"
    )
)


scatter_fig.update_layout(
    xaxis_title="Actual Occupied Beds",
    yaxis_title="Predicted Occupied Beds"
)


st.plotly_chart(
    scatter_fig,
    use_container_width=True
)


# ==========================================
# ABSOLUTE ERROR TREND
# ==========================================

st.subheader("Absolute Error by Date")


error_fig = px.bar(
    analysis_df,
    x="date",
    y="Absolute Error",
    title="Absolute Forecast Error"
)


error_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Absolute Error in Beds"
)


st.plotly_chart(
    error_fig,
    use_container_width=True
)


# ==========================================
# WARD PERFORMANCE LEADERBOARD
# ==========================================

st.subheader("Ward Performance Leaderboard")


if not leaderboard_df.empty:

    best_ward = leaderboard_df.iloc[0]


    worst_ward = leaderboard_df.iloc[-1]


    l1, l2, l3, l4 = st.columns(4)


    l1.metric(
        "Best Performing Ward",
        best_ward["ward"]
    )


    l2.metric(
        "Best Ward MAE",
        f"{best_ward['MAE']:.2f}"
    )


    l3.metric(
        "Highest-Error Ward",
        worst_ward["ward"]
    )


    l4.metric(
        "Highest Ward MAE",
        f"{worst_ward['MAE']:.2f}"
    )


    leaderboard_display = (
        leaderboard_df[
            [
                "Rank",
                "hospital_id",
                "ward",
                "MAE",
                "RMSE",
                "MAPE",
                "Bias",
                "Maximum Error",
                "Records"
            ]
        ]
        .copy()
    )


    numeric_display_columns = [
        "MAE",
        "RMSE",
        "MAPE",
        "Bias",
        "Maximum Error"
    ]


    leaderboard_display[
        numeric_display_columns
    ] = leaderboard_display[
        numeric_display_columns
    ].round(2)


    st.dataframe(
        leaderboard_display,
        use_container_width=True,
        hide_index=True
    )


    leaderboard_fig = px.bar(
        leaderboard_df.head(15),
        x="ward",
        y="MAE",
        color="hospital_id",
        title=(
            "Top 15 Wards Ranked by "
            "Lowest MAE"
        )
    )


    leaderboard_fig.update_layout(
        xaxis_title="Ward",
        yaxis_title="MAE"
    )


    st.plotly_chart(
        leaderboard_fig,
        use_container_width=True
    )

else:

    st.info(
        "A ward-level leaderboard could not "
        "be calculated."
    )


# ==========================================
# ERROR QUALITY BANDS
# ==========================================

st.subheader("Forecast Quality Distribution")


quality_df = leaderboard_df.copy()


if not quality_df.empty:

    quality_df["Accuracy Band"] = pd.cut(
        quality_df["MAE"],
        bins=[
            -np.inf,
            1,
            3,
            np.inf
        ],
        labels=[
            "Strong",
            "Moderate",
            "Needs Improvement"
        ]
    )


    quality_summary = (
        quality_df["Accuracy Band"]
        .value_counts()
        .rename_axis("Accuracy Band")
        .reset_index(name="Wards")
    )


    quality_fig = px.bar(
        quality_summary,
        x="Accuracy Band",
        y="Wards",
        text="Wards",
        title=(
            "Number of Wards by "
            "Forecast Quality"
        )
    )


    quality_fig.update_layout(
        xaxis_title="Forecast Quality",
        yaxis_title="Number of Wards"
    )


    st.plotly_chart(
        quality_fig,
        use_container_width=True
    )


# ==========================================
# METRICS TABLE
# ==========================================

st.subheader("Selected Performance Metrics")


metrics_table = pd.DataFrame(
    {
        "Metric": [
            "MAE",
            "RMSE",
            "MAPE",
            "Bias",
            "Maximum Error",
            "Evaluation Records"
        ],
        "Value": [
            mae,
            rmse,
            mape,
            bias,
            max_error,
            len(analysis_df)
        ]
    }
)


metrics_table["Value"] = (
    metrics_table["Value"]
    .round(2)
)


st.dataframe(
    metrics_table,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# DOWNLOADS
# ==========================================

safe_hospital = re.sub(
    r"[^A-Za-z0-9_-]+",
    "_",
    str(hospital)
)


safe_ward = re.sub(
    r"[^A-Za-z0-9_-]+",
    "_",
    str(ward)
)


safe_model = re.sub(
    r"[^A-Za-z0-9_-]+",
    "_",
    model_choice
)


metrics_csv = metrics_table.to_csv(
    index=False
)


st.download_button(
    "Download Selected Metrics",
    metrics_csv,
    file_name=(
        f"{safe_hospital}_"
        f"{safe_ward}_"
        f"{safe_model}_metrics.csv"
    ),
    mime="text/csv"
)


if not leaderboard_df.empty:

    leaderboard_csv = (
        leaderboard_df.to_csv(
            index=False
        )
    )


    st.download_button(
        "Download Ward Leaderboard",
        leaderboard_csv,
        file_name=(
            f"{safe_model}_"
            f"ward_performance_leaderboard.csv"
        ),
        mime="text/csv"
    )


evaluation_csv = analysis_df.to_csv(
    index=False
)


st.download_button(
    "Download Evaluation Data",
    evaluation_csv,
    file_name=(
        f"{safe_hospital}_"
        f"{safe_ward}_"
        f"{safe_model}_evaluation.csv"
    ),
    mime="text/csv"
)