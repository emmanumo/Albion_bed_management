import re

import pandas as pd
import plotly.express as px
import streamlit as st

from api.client import (
    get_forecast,
    get_xgb_forecast
)


st.set_page_config(
    page_title="Forecasting",
    layout="wide"
)


# ==========================================
# PAGE TITLE
# ==========================================

st.title("Forecasting Dashboard")

st.markdown(
    """
Forecast analysis using:

- SARIMAX 24–72-hour backtesting
- SARIMAX 7–30-day backtesting
- SARIMAX 30-day future forecasting through FastAPI
- XGBoost 30-day future forecasting through FastAPI
"""
)


# ==========================================
# VALIDATE SESSION STATE
# ==========================================

required_session_items = [
    "hospital",
    "ward",
    "sarimax_short",
    "sarimax_long"
]

missing_items = [
    item
    for item in required_session_items
    if item not in st.session_state
]

if missing_items:

    st.error(
        "Required forecasting data is unavailable. "
        "Please return to the main dashboard."
    )

    st.stop()


# ==========================================
# LOAD FILTERS AND LOCAL BACKTESTING DATA
# ==========================================

hospital = st.session_state["hospital"]

ward = st.session_state["ward"]

sarimax_short = (
    st.session_state["sarimax_short"]
    .copy()
)

sarimax_long = (
    st.session_state["sarimax_long"]
    .copy()
)


# ==========================================
# LOAD FUTURE FORECASTS FROM FASTAPI
# ==========================================

try:

    sarimax_api_future = get_forecast(
        hospital,
        ward
    )

    xgb_api_future = get_xgb_forecast(
        hospital,
        ward
    )

except Exception as error:

    st.error(
        "Unable to retrieve future forecasts "
        f"from FastAPI: {error}"
    )

    st.info(
        "Confirm that FastAPI is running with: "
        "`uvicorn api.main:app --reload`"
    )

    st.stop()


# ==========================================
# ENSURE DATE FORMAT
# ==========================================

datasets = [
    sarimax_short,
    sarimax_long,
    sarimax_api_future,
    xgb_api_future
]

for dataframe in datasets:

    if "date" in dataframe.columns:

        dataframe["date"] = pd.to_datetime(
            dataframe["date"],
            errors="coerce"
        )


# ==========================================
# FILTER LOCAL BACKTESTING DATA
# ==========================================

sarimax_short_filtered = sarimax_short[
    (
        sarimax_short["hospital_id"]
        == hospital
    )
    &
    (
        sarimax_short["ward"]
        == ward
    )
].copy()


sarimax_long_filtered = sarimax_long[
    (
        sarimax_long["hospital_id"]
        == hospital
    )
    &
    (
        sarimax_long["ward"]
        == ward
    )
].copy()


# API responses are already filtered by
# hospital and ward.

sarimax_future_filtered = (
    sarimax_api_future.copy()
)

xgb_future_filtered = (
    xgb_api_future.copy()
)


# ==========================================
# PREPARE FUTURE FORECASTS
# ==========================================

if not sarimax_future_filtered.empty:

    sarimax_future_filtered[
        "forecast_occupied_beds"
    ] = pd.to_numeric(
        sarimax_future_filtered[
            "forecast_occupied_beds"
        ],
        errors="coerce"
    )

    sarimax_future_filtered = (
        sarimax_future_filtered
        .dropna(
            subset=[
                "date",
                "forecast_occupied_beds"
            ]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


if not xgb_future_filtered.empty:

    xgb_future_filtered[
        "predicted_occupied_beds"
    ] = pd.to_numeric(
        xgb_future_filtered[
            "predicted_occupied_beds"
        ],
        errors="coerce"
    )

    xgb_future_filtered = (
        xgb_future_filtered
        .dropna(
            subset=[
                "date",
                "predicted_occupied_beds"
            ]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


# ==========================================
# FORECAST MODEL SELECTION
# ==========================================

st.sidebar.header("Forecast Controls")

forecast_selection = st.sidebar.selectbox(
    "Forecast Model",
    [
        "SARIMAX 24–72 Hours",
        "SARIMAX 7–30 Days",
        "SARIMAX Future 30 Days",
        "XGBoost Future 30 Days"
    ]
)


# ==========================================
# SELECT FORECAST DATASET
# ==========================================

if forecast_selection == "SARIMAX 24–72 Hours":

    forecast_df = (
        sarimax_short_filtered.copy()
    )

    forecast_column = (
        "predicted_occupied_beds"
    )

    data_source = (
        "Local SARIMAX backtesting dataset"
    )


elif forecast_selection == "SARIMAX 7–30 Days":

    forecast_df = (
        sarimax_long_filtered.copy()
    )

    forecast_column = (
        "predicted_occupied_beds"
    )

    data_source = (
        "Local SARIMAX backtesting dataset"
    )


elif forecast_selection == "SARIMAX Future 30 Days":

    forecast_df = (
        sarimax_future_filtered.copy()
    )

    forecast_column = (
        "forecast_occupied_beds"
    )

    data_source = (
        "FastAPI SARIMAX forecast service"
    )


else:

    forecast_df = (
        xgb_future_filtered.copy()
    )

    forecast_column = (
        "predicted_occupied_beds"
    )

    data_source = (
        "FastAPI XGBoost forecast service"
    )


# ==========================================
# VALIDATE SELECTED DATASET
# ==========================================

if forecast_df.empty:

    st.warning(
        "No forecast is available for the "
        "selected hospital, ward and model."
    )

    st.stop()


if forecast_column not in forecast_df.columns:

    st.error(
        f"The expected forecast column "
        f"`{forecast_column}` is missing."
    )

    st.stop()


forecast_df[forecast_column] = pd.to_numeric(
    forecast_df[forecast_column],
    errors="coerce"
)


forecast_df = (
    forecast_df
    .dropna(
        subset=[
            "date",
            forecast_column
        ]
    )
    .sort_values("date")
    .reset_index(drop=True)
)


if forecast_df.empty:

    st.warning(
        "The selected forecast contains no "
        "valid dates or forecast values."
    )

    st.stop()


# ==========================================
# FORECAST CONTEXT
# ==========================================

st.caption(
    f"Hospital: **{hospital}** | "
    f"Ward: **{ward}** | "
    f"Source: **{data_source}**"
)


# ==========================================
# FORECAST SUMMARY KPIs
# ==========================================

st.subheader("Forecast Summary")

latest = forecast_df.iloc[-1]

average_forecast = (
    forecast_df[forecast_column]
    .mean()
)

peak_forecast = (
    forecast_df[forecast_column]
    .max()
)

peak_index = (
    forecast_df[forecast_column]
    .idxmax()
)

peak_date = pd.to_datetime(
    forecast_df.loc[
        peak_index,
        "date"
    ],
    errors="coerce"
)


if pd.isna(peak_date):

    peak_date_label = "Unknown"

else:

    peak_date_label = (
        peak_date.strftime("%Y-%m-%d")
    )


forecast_start = (
    forecast_df["date"]
    .min()
)

forecast_end = (
    forecast_df["date"]
    .max()
)


c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Latest Forecast",
    round(
        float(
            latest[forecast_column]
        ),
        1
    )
)

c2.metric(
    "Average Forecast",
    round(
        float(average_forecast),
        1
    )
)

c3.metric(
    "Peak Demand",
    round(
        float(peak_forecast),
        1
    )
)

c4.metric(
    "Peak Date",
    peak_date_label
)

c5.metric(
    "Forecast Records",
    len(forecast_df)
)


# ==========================================
# FORECAST PERIOD
# ==========================================

st.info(
    f"Forecast period: "
    f"**{forecast_start:%Y-%m-%d}** to "
    f"**{forecast_end:%Y-%m-%d}**."
)


# ==========================================
# FORECAST CURVE
# ==========================================

st.subheader(forecast_selection)

forecast_fig = px.line(
    forecast_df,
    x="date",
    y=forecast_column,
    markers=True,
    title="Predicted Occupied Beds"
)

forecast_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Occupied Beds",
    legend_title_text=""
)

st.plotly_chart(
    forecast_fig,
    use_container_width=True
)


# ==========================================
# ACTUAL VS FORECAST
# ==========================================

required_comparison_columns = {
    "actual_occupied_beds",
    "predicted_occupied_beds"
}


if required_comparison_columns.issubset(
    forecast_df.columns
):

    comparison_df = forecast_df.copy()

    comparison_df[
        "actual_occupied_beds"
    ] = pd.to_numeric(
        comparison_df[
            "actual_occupied_beds"
        ],
        errors="coerce"
    )

    comparison_df[
        "predicted_occupied_beds"
    ] = pd.to_numeric(
        comparison_df[
            "predicted_occupied_beds"
        ],
        errors="coerce"
    )

    comparison_df = (
        comparison_df
        .dropna(
            subset=[
                "actual_occupied_beds",
                "predicted_occupied_beds"
            ]
        )
    )


    if not comparison_df.empty:

        st.subheader("Actual vs Forecast")

        actual_comparison_fig = px.line(
            comparison_df,
            x="date",
            y=[
                "actual_occupied_beds",
                "predicted_occupied_beds"
            ],
            markers=True,
            title=(
                "Actual and Predicted "
                "Occupied Beds"
            )
        )

        actual_comparison_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Occupied Beds",
            legend_title_text=""
        )

        st.plotly_chart(
            actual_comparison_fig,
            use_container_width=True
        )


# ==========================================
# FORECAST CHANGE ANALYSIS
# ==========================================

st.subheader("Forecast Change Analysis")

change_df = forecast_df[
    [
        "date",
        forecast_column
    ]
].copy()

change_df["daily_change"] = (
    change_df[forecast_column]
    .diff()
)


change_fig = px.bar(
    change_df,
    x="date",
    y="daily_change",
    title=(
        "Day-to-Day Change in "
        "Forecast Bed Demand"
    )
)

change_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Change in Beds"
)

st.plotly_chart(
    change_fig,
    use_container_width=True
)


# ==========================================
# FUTURE MODEL COMPARISON
# ==========================================

st.subheader(
    "SARIMAX Future vs XGBoost Future"
)


if (
    not sarimax_future_filtered.empty
    and
    not xgb_future_filtered.empty
):

    future_comparison = (
        sarimax_future_filtered.merge(
            xgb_future_filtered,
            on=[
                "hospital_id",
                "ward",
                "date"
            ],
            how="inner"
        )
    )


    if not future_comparison.empty:

        future_comparison = (
            future_comparison
            .sort_values("date")
            .reset_index(drop=True)
        )


        future_comparison[
            "model_difference"
        ] = (
            future_comparison[
                "predicted_occupied_beds"
            ]
            -
            future_comparison[
                "forecast_occupied_beds"
            ]
        )


        future_comparison[
            "absolute_difference"
        ] = (
            future_comparison[
                "model_difference"
            ]
            .abs()
        )


        comparison_fig = px.line(
            future_comparison,
            x="date",
            y=[
                "forecast_occupied_beds",
                "predicted_occupied_beds"
            ],
            markers=True,
            title=(
                "30-Day SARIMAX and "
                "XGBoost Forecast Comparison"
            )
        )

        comparison_fig.update_layout(
            xaxis_title="Forecast Date",
            yaxis_title="Occupied Beds",
            legend_title_text="Model"
        )

        st.plotly_chart(
            comparison_fig,
            use_container_width=True
        )


        # ==================================
        # COMPARISON KPIs
        # ==================================

        sarimax_average = (
            future_comparison[
                "forecast_occupied_beds"
            ]
            .mean()
        )

        xgb_average = (
            future_comparison[
                "predicted_occupied_beds"
            ]
            .mean()
        )

        sarimax_peak = (
            future_comparison[
                "forecast_occupied_beds"
            ]
            .max()
        )

        xgb_peak = (
            future_comparison[
                "predicted_occupied_beds"
            ]
            .max()
        )

        mean_absolute_difference = (
            future_comparison[
                "absolute_difference"
            ]
            .mean()
        )

        maximum_difference = (
            future_comparison[
                "absolute_difference"
            ]
            .max()
        )


        m1, m2, m3, m4, m5, m6 = (
            st.columns(6)
        )


        m1.metric(
            "SARIMAX Average",
            round(
                float(sarimax_average),
                1
            )
        )


        m2.metric(
            "XGBoost Average",
            round(
                float(xgb_average),
                1
            )
        )


        m3.metric(
            "SARIMAX Peak",
            round(
                float(sarimax_peak),
                1
            )
        )


        m4.metric(
            "XGBoost Peak",
            round(
                float(xgb_peak),
                1
            )
        )


        m5.metric(
            "Average Model Gap",
            round(
                float(
                    mean_absolute_difference
                ),
                2
            )
        )


        m6.metric(
            "Maximum Model Gap",
            round(
                float(
                    maximum_difference
                ),
                2
            )
        )


        # ==================================
        # MODEL DIFFERENCE CHART
        # ==================================

        difference_fig = px.bar(
            future_comparison,
            x="date",
            y="model_difference",
            title=(
                "XGBoost Forecast Minus "
                "SARIMAX Forecast"
            )
        )

        difference_fig.update_layout(
            xaxis_title="Forecast Date",
            yaxis_title="Difference in Beds"
        )

        st.plotly_chart(
            difference_fig,
            use_container_width=True
        )


        # ==================================
        # COMPARISON INTERPRETATION
        # ==================================

        if mean_absolute_difference <= 1:

            st.success(
                """
🟢 The two future models are closely aligned. Their average forecasts differ by no more than approximately one occupied bed.
"""
            )

        elif mean_absolute_difference <= 3:

            st.warning(
                """
🟡 The two future models show a moderate level of disagreement. Operational teams should review both forecast ranges when planning capacity.
"""
            )

        else:

            st.error(
                """
🔴 The two future models show a material difference in predicted bed demand. Further model validation and investigation of demand drivers is recommended.
"""
            )


        # ==================================
        # FUTURE COMPARISON TABLE
        # ==================================

        st.subheader(
            "Future Forecast Comparison Detail"
        )

        st.dataframe(
            future_comparison[
                [
                    "date",
                    "forecast_occupied_beds",
                    "predicted_occupied_beds",
                    "model_difference",
                    "absolute_difference"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


    else:

        st.warning(
            "SARIMAX and XGBoost forecasts "
            "do not contain matching dates."
        )


else:

    st.warning(
        "Both future forecast datasets are "
        "required for model comparison."
    )


# ==========================================
# FORECAST DATA TABLE
# ==========================================

st.subheader("Selected Forecast Data")

display_columns = [
    column
    for column in [
        "hospital_id",
        "ward",
        "date",
        "actual_occupied_beds",
        forecast_column
    ]
    if column in forecast_df.columns
]

st.dataframe(
    forecast_df[
        display_columns
    ].sort_values("date"),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# DOWNLOAD SELECTED FORECAST
# ==========================================

csv = forecast_df.to_csv(
    index=False
)


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
    forecast_selection
)


st.download_button(
    label="Download Selected Forecast CSV",
    data=csv,
    file_name=(
        f"{safe_hospital}_"
        f"{safe_ward}_"
        f"{safe_model}.csv"
    ),
    mime="text/csv"
)


# ==========================================
# DOWNLOAD MODEL COMPARISON
# ==========================================

if (
    "future_comparison" in locals()
    and
    not future_comparison.empty
):

    comparison_csv = (
        future_comparison.to_csv(
            index=False
        )
    )


    st.download_button(
        label="Download Future Model Comparison",
        data=comparison_csv,
        file_name=(
            f"{safe_hospital}_"
            f"{safe_ward}_"
            f"SARIMAX_XGBoost_"
            f"comparison.csv"
        ),
        mime="text/csv"
    )