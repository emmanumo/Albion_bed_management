import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from api.client import get_dashboard_summary


st.set_page_config(
    page_title="Overview",
    layout="wide"
)


# ==========================================
# PAGE HEADER
# ==========================================

st.title("Overview Dashboard")

st.markdown(
    """
Executive operational view of:

- 🛏 Bed demand forecast
- 🚑 Patient flow pressure
- ⚠️ Capacity risk
- 👩‍⚕️ Staffing readiness
"""
)


# ==========================================
# VALIDATE SESSION STATE
# ==========================================

required_session_items = [
    "hospital",
    "ward",
    "model_df",
    "sarimax_future",
    "staffing",
    "xgb_future"
]

missing_items = [
    item
    for item in required_session_items
    if item not in st.session_state
]

if missing_items:

    st.error(
        "Required dashboard data is unavailable. "
        "Please return to the main dashboard."
    )

    st.stop()


# ==========================================
# RECEIVE SHARED DATA AND FILTERS
# ==========================================

hospital = st.session_state["hospital"]

ward = st.session_state["ward"]

model_df = st.session_state["model_df"].copy()

sarimax_future = (
    st.session_state["sarimax_future"]
    .copy()
)

staffing = (
    st.session_state["staffing"]
    .copy()
)

xgb_future = (
    st.session_state["xgb_future"]
    .copy()
)


# ==========================================
# LOAD EXECUTIVE SUMMARY FROM FASTAPI
# ==========================================

try:

    summary = get_dashboard_summary(
        hospital,
        ward
    )

except Exception as error:

    st.error(
        f"Unable to load dashboard summary "
        f"from FastAPI: {error}"
    )

    st.info(
        "Confirm that FastAPI is running with: "
        "`uvicorn api.main:app --reload`"
    )

    st.stop()


if not summary:

    st.warning(
        "No dashboard summary is available for "
        "the selected hospital and ward."
    )

    st.stop()


# ==========================================
# ENSURE DATE COLUMNS ARE DATETIME
# ==========================================

for dataframe in [
    model_df,
    sarimax_future,
    staffing,
    xgb_future
]:

    if "date" in dataframe.columns:

        dataframe["date"] = pd.to_datetime(
            dataframe["date"],
            errors="coerce"
        )


# ==========================================
# HOSPITAL SNAPSHOT
# ==========================================

st.subheader("Hospital Snapshot")

s1, s2, s3 = st.columns(3)

s1.metric(
    "Hospital",
    hospital
)

s2.metric(
    "Ward",
    ward
)

s3.metric(
    "Forecast Horizon",
    f"{summary.get('forecast_days', 30)} Days"
)


# ==========================================
# FILTER LOCAL DATASETS
# ==========================================

operations = model_df[
    (model_df["hospital_id"] == hospital)
    &
    (model_df["ward"] == ward)
].copy()


forecast = sarimax_future[
    (sarimax_future["hospital_id"] == hospital)
    &
    (sarimax_future["ward"] == ward)
].copy()


staff = staffing[
    (staffing["hospital_id"] == hospital)
    &
    (staffing["ward"] == ward)
].copy()


xgb_filtered = xgb_future[
    (xgb_future["hospital_id"] == hospital)
    &
    (xgb_future["ward"] == ward)
].copy()


operations = operations.sort_values("date")

forecast = forecast.sort_values("date")

staff = staff.sort_values("date")

xgb_filtered = xgb_filtered.sort_values("date")


if operations.empty:

    st.warning(
        "No operational data is available for "
        "the selected hospital and ward."
    )

    st.stop()


if forecast.empty:

    st.warning(
        "No future forecast is available for "
        "the selected hospital and ward."
    )

    st.stop()


# ==========================================
# LATEST OPERATIONAL VALUES
# ==========================================

latest_operation = operations.iloc[-1]

latest_forecast = forecast.iloc[-1]


latest_total_beds = (
    pd.to_numeric(
        latest_operation.get("total_beds"),
        errors="coerce"
    )
)

latest_staffed_beds = (
    pd.to_numeric(
        latest_operation.get("staffed_beds"),
        errors="coerce"
    )
)


if pd.isna(latest_total_beds):

    latest_total_beds = (
        summary.get("total_beds", 0)
        or 0
    )


if pd.isna(latest_staffed_beds):

    latest_staffed_beds = (
        summary.get("staffed_beds", 0)
        or 0
    )


# ==========================================
# CREATE DASHBOARD DATASET
# ==========================================

dashboard_df = forecast.copy()


# Future forecast dates do not exist in the
# historical operational dataset. Use the
# latest known capacity for the forecast period.

dashboard_df["total_beds"] = (
    latest_total_beds
)

dashboard_df["staffed_beds"] = (
    latest_staffed_beds
)


dashboard_df["occupancy_forecast_rate"] = np.where(

    dashboard_df["total_beds"] > 0,

    dashboard_df["forecast_occupied_beds"]
    /
    dashboard_df["total_beds"],

    np.nan

)


dashboard_df["staffed_occupancy_rate"] = np.where(

    dashboard_df["staffed_beds"] > 0,

    dashboard_df["forecast_occupied_beds"]
    /
    dashboard_df["staffed_beds"],

    np.nan

)


dashboard_df["capacity_risk"] = np.select(

    [
        dashboard_df["forecast_occupied_beds"]
        >
        dashboard_df["total_beds"],

        dashboard_df["forecast_occupied_beds"]
        >
        dashboard_df["staffed_beds"],

        dashboard_df[
            "staffed_occupancy_rate"
        ] >= 0.90
    ],

    [
        "Over Capacity",
        "High Risk",
        "Moderate"
    ],

    default="Normal"

)


# ==========================================
# KPI CALCULATIONS FROM FASTAPI
# ==========================================

forecast_beds = (
    summary.get("forecast_beds", 0)
    or 0
)

total_beds = (
    summary.get("total_beds", 0)
    or 0
)

staffed_beds = (
    summary.get("staffed_beds", 0)
    or 0
)

available_beds = (
    summary.get("available_beds", 0)
    or 0
)

occupancy_rate = (
    summary.get("occupancy_rate", 0)
    or 0
)

forecast_days = (
    summary.get("forecast_days", 0)
    or 0
)


# ==========================================
# CAPACITY STATUS FORMATTING
# ==========================================

capacity_status_raw = (
    summary.get("capacity_risk")
    or "Unknown"
)

capacity_status_map = {
    "Critical": "🔴 Critical",
    "Over Capacity": "🔴 Over Capacity",
    "High": "🟠 High Risk",
    "High Risk": "🟠 High Risk",
    "Moderate": "🟡 Moderate",
    "Normal": "🟢 Normal",
    "Unknown": "⚪ Unknown"
}

capacity_status = capacity_status_map.get(
    capacity_status_raw,
    f"⚪ {capacity_status_raw}"
)


# ==========================================
# STAFFING STATUS FORMATTING
# ==========================================

staffing_status_raw = (
    summary.get("staffing_risk")
    or "Unknown"
)

staffing_status_map = {
    "Critical": "🔴 Critical",
    "Critical Risk": "🔴 Critical Risk",
    "High": "🟠 High Risk",
    "High Risk": "🟠 High Risk",
    "Moderate": "🟡 Moderate",
    "Medium": "🟡 Medium Risk",
    "Medium Risk": "🟡 Medium Risk",
    "Low": "🟢 Low Risk",
    "Low Risk": "🟢 Low Risk",
    "Normal": "🟢 Normal",
    "Safe": "🟢 Safe",
    "Unknown": "⚪ Unknown"
}

staffing_status = staffing_status_map.get(
    staffing_status_raw,
    f"⚪ {staffing_status_raw}"
)


# ==========================================
# KPI CARDS
# ==========================================

st.subheader("Hospital Status")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Forecast Occupied Beds",
    round(float(forecast_beds), 1)
)

c2.metric(
    "Occupancy Rate",
    f"{float(occupancy_rate):.1%}"
)

c3.metric(
    "Available Beds",
    round(float(available_beds), 1)
)

c4.metric(
    "Capacity Risk",
    capacity_status
)

c5.metric(
    "Staffing Risk",
    staffing_status
)


# ==========================================
# PATIENT FLOW PRESSURE
# ==========================================

st.subheader("Patient Flow Pressure")

latest_admissions = (
    pd.to_numeric(
        latest_operation.get(
            "daily_admissions",
            0
        ),
        errors="coerce"
    )
)

latest_discharges = (
    pd.to_numeric(
        latest_operation.get(
            "daily_discharges",
            0
        ),
        errors="coerce"
    )
)


if pd.isna(latest_admissions):

    latest_admissions = 0


if pd.isna(latest_discharges):

    latest_discharges = 0


flow_balance = (
    latest_admissions
    -
    latest_discharges
)


if flow_balance > 0:

    flow_status = (
        "🔴 Increasing Pressure"
    )

elif flow_balance < 0:

    flow_status = (
        "🟢 Reducing Pressure"
    )

else:

    flow_status = (
        "🟡 Balanced"
    )


f1, f2, f3, f4 = st.columns(4)

f1.metric(
    "Daily Admissions",
    round(float(latest_admissions), 1)
)

f2.metric(
    "Daily Discharges",
    round(float(latest_discharges), 1)
)

f3.metric(
    "Net Patient Flow",
    round(float(flow_balance), 1)
)

f4.metric(
    "Flow Status",
    flow_status
)


# ==========================================
# BED PRESSURE INDEX
# ==========================================

st.subheader("Bed Pressure Index")


if staffed_beds > 0:

    pressure_index = (
        forecast_beds
        /
        staffed_beds
    )

else:

    pressure_index = 0


if pressure_index >= 1:

    pressure_status = "🔴 Critical"

elif pressure_index >= 0.90:

    pressure_status = (
        "🟠 High Pressure"
    )

elif pressure_index >= 0.75:

    pressure_status = "🟡 Moderate"

else:

    pressure_status = "🟢 Low"


p1, p2 = st.columns(2)

p1.metric(
    "Pressure Index",
    f"{float(pressure_index):.1%}"
)

p2.metric(
    "Operational Status",
    pressure_status
)


# ==========================================
# FORECAST TREND
# ==========================================

st.subheader("30-Day Forecast Occupancy")

forecast_fig = px.line(
    forecast,
    x="date",
    y="forecast_occupied_beds",
    markers=True,
    title="Forecast Occupied Beds"
)

forecast_fig.update_layout(
    xaxis_title="Forecast Date",
    yaxis_title="Occupied Beds",
    legend_title_text=""
)

st.plotly_chart(
    forecast_fig,
    use_container_width=True
)


# ==========================================
# CAPACITY UTILISATION
# ==========================================

st.subheader("Capacity Utilisation")

capacity_fig = px.line(
    dashboard_df,
    x="date",
    y=[
        "forecast_occupied_beds",
        "staffed_beds",
        "total_beds"
    ],
    markers=True,
    title=(
        "Forecast Demand vs Operational Capacity"
    )
)

capacity_fig.update_layout(
    xaxis_title="Forecast Date",
    yaxis_title="Beds",
    legend_title_text=""
)

st.plotly_chart(
    capacity_fig,
    use_container_width=True
)


# ==========================================
# CAPACITY RISK SUMMARY
# ==========================================

st.subheader("Capacity Risk Summary")

risk_summary = (
    dashboard_df
    .groupby("capacity_risk")
    .size()
    .reset_index(name="days")
)


risk_fig = px.bar(
    risk_summary,
    x="capacity_risk",
    y="days",
    text="days",
    title="Number of Forecast Days by Risk Level"
)

risk_fig.update_layout(
    xaxis_title="Capacity Risk",
    yaxis_title="Forecast Days"
)

st.plotly_chart(
    risk_fig,
    use_container_width=True
)


# ==========================================
# RISK DISTRIBUTION KPIs
# ==========================================

risk_counts = (
    dashboard_df["capacity_risk"]
    .value_counts()
)


r1, r2, r3, r4 = st.columns(4)

r1.metric(
    "Normal Days",
    int(risk_counts.get("Normal", 0))
)

r2.metric(
    "Moderate Days",
    int(risk_counts.get("Moderate", 0))
)

r3.metric(
    "High-Risk Days",
    int(risk_counts.get("High Risk", 0))
)

r4.metric(
    "Over-Capacity Days",
    int(
        risk_counts.get(
            "Over Capacity",
            0
        )
    )
)


# ==========================================
# MODEL COMPARISON
# ==========================================

st.subheader("Forecast Model Comparison")


if not xgb_filtered.empty:

    comparison = forecast.merge(
        xgb_filtered,
        on=[
            "hospital_id",
            "ward",
            "date"
        ],
        how="inner"
    )

else:

    comparison = pd.DataFrame()


if not comparison.empty:

    comparison_fig = px.line(
        comparison,
        x="date",
        y=[
            "forecast_occupied_beds",
            "predicted_occupied_beds"
        ],
        markers=True,
        title=(
            "SARIMAX vs XGBoost Forecast"
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

    comparison["forecast_difference"] = (
        comparison[
            "predicted_occupied_beds"
        ]
        -
        comparison[
            "forecast_occupied_beds"
        ]
    )

    average_difference = (
        comparison["forecast_difference"]
        .abs()
        .mean()
    )

    st.metric(
        "Average Difference Between Models",
        round(float(average_difference), 2)
    )

else:

    st.warning(
        "No matching XGBoost forecast is "
        "available for model comparison."
    )


# ==========================================
# OPERATIONAL ALERTS
# ==========================================

st.subheader("Operational Alerts")

critical_days = dashboard_df[
    dashboard_df["capacity_risk"]
    ==
    "Over Capacity"
]

high_risk_days = dashboard_df[
    dashboard_df["capacity_risk"]
    ==
    "High Risk"
]

moderate_days = dashboard_df[
    dashboard_df["capacity_risk"]
    ==
    "Moderate"
]


if not critical_days.empty:

    first_critical_date = (
        critical_days["date"]
        .min()
        .strftime("%Y-%m-%d")
    )

    st.error(
        f"""
🔴 **Critical capacity alert**

{len(critical_days)} forecast day(s) exceed total bed capacity.

The first critical date is **{first_critical_date}**.
"""
    )

elif not high_risk_days.empty:

    first_high_risk_date = (
        high_risk_days["date"]
        .min()
        .strftime("%Y-%m-%d")
    )

    st.warning(
        f"""
🟠 **High capacity pressure**

{len(high_risk_days)} forecast day(s) exceed staffed-bed capacity.

The first high-risk date is **{first_high_risk_date}**.
"""
    )

elif not moderate_days.empty:

    st.warning(
        f"""
🟡 **Moderate capacity pressure**

{len(moderate_days)} forecast day(s) are expected to operate at or above 90% of staffed capacity.
"""
    )

else:

    st.success(
        """
🟢 No significant bed-capacity risk was detected during the forecast period.
"""
    )


if staffing_status_raw in [
    "Critical",
    "Critical Risk",
    "High",
    "High Risk"
]:

    st.error(
        f"""
👩‍⚕️ **Staffing alert**

The current forecast staffing status is **{staffing_status_raw}**. Review planned and actual staffing levels for this unit.
"""
    )