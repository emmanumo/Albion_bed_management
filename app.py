import pandas as pd
import streamlit as st

from api.client import (
    get_hospitals,
    get_wards,
    BASE_URL
)

from styles.dashboard_style import load_css


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title=(
        "Albion Care Network "
        "Hospital Operations Intelligence Platform"
    ),
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# SHARED STYLING
# ==========================================

load_css()


# ==========================================
# API CONNECTION CHECK
# ==========================================

st.sidebar.caption(
    f"API: {BASE_URL}"
)


# ==========================================
# LOAD LOCAL DATASETS
# ==========================================

@st.cache_data
def load_local_data():

    model_df = pd.read_csv(
        "data/processed/"
        "bed_occupancy_modelling_dataset.csv"
    )

    timeseries = pd.read_csv(
        "data/processed/"
        "bed_occupancy_timeseries.csv"
    )

    sarimax_short = pd.read_csv(
        "data/processed/"
        "sarimax_24_72_forecast.csv"
    )

    sarimax_long = pd.read_csv(
        "data/processed/"
        "sarimax_7_30_forecast.csv"
    )

    sarimax_future = pd.read_csv(
        "data/processed/"
        "sarimax_future_forecast.csv"
    )

    xgb_future = pd.read_csv(
        "data/processed/"
        "xgb_future_forecast.csv"
    )

    staffing = pd.read_csv(
        "data/processed/"
        "staffing_risk_forecast.csv"
    )


    datasets = [
        model_df,
        timeseries,
        sarimax_short,
        sarimax_long,
        sarimax_future,
        xgb_future,
        staffing
    ]


    for dataframe in datasets:

        if "date" in dataframe.columns:

            dataframe["date"] = pd.to_datetime(
                dataframe["date"],
                errors="coerce"
            )


    return (
        model_df,
        timeseries,
        sarimax_short,
        sarimax_long,
        sarimax_future,
        xgb_future,
        staffing
    )


(
    model_df,
    timeseries,
    sarimax_short,
    sarimax_long,
    sarimax_future,
    xgb_future,
    staffing
) = load_local_data()


# ==========================================
# PLATFORM HEADER
# ==========================================

st.markdown(
    """
# 🏥 Albion Care Network

## Hospital Operations Intelligence Platform

Real-time analytics for:

- 🛏 Bed occupancy forecasting
- 🚑 Patient flow monitoring
- ⚠️ Capacity risk prediction
- 👩‍⚕️ Staffing feasibility analysis
- 🤖 Predictive model monitoring
"""
)


st.info(
    """
**Purpose**

This platform supports hospital operational decisions by predicting
future bed demand, identifying capacity pressure, and highlighting
staffing risks from 24 hours to 30 days ahead.
"""
)


# ==========================================
# LOAD FILTER VALUES FROM FASTAPI
# ==========================================

try:

    hospital_list = get_hospitals()

except Exception as error:

    st.error(
        "Unable to retrieve the hospital list "
        f"from FastAPI: {error}"
    )

    st.info(
        "Confirm that FastAPI is running with: "
        "`uvicorn api.main:app --reload`"
    )

    st.stop()


if not hospital_list:

    st.warning(
        "FastAPI returned no available hospitals."
    )

    st.stop()


# ==========================================
# SIDEBAR CONTROL PANEL
# ==========================================

st.sidebar.markdown(
    """
## 🏥 Control Panel

Select the hospital and operational unit:
"""
)


hospital = st.sidebar.selectbox(
    "Hospital",
    hospital_list
)


try:

    ward_list = get_wards(
        hospital
    )

except Exception as error:

    st.sidebar.error(
        "Unable to retrieve wards from FastAPI: "
        f"{error}"
    )

    st.stop()


if not ward_list:

    st.sidebar.warning(
        "No wards are available for the "
        "selected hospital."
    )

    st.stop()


ward = st.sidebar.selectbox(
    "Ward / Unit",
    ward_list
)


# ==========================================
# BED TYPE FILTER
# ==========================================

hospital_ward_data = model_df[
    (
        model_df["hospital_id"]
        == hospital
    )
    &
    (
        model_df["ward"]
        == ward
    )
]


available_bed_types = sorted(
    hospital_ward_data[
        "bed_type"
    ]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


bed_type = st.sidebar.selectbox(
    "Patient / Bed Type",
    [
        "All"
    ]
    +
    available_bed_types
)


st.sidebar.divider()


st.sidebar.caption(
    """
Albion Care Network Analytics Platform

**Models:**  
SARIMAX and XGBoost

**Forecast coverage:**  
24 hours to 30 days

**Backend:**  
FastAPI forecasting service
"""
)


# ==========================================
# SHARE DATA THROUGH SESSION STATE
# ==========================================

st.session_state["model_df"] = (
    model_df
)

st.session_state["timeseries"] = (
    timeseries
)

st.session_state["sarimax_short"] = (
    sarimax_short
)

st.session_state["sarimax_long"] = (
    sarimax_long
)

st.session_state["sarimax_future"] = (
    sarimax_future
)

st.session_state["xgb_future"] = (
    xgb_future
)

st.session_state["staffing"] = (
    staffing
)

st.session_state["hospital"] = (
    hospital
)

st.session_state["ward"] = (
    ward
)

st.session_state["bed_type"] = (
    bed_type
)


# ==========================================
# NAVIGATION
# ==========================================

overview = st.Page(
    "pages/overview.py",
    title="Executive Overview",
    icon="📊",
    default=True
)


forecasting = st.Page(
    "pages/forecasting.py",
    title="Demand Forecasting",
    icon="📈"
)


patient_flow = st.Page(
    "pages/patient_flow.py",
    title="Patient Flow Analytics",
    icon="🚑"
)


capacity_staffing = st.Page(
    "pages/capacity_staffing.py",
    title="Capacity & Staffing Risk",
    icon="⚠️"
)


model_performance = st.Page(
    "pages/model_performance.py",
    title="Model Performance",
    icon="🤖"
)


navigation = st.navigation(
    [
        overview,
        forecasting,
        patient_flow,
        capacity_staffing,
        model_performance
    ]
)


navigation.run()