import pandas as pd
import plotly.express as px
import streamlit as st

from api.client import (
    get_capacity_risk,
    get_dashboard_summary,
    get_staffing
)


st.set_page_config(
    page_title="Capacity & Staffing",
    layout="wide"
)


# ==========================================
# PAGE TITLE
# ==========================================

st.title("Capacity Risk & Staffing Analysis")

st.markdown(
    """
Operational risk monitoring for:

- Bed-capacity pressure
- Overcapacity detection
- Staffing feasibility
- Safe staffing-ratio compliance
"""
)


# ==========================================
# VALIDATE SESSION STATE
# ==========================================

required_items = [
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
        "Hospital and ward filters are unavailable. "
        "Please return to the main dashboard."
    )

    st.stop()


# ==========================================
# LOAD FILTERS
# ==========================================

hospital = st.session_state["hospital"]

ward = st.session_state["ward"]


# ==========================================
# LOAD DATA FROM FASTAPI
# ==========================================

try:

    capacity_df = get_capacity_risk(
        hospital,
        ward
    )

    staff = get_staffing(
        hospital,
        ward
    )

    summary = get_dashboard_summary(
        hospital,
        ward
    )

except Exception as error:

    st.error(
        f"Unable to retrieve capacity and staffing "
        f"data from FastAPI: {error}"
    )

    st.info(
        "Confirm that FastAPI is running with: "
        "`uvicorn api.main:app --reload`"
    )

    st.stop()


# ==========================================
# VALIDATE API RESPONSES
# ==========================================

if capacity_df.empty:

    st.warning(
        "No capacity-risk forecast is available for "
        "the selected hospital and ward."
    )

    st.stop()


# ==========================================
# PREPARE CAPACITY DATA
# ==========================================

capacity_df["date"] = pd.to_datetime(
    capacity_df["date"],
    errors="coerce"
)


capacity_numeric_columns = [
    "forecast_occupied_beds",
    "staffed_beds",
    "total_beds",
    "occupancy_rate"
]


for column in capacity_numeric_columns:

    if column in capacity_df.columns:

        capacity_df[column] = pd.to_numeric(
            capacity_df[column],
            errors="coerce"
        )


capacity_df = (
    capacity_df
    .dropna(
        subset=[
            "date",
            "forecast_occupied_beds"
        ]
    )
    .sort_values("date")
    .reset_index(drop=True)
)


if capacity_df.empty:

    st.warning(
        "The capacity-risk endpoint returned "
        "no valid forecast records."
    )

    st.stop()


# ==========================================
# PREPARE STAFFING DATA
# ==========================================

if not staff.empty:

    staff["date"] = pd.to_datetime(
        staff["date"],
        errors="coerce"
    )


    staffing_numeric_columns = [
        "predicted_occupied_beds",
        "planned_staff",
        "actual_staff",
        "staffing_ratio"
    ]


    for column in staffing_numeric_columns:

        if column in staff.columns:

            staff[column] = pd.to_numeric(
                staff[column],
                errors="coerce"
            )


    staff = (
        staff
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


# ==========================================
# PAGE CONTEXT
# ==========================================

st.caption(
    f"Hospital: **{hospital}** | "
    f"Ward: **{ward}** | "
    f"Source: **FastAPI capacity and staffing services**"
)


# ==========================================
# KPI CALCULATIONS
# ==========================================

latest_capacity = capacity_df.iloc[-1]


forecast_beds = (
    summary.get("forecast_beds", 0)
    or latest_capacity.get(
        "forecast_occupied_beds",
        0
    )
    or 0
)


total_beds = (
    summary.get("total_beds", 0)
    or latest_capacity.get(
        "total_beds",
        0
    )
    or 0
)


staffed_beds = (
    summary.get("staffed_beds", 0)
    or latest_capacity.get(
        "staffed_beds",
        0
    )
    or 0
)


occupancy_rate = (
    summary.get("occupancy_rate", 0)
    or latest_capacity.get(
        "occupancy_rate",
        0
    )
    or 0
)


risk_days = int(
    (
        capacity_df["risk_level"]
        != "Normal"
    ).sum()
)


critical_days = int(
    (
        capacity_df["risk_level"]
        == "Critical"
    ).sum()
)


capacity_status_raw = (
    summary.get("capacity_risk")
    or latest_capacity.get("risk_level")
    or "Unknown"
)


capacity_status_map = {
    "Critical": "🔴 Critical",
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
# CAPACITY OVERVIEW
# ==========================================

st.subheader("Capacity Overview")


c1, c2, c3, c4, c5, c6 = st.columns(6)


c1.metric(
    "Forecast Beds",
    round(
        float(forecast_beds),
        1
    )
)


c2.metric(
    "Total Beds",
    int(
        round(
            float(total_beds)
        )
    )
)


c3.metric(
    "Staffed Beds",
    int(
        round(
            float(staffed_beds)
        )
    )
)


c4.metric(
    "Occupancy",
    f"{float(occupancy_rate):.1%}"
)


c5.metric(
    "Capacity Status",
    capacity_status
)


c6.metric(
    "Risk Days",
    risk_days
)


# ==========================================
# CAPACITY PRESSURE FORECAST
# ==========================================

st.subheader("Capacity Pressure Forecast")


capacity_fig = px.line(
    capacity_df,
    x="date",
    y=[
        "forecast_occupied_beds",
        "staffed_beds",
        "total_beds"
    ],
    markers=True,
    title="Forecast Demand Compared with Available Capacity"
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
# CAPACITY RISK TIMELINE
# ==========================================

st.subheader("Capacity Risk Timeline")


risk_fig = px.scatter(
    capacity_df,
    x="date",
    y="forecast_occupied_beds",
    color="risk_level",
    size="occupancy_rate",
    hover_data=[
        "staffed_beds",
        "total_beds",
        "occupancy_rate"
    ],
    title="Forecast Risk by Date"
)


risk_fig.update_layout(
    xaxis_title="Forecast Date",
    yaxis_title="Forecast Occupied Beds",
    legend_title_text="Risk Level"
)


st.plotly_chart(
    risk_fig,
    use_container_width=True
)


# ==========================================
# CAPACITY RISK DISTRIBUTION
# ==========================================

st.subheader("Capacity Risk Distribution")


risk_summary = (
    capacity_df["risk_level"]
    .value_counts()
    .rename_axis("risk_level")
    .reset_index(name="days")
)


risk_bar = px.bar(
    risk_summary,
    x="risk_level",
    y="days",
    text="days",
    title="Forecast Days by Capacity Risk Level"
)


risk_bar.update_layout(
    xaxis_title="Risk Level",
    yaxis_title="Forecast Days"
)


st.plotly_chart(
    risk_bar,
    use_container_width=True
)


# ==========================================
# RISK KPI CARDS
# ==========================================

risk_counts = (
    capacity_df["risk_level"]
    .value_counts()
)


r1, r2, r3, r4 = st.columns(4)


r1.metric(
    "Normal Days",
    int(
        risk_counts.get(
            "Normal",
            0
        )
    )
)


r2.metric(
    "Moderate Days",
    int(
        risk_counts.get(
            "Moderate",
            0
        )
    )
)


r3.metric(
    "High-Risk Days",
    int(
        risk_counts.get(
            "High",
            0
        )
    )
)


r4.metric(
    "Critical Days",
    critical_days
)


# ==========================================
# STAFFING ALIGNMENT
# ==========================================

st.subheader("Staffing Alignment")


if not staff.empty:

    latest_staff = staff.iloc[-1]


    planned_staff = (
        latest_staff.get(
            "planned_staff",
            0
        )
        or 0
    )


    actual_staff = (
        latest_staff.get(
            "actual_staff",
            0
        )
        or 0
    )


    staffing_ratio = (
        latest_staff.get(
            "staffing_ratio",
            0
        )
        or 0
    )


    safe_ratio_met = (
        latest_staff.get(
            "safe_ratio_met",
            "Unknown"
        )
    )


    s1, s2, s3, s4, s5 = st.columns(5)


    s1.metric(
        "Planned Staff",
        round(
            float(planned_staff),
            1
        )
    )


    s2.metric(
        "Actual Staff",
        round(
            float(actual_staff),
            1
        )
    )


    s3.metric(
        "Staffing Ratio",
        round(
            float(staffing_ratio),
            2
        )
    )


    s4.metric(
        "Safe Ratio Met",
        str(safe_ratio_met)
    )


    s5.metric(
        "Staffing Risk",
        staffing_status
    )


    # ======================================
    # STAFFING LEVELS
    # ======================================

    staffing_columns = [
        column
        for column in [
            "planned_staff",
            "actual_staff"
        ]
        if column in staff.columns
    ]


    if staffing_columns:

        staff_fig = px.line(
            staff,
            x="date",
            y=staffing_columns,
            markers=True,
            title="Planned vs Actual Staffing"
        )


        staff_fig.update_layout(
            xaxis_title="Forecast Date",
            yaxis_title="Staff Count",
            legend_title_text=""
        )


        st.plotly_chart(
            staff_fig,
            use_container_width=True
        )


    # ======================================
    # STAFFING RATIO
    # ======================================

    if "staffing_ratio" in staff.columns:

        ratio_fig = px.line(
            staff,
            x="date",
            y="staffing_ratio",
            markers=True,
            title="Staffing Ratio Trend"
        )


        ratio_fig.add_hline(
            y=1,
            line_dash="dash",
            annotation_text="Target Ratio"
        )


        ratio_fig.update_layout(
            xaxis_title="Forecast Date",
            yaxis_title="Staffing Ratio"
        )


        st.plotly_chart(
            ratio_fig,
            use_container_width=True
        )


    # ======================================
    # STAFFING COMPLIANCE
    # ======================================

    if "safe_ratio_met" in staff.columns:

        safe_values = (
            staff["safe_ratio_met"]
            .astype(str)
            .str.lower()
        )


        safe_days = int(
            safe_values.isin(
                [
                    "true",
                    "yes",
                    "1",
                    "safe"
                ]
            ).sum()
        )


        total_staffing_days = len(staff)


        compliance_rate = (
            safe_days
            /
            total_staffing_days
            if total_staffing_days
            else 0
        )


        st.metric(
            "Safe Staffing Compliance",
            f"{compliance_rate:.1%}",
            help=(
                "Percentage of forecast days where "
                "the safe staffing ratio is met."
            )
        )


else:

    st.warning(
        "No staffing forecast is available for "
        "the selected hospital and ward."
    )


# ==========================================
# OPERATIONAL ALERTS
# ==========================================

st.subheader("Operational Alerts")


critical_capacity = capacity_df[
    capacity_df["risk_level"]
    ==
    "Critical"
]


high_capacity = capacity_df[
    capacity_df["risk_level"]
    ==
    "High"
]


moderate_capacity = capacity_df[
    capacity_df["risk_level"]
    ==
    "Moderate"
]


if not critical_capacity.empty:

    first_critical_date = (
        critical_capacity["date"]
        .min()
        .strftime("%Y-%m-%d")
    )


    st.error(
        f"""
🔴 **Critical bed-capacity alert**

{len(critical_capacity)} forecast day(s) exceed total bed capacity.

The first critical date is **{first_critical_date}**.
"""
    )


elif not high_capacity.empty:

    first_high_date = (
        high_capacity["date"]
        .min()
        .strftime("%Y-%m-%d")
    )


    st.warning(
        f"""
🟠 **High capacity pressure**

{len(high_capacity)} forecast day(s) exceed staffed-bed capacity.

The first high-risk date is **{first_high_date}**.
"""
    )


elif not moderate_capacity.empty:

    st.warning(
        f"""
🟡 **Moderate capacity pressure**

{len(moderate_capacity)} forecast day(s) are expected to operate at or above 90% of staffed capacity.
"""
    )


else:

    st.success(
        """
🟢 No significant bed-capacity pressure is forecast for this unit.
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
👩‍⚕️ **Staffing risk alert**

The current staffing forecast is **{staffing_status_raw}**. Review planned and actual staffing levels for this unit.
"""
    )


# ==========================================
# CAPACITY RISK TABLE
# ==========================================

st.subheader("Capacity Risk Detail")


capacity_display_columns = [
    column
    for column in [
        "date",
        "forecast_occupied_beds",
        "staffed_beds",
        "total_beds",
        "occupancy_rate",
        "risk_level"
    ]
    if column in capacity_df.columns
]


st.dataframe(
    capacity_df[
        capacity_display_columns
    ].sort_values(
        "date",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# STAFFING RISK TABLE
# ==========================================

st.subheader("Staffing Risk Alerts")


if not staff.empty:

    staffing_display_columns = [
        column
        for column in [
            "date",
            "predicted_occupied_beds",
            "planned_staff",
            "actual_staff",
            "staffing_ratio",
            "safe_ratio_met",
            "staffing_risk"
        ]
        if column in staff.columns
    ]


    st.dataframe(
        staff[
            staffing_display_columns
        ].sort_values(
            "date",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# DOWNLOADS
# ==========================================

capacity_csv = capacity_df.to_csv(
    index=False
)


st.download_button(
    label="Download Capacity Risk CSV",
    data=capacity_csv,
    file_name=(
        f"{hospital}_"
        f"{ward.replace(' ', '_')}_"
        f"capacity_risk.csv"
    ),
    mime="text/csv"
)


if not staff.empty:

    staffing_csv = staff.to_csv(
        index=False
    )


    st.download_button(
        label="Download Staffing Risk CSV",
        data=staffing_csv,
        file_name=(
            f"{hospital}_"
            f"{ward.replace(' ', '_')}_"
            f"staffing_risk.csv"
        ),
        mime="text/csv"
    )