import pandas as pd
import plotly.express as px
import streamlit as st

from api.client import get_patient_flow


st.set_page_config(
    page_title="Patient Flow",
    layout="wide"
)


# ==========================================
# PAGE TITLE
# ==========================================

st.title("Patient Flow Analysis")

st.markdown(
    """
Monitor how patients move through the selected hospital unit:

- Admissions and discharges
- Emergency department arrivals
- Net patient flow
- Length of stay
- Elective surgery demand
"""
)


# ==========================================
# VALIDATE SESSION STATE
# ==========================================

required_items = [
    "hospital",
    "ward",
    "model_df"
]

missing_items = [
    item
    for item in required_items
    if item not in st.session_state
]

if missing_items:

    st.error(
        "Required dashboard data is unavailable. "
        "Please start from the main dashboard."
    )

    st.stop()


# ==========================================
# LOAD FILTERS
# ==========================================

hospital = st.session_state["hospital"]

ward = st.session_state["ward"]

model_df = st.session_state["model_df"].copy()


# ==========================================
# LOAD PATIENT FLOW FROM FASTAPI
# ==========================================

try:

    flow_df = get_patient_flow(
        hospital,
        ward
    )

except Exception as error:

    st.error(
        f"Unable to retrieve patient-flow data "
        f"from FastAPI: {error}"
    )

    st.info(
        "Confirm that FastAPI is running with: "
        "`uvicorn api.main:app --reload`"
    )

    st.stop()


if flow_df.empty:

    st.warning(
        "No patient-flow data is available for "
        "the selected hospital and ward."
    )

    st.stop()


# ==========================================
# PREPARE API DATA
# ==========================================

flow_df["date"] = pd.to_datetime(
    flow_df["date"],
    errors="coerce"
)


numeric_columns = [
    "daily_admissions",
    "daily_discharges",
    "daily_ed_arrivals",
    "net_flow"
]


for column in numeric_columns:

    if column in flow_df.columns:

        flow_df[column] = pd.to_numeric(
            flow_df[column],
            errors="coerce"
        ).fillna(0)


flow_df = (
    flow_df
    .dropna(subset=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)


if flow_df.empty:

    st.warning(
        "The patient-flow endpoint returned "
        "no valid dated records."
    )

    st.stop()


# ==========================================
# LOAD SUPPLEMENTARY LOCAL DATA
# ==========================================

model_df["date"] = pd.to_datetime(
    model_df["date"],
    errors="coerce"
)


supplementary_df = model_df[
    (model_df["hospital_id"] == hospital)
    &
    (model_df["ward"] == ward)
].copy()


supplementary_columns = [
    column
    for column in [
        "date",
        "avg_los_hours",
        "scheduled_surgeries"
    ]
    if column in supplementary_df.columns
]


if supplementary_columns:

    supplementary_df = (
        supplementary_df[
            supplementary_columns
        ]
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last"
        )
    )


    flow_df = flow_df.merge(
        supplementary_df,
        on="date",
        how="left"
    )


# ==========================================
# DERIVED FLOW STATUS
# ==========================================

def classify_flow(net_flow):

    if net_flow > 0:
        return "🔴 Increasing Pressure"

    if net_flow < 0:
        return "🟢 Reducing Pressure"

    return "🟡 Balanced"


flow_df["flow_pressure"] = (
    flow_df["net_flow"]
    .apply(classify_flow)
)


# ==========================================
# PAGE CONTEXT
# ==========================================

st.caption(
    f"Hospital: **{hospital}** | "
    f"Ward: **{ward}** | "
    f"Source: **FastAPI patient-flow service**"
)


# ==========================================
# KPI CARDS
# ==========================================

st.subheader("Patient Flow Summary")

latest = flow_df.iloc[-1]


latest_admissions = (
    latest.get("daily_admissions", 0)
    or 0
)

latest_discharges = (
    latest.get("daily_discharges", 0)
    or 0
)

latest_ed_arrivals = (
    latest.get("daily_ed_arrivals", 0)
    or 0
)

latest_net_flow = (
    latest.get("net_flow", 0)
    or 0
)

latest_los = latest.get(
    "avg_los_hours",
    None
)


c1, c2, c3, c4, c5 = st.columns(5)


c1.metric(
    "Daily Admissions",
    round(float(latest_admissions), 1)
)


c2.metric(
    "Daily Discharges",
    round(float(latest_discharges), 1)
)


c3.metric(
    "ED Arrivals",
    round(float(latest_ed_arrivals), 1)
)


if pd.notna(latest_los):

    c4.metric(
        "Average LOS",
        f"{float(latest_los):.1f} hrs"
    )

else:

    c4.metric(
        "Average LOS",
        "Unavailable"
    )


c5.metric(
    "Flow Status",
    latest["flow_pressure"]
)


# ==========================================
# FLOW BALANCE KPIs
# ==========================================

st.subheader("Current Flow Balance")

b1, b2 = st.columns(2)


b1.metric(
    "Net Patient Flow",
    round(float(latest_net_flow), 1),
    help=(
        "Admissions minus discharges. "
        "A positive result indicates rising "
        "bed demand."
    )
)


if latest_net_flow > 0:

    balance_message = (
        "More patients are entering than leaving."
    )

elif latest_net_flow < 0:

    balance_message = (
        "More patients are leaving than entering."
    )

else:

    balance_message = (
        "Admissions and discharges are balanced."
    )


b2.metric(
    "Operational Interpretation",
    balance_message
)


# ==========================================
# ADMISSIONS VS DISCHARGES
# ==========================================

st.subheader("Admissions vs Discharges")


movement_fig = px.line(
    flow_df,
    x="date",
    y=[
        "daily_admissions",
        "daily_discharges"
    ],
    markers=True,
    title="Patient Movement"
)


movement_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Patients",
    legend_title_text=""
)


st.plotly_chart(
    movement_fig,
    use_container_width=True
)


# ==========================================
# EMERGENCY DEPARTMENT DEMAND
# ==========================================

st.subheader("Emergency Department Demand")


ed_fig = px.area(
    flow_df,
    x="date",
    y="daily_ed_arrivals",
    title="Daily Emergency Department Arrivals"
)


ed_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="ED Arrivals"
)


st.plotly_chart(
    ed_fig,
    use_container_width=True
)


# ==========================================
# NET FLOW
# ==========================================

st.subheader("Bed Pressure from Patient Flow")


net_fig = px.bar(
    flow_df,
    x="date",
    y="net_flow",
    title="Admissions Minus Discharges"
)


net_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Net Patient Flow"
)


st.plotly_chart(
    net_fig,
    use_container_width=True
)


# ==========================================
# LENGTH OF STAY
# ==========================================

if (
    "avg_los_hours" in flow_df.columns
    and
    flow_df["avg_los_hours"].notna().any()
):

    st.subheader("Length of Stay Trend")


    los_fig = px.line(
        flow_df,
        x="date",
        y="avg_los_hours",
        markers=True,
        title="Average Patient Length of Stay"
    )


    los_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Average LOS (hours)"
    )


    st.plotly_chart(
        los_fig,
        use_container_width=True
    )

else:

    st.info(
        "Length-of-stay information is not "
        "available for this selection."
    )


# ==========================================
# SURGERY DEMAND
# ==========================================

if (
    "scheduled_surgeries" in flow_df.columns
    and
    flow_df["scheduled_surgeries"]
    .notna()
    .any()
):

    st.subheader("Scheduled Surgery Demand")


    surgery_fig = px.line(
        flow_df,
        x="date",
        y="scheduled_surgeries",
        markers=True,
        title="Scheduled Surgery Activity"
    )


    surgery_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Scheduled Surgeries"
    )


    st.plotly_chart(
        surgery_fig,
        use_container_width=True
    )


# ==========================================
# PRESSURE SUMMARY
# ==========================================

st.subheader("Flow Pressure Summary")


pressure_summary = (
    flow_df["flow_pressure"]
    .value_counts()
    .rename_axis("flow_pressure")
    .reset_index(name="days")
)


pressure_fig = px.bar(
    pressure_summary,
    x="flow_pressure",
    y="days",
    text="days",
    title="Days by Patient-Flow Status"
)


pressure_fig.update_layout(
    xaxis_title="Flow Status",
    yaxis_title="Days"
)


st.plotly_chart(
    pressure_fig,
    use_container_width=True
)


# ==========================================
# PATIENT FLOW TABLE
# ==========================================

st.subheader("Patient Flow Detail")


display_columns = [
    column
    for column in [
        "date",
        "daily_admissions",
        "daily_discharges",
        "daily_ed_arrivals",
        "avg_los_hours",
        "scheduled_surgeries",
        "net_flow",
        "flow_pressure"
    ]
    if column in flow_df.columns
]


st.dataframe(
    flow_df[
        display_columns
    ].sort_values(
        "date",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# DOWNLOAD
# ==========================================

csv = flow_df.to_csv(
    index=False
)


st.download_button(
    label="Download Patient Flow CSV",
    data=csv,
    file_name=(
        f"{hospital}_"
        f"{ward.replace(' ', '_')}_"
        f"patient_flow.csv"
    ),
    mime="text/csv"
)