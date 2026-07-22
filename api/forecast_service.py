from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================
# FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"

MODEL_PATH = (
    DATA_DIR
    / "bed_occupancy_modelling_dataset.csv"
)

FORECAST_PATH = (
    DATA_DIR
    / "sarimax_future_forecast.csv"
)

XGB_PATH = (
    DATA_DIR
    / "xgb_future_forecast.csv"
)

STAFF_PATH = (
    DATA_DIR
    / "staffing_risk_forecast.csv"
)


# ==========================================
# JSON-SAFE DATAFRAME CONVERSION
# ==========================================

def _prepare_records(
    dataframe: pd.DataFrame
) -> list[dict]:
    """
    Convert a dataframe into JSON-safe records.

    Dates are converted to YYYY-MM-DD strings.
    NaN and infinite values are converted to None.
    """

    result = dataframe.copy()

    if "date" in result.columns:

        result["date"] = pd.to_datetime(
            result["date"],
            errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    )

    result = result.where(
        pd.notna(result),
        None
    )

    return result.to_dict(
        orient="records"
    )


# ==========================================
# DATA LOADERS
# ==========================================

@lru_cache(maxsize=1)
def load_operations() -> pd.DataFrame:
    """
    Load the operational modelling dataset.
    """

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Operational dataset not found: "
            f"{MODEL_PATH}"
        )

    operations = pd.read_csv(
        MODEL_PATH
    )

    operations["date"] = pd.to_datetime(
        operations["date"],
        errors="coerce"
    )

    return operations


@lru_cache(maxsize=1)
def load_forecasts() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame
]:
    """
    Load SARIMAX, XGBoost and staffing
    forecast datasets.
    """

    required_files = [
        FORECAST_PATH,
        XGB_PATH,
        STAFF_PATH
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Forecast dataset not found: {path}"
            )

    sarimax = pd.read_csv(
        FORECAST_PATH
    )

    xgb = pd.read_csv(
        XGB_PATH
    )

    staffing = pd.read_csv(
        STAFF_PATH
    )

    for dataframe in [
        sarimax,
        xgb,
        staffing
    ]:

        dataframe["date"] = pd.to_datetime(
            dataframe["date"],
            errors="coerce"
        )

    return (
        sarimax,
        xgb,
        staffing
    )


# ==========================================
# METADATA SERVICES
# ==========================================

def get_hospitals() -> list[str]:
    """
    Return all available hospital IDs.
    """

    operations = load_operations()

    return sorted(
        operations["hospital_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def get_wards(
    hospital_id: str
) -> list[str]:
    """
    Return wards belonging to a hospital.
    """

    operations = load_operations()

    wards = operations.loc[
        operations["hospital_id"]
        == hospital_id,
        "ward"
    ]

    return sorted(
        wards
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


# ==========================================
# SARIMAX FORECAST SERVICE
# ==========================================

def get_forecast(
    hospital_id: str,
    ward: str
) -> list[dict]:
    """
    Return the future SARIMAX forecast
    for one hospital and ward.
    """

    sarimax, _, _ = load_forecasts()

    result = sarimax[
        (
            sarimax["hospital_id"]
            == hospital_id
        )
        &
        (
            sarimax["ward"]
            == ward
        )
    ].sort_values(
        "date"
    ).copy()

    return _prepare_records(
        result
    )


# ==========================================
# XGBOOST FORECAST SERVICE
# ==========================================

def get_xgb_forecast(
    hospital_id: str,
    ward: str
) -> list[dict]:
    """
    Return the future XGBoost forecast
    for one hospital and ward.
    """

    _, xgb, _ = load_forecasts()

    result = xgb[
        (
            xgb["hospital_id"]
            == hospital_id
        )
        &
        (
            xgb["ward"]
            == ward
        )
    ].sort_values(
        "date"
    ).copy()

    return _prepare_records(
        result
    )


# ==========================================
# STAFFING-RISK SERVICE
# ==========================================

def get_staffing_risk(
    hospital_id: str,
    ward: str
) -> list[dict]:
    """
    Return staffing forecasts and risk
    indicators for one hospital and ward.
    """

    _, _, staffing = load_forecasts()

    result = staffing[
        (
            staffing["hospital_id"]
            == hospital_id
        )
        &
        (
            staffing["ward"]
            == ward
        )
    ].sort_values(
        "date"
    ).copy()

    return _prepare_records(
        result
    )


# ==========================================
# OCCUPANCY SERVICE
# ==========================================

def get_occupancy(
    hospital_id: str,
    ward: str
) -> list[dict]:
    """
    Return the latest 30 operational
    occupancy records.
    """

    operations = load_operations()

    result = operations[
        (
            operations["hospital_id"]
            == hospital_id
        )
        &
        (
            operations["ward"]
            == ward
        )
    ].sort_values(
        "date"
    ).copy()

    columns = [
        "date",
        "total_beds",
        "staffed_beds",
        "occupied_beds",
        "occupancy_rate",
        "available_beds"
    ]

    available_columns = [
        column
        for column in columns
        if column in result.columns
    ]

    return _prepare_records(
        result[
            available_columns
        ].tail(30)
    )


# ==========================================
# PATIENT-FLOW SERVICE
# ==========================================

def get_patient_flow(
    hospital_id: str,
    ward: str
) -> list[dict]:
    """
    Return the latest 30 patient-flow
    records and calculate net flow.
    """

    operations = load_operations()

    result = operations[
        (
            operations["hospital_id"]
            == hospital_id
        )
        &
        (
            operations["ward"]
            == ward
        )
    ].sort_values(
        "date"
    ).copy()

    if result.empty:

        return []

    result["daily_admissions"] = pd.to_numeric(
        result["daily_admissions"],
        errors="coerce"
    ).fillna(0)

    result["daily_discharges"] = pd.to_numeric(
        result["daily_discharges"],
        errors="coerce"
    ).fillna(0)

    result["daily_ed_arrivals"] = pd.to_numeric(
        result["daily_ed_arrivals"],
        errors="coerce"
    ).fillna(0)

    result["net_flow"] = (
        result["daily_admissions"]
        -
        result["daily_discharges"]
    )

    columns = [
        "date",
        "daily_admissions",
        "daily_discharges",
        "daily_ed_arrivals",
        "net_flow"
    ]

    return _prepare_records(
        result[columns].tail(30)
    )


# ==========================================
# CAPACITY-RISK SERVICE
# ==========================================

def get_capacity_risk(
    hospital_id: str,
    ward: str
) -> list[dict]:
    """
    Compare forecast demand with staffed
    and total bed capacity.
    """

    sarimax, _, _ = load_forecasts()

    operations = load_operations()

    forecast = sarimax[
        (
            sarimax["hospital_id"]
            == hospital_id
        )
        &
        (
            sarimax["ward"]
            == ward
        )
    ].sort_values(
        "date"
    ).copy()

    unit_operations = operations[
        (
            operations["hospital_id"]
            == hospital_id
        )
        &
        (
            operations["ward"]
            == ward
        )
    ].sort_values(
        "date"
    ).copy()

    if (
        forecast.empty
        or unit_operations.empty
    ):

        return []

    latest_capacity = (
        unit_operations.iloc[-1]
    )

    latest_total_beds = pd.to_numeric(
        latest_capacity.get(
            "total_beds",
            0
        ),
        errors="coerce"
    )

    latest_staffed_beds = pd.to_numeric(
        latest_capacity.get(
            "staffed_beds",
            0
        ),
        errors="coerce"
    )

    if pd.isna(latest_total_beds):

        latest_total_beds = 0

    if pd.isna(latest_staffed_beds):

        latest_staffed_beds = 0

    beds = (
        unit_operations[
            [
                "date",
                "total_beds",
                "staffed_beds"
            ]
        ]
        .drop_duplicates(
            subset=["date"]
        )
    )

    risk = forecast.merge(
        beds,
        on="date",
        how="left"
    )

    risk["total_beds"] = (
        pd.to_numeric(
            risk["total_beds"],
            errors="coerce"
        )
        .fillna(latest_total_beds)
    )

    risk["staffed_beds"] = (
        pd.to_numeric(
            risk["staffed_beds"],
            errors="coerce"
        )
        .fillna(latest_staffed_beds)
    )

    risk["forecast_occupied_beds"] = (
        pd.to_numeric(
            risk["forecast_occupied_beds"],
            errors="coerce"
        )
    )

    risk["occupancy_rate"] = np.where(

        risk["staffed_beds"] > 0,

        risk["forecast_occupied_beds"]
        /
        risk["staffed_beds"],

        np.nan

    )

    risk["risk_level"] = np.select(
        [
            risk["forecast_occupied_beds"]
            >
            risk["total_beds"],

            risk["forecast_occupied_beds"]
            >
            risk["staffed_beds"],

            risk["occupancy_rate"]
            >= 0.90
        ],
        [
            "Critical",
            "High",
            "Moderate"
        ],
        default="Normal"
    )

    columns = [
        "date",
        "forecast_occupied_beds",
        "staffed_beds",
        "total_beds",
        "occupancy_rate",
        "risk_level"
    ]

    return _prepare_records(
        risk[columns]
    )


# ==========================================
# EXECUTIVE DASHBOARD SUMMARY
# ==========================================

def get_dashboard_summary(
    hospital_id: str,
    ward: str
) -> dict | None:
    """
    Return the latest executive summary
    for a hospital ward.
    """

    forecast_records = get_forecast(
        hospital_id,
        ward
    )

    capacity_records = get_capacity_risk(
        hospital_id,
        ward
    )

    staffing_records = get_staffing_risk(
        hospital_id,
        ward
    )

    if (
        not forecast_records
        or not capacity_records
    ):

        return None

    latest_forecast = (
        forecast_records[-1]
    )

    latest_capacity = (
        capacity_records[-1]
    )

    staffing_risk = "Unknown"

    if staffing_records:

        staffing_risk = (
            staffing_records[-1]
            .get("staffing_risk")
            or "Unknown"
        )

    total_beds = (
        latest_capacity.get(
            "total_beds"
        )
        or 0
    )

    staffed_beds = (
        latest_capacity.get(
            "staffed_beds"
        )
        or 0
    )

    forecast_beds = (
        latest_forecast.get(
            "forecast_occupied_beds"
        )
        or 0
    )

    occupancy_rate = (
        forecast_beds / staffed_beds
        if staffed_beds > 0
        else 0
    )

    available_beds = (
        total_beds
        -
        forecast_beds
    )

    return {
        "hospital_id": hospital_id,
        "ward": ward,
        "forecast_beds": float(
            forecast_beds
        ),
        "total_beds": float(
            total_beds
        ),
        "staffed_beds": float(
            staffed_beds
        ),
        "occupancy_rate": float(
            occupancy_rate
        ),
        "available_beds": float(
            available_beds
        ),
        "capacity_risk": (
            latest_capacity.get(
                "risk_level"
            )
            or "Unknown"
        ),
        "staffing_risk": (
            staffing_risk
        ),
        "forecast_days": len(
            forecast_records
        )
    }