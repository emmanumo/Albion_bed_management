import os
import pandas as pd
import requests


BASE_URL = os.getenv(
    "ALBION_API_URL",
    "http://127.0.0.1:8000"
)


def _get_json(endpoint):

    response = requests.get(endpoint)

    response.raise_for_status()

    return response.json()


def _get_dataframe(endpoint):

    return pd.DataFrame(
        _get_json(endpoint)
    )


def get_forecast(hospital_id, ward):

    return _get_dataframe(
        f"{BASE_URL}/forecast/{hospital_id}/{ward}"
    )


def get_staffing(hospital_id, ward):

    return _get_dataframe(
        f"{BASE_URL}/staffing/{hospital_id}/{ward}"
    )


def get_capacity_risk(hospital_id, ward):

    return _get_dataframe(
        f"{BASE_URL}/capacity-risk/{hospital_id}/{ward}"
    )


def get_occupancy(hospital_id, ward):

    return _get_dataframe(
        f"{BASE_URL}/occupancy/{hospital_id}/{ward}"
    )


def get_patient_flow(hospital_id, ward):

    return _get_dataframe(
        f"{BASE_URL}/patient-flow/{hospital_id}/{ward}"
    )


def get_dashboard_summary(hospital_id, ward):

    return _get_json(
        f"{BASE_URL}/dashboard-summary/{hospital_id}/{ward}"
    )


def get_hospitals():

    return _get_json(
        f"{BASE_URL}/hospitals"
    )


def get_wards(hospital_id):

    return _get_json(
        f"{BASE_URL}/wards/{hospital_id}"
    )

def get_xgb_forecast(
    hospital_id,
    ward
):

    return _get_dataframe(
        f"{BASE_URL}/xgb-forecast/{hospital_id}/{ward}"
    )
