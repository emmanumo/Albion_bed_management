import requests
import pandas as pd

BASE_URL = "http://127.0.0.1:8000"


def get_forecast(hospital, ward):

    response = requests.get(
        f"{BASE_URL}/forecast/{hospital}/{ward}"
    )

    response.raise_for_status()

    return pd.DataFrame(response.json())


def get_staffing(hospital, ward):

    response = requests.get(
        f"{BASE_URL}/staffing/{hospital}/{ward}"
    )

    response.raise_for_status()

    return pd.DataFrame(response.json())


def get_capacity(hospital, ward):

    response = requests.get(
        f"{BASE_URL}/capacity-risk/{hospital}/{ward}"
    )

    response.raise_for_status()

    return pd.DataFrame(response.json())


def get_patient_flow(hospital, ward):

    response = requests.get(
        f"{BASE_URL}/patient-flow/{hospital}/{ward}"
    )

    response.raise_for_status()

    return pd.DataFrame(response.json())


def get_occupancy(hospital, ward):

    response = requests.get(
        f"{BASE_URL}/occupancy/{hospital}/{ward}"
    )

    response.raise_for_status()

    return pd.DataFrame(response.json())