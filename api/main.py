from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.forecast_service import (
    get_capacity_risk,
    get_dashboard_summary,
    get_forecast,
    get_hospitals,
    get_occupancy,
    get_patient_flow,
    get_staffing_risk,
    get_wards,
    get_xgb_forecast
)

from api.schemas import (
    CapacityRiskRecord,
    DashboardSummary,
    ForecastRecord,
    HealthResponse,
    OccupancyRecord,
    PatientFlowRecord,
    StaffingRecord,
    XGBoostForecastRecord
)


tags_metadata = [
    {
        "name": "System",
        "description": "API health and service information."
    },
    {
        "name": "Metadata",
        "description": "Hospital and ward values used by dashboard filters."
    },
    {
        "name": "Forecasting",
        "description": "Future bed occupancy predictions."
    },
    {
        "name": "Operations",
        "description": "Occupancy, patient flow and capacity intelligence."
    },
    {
        "name": "Workforce",
        "description": "Staffing forecasts and feasibility risk."
    }
]


app = FastAPI(
    title="Albion Care Network Bed Intelligence API",
    description="""
API service supporting the Albion Care Network hospital operations
dashboard.

The service exposes bed occupancy forecasts, patient-flow metrics,
capacity-risk indicators and staffing-feasibility insights.
""",
    version="1.0.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"]
)


@app.get(
    "/",
    response_model=HealthResponse,
    tags=["System"],
    summary="API service status"
)
def root():

    return {
        "service": "Albion Forecast API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check"
)
def health():

    return {
        "service": "Albion Forecast API",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get(
    "/hospitals",
    response_model=list[str],
    tags=["Metadata"],
    summary="List available hospitals"
)
def hospitals():

    return get_hospitals()


@app.get(
    "/wards/{hospital_id}",
    response_model=list[str],
    tags=["Metadata"],
    summary="List wards for a hospital"
)
def wards(hospital_id: str):

    result = get_wards(hospital_id)

    if not result:
        raise HTTPException(
            status_code=404,
            detail="Hospital not found or has no wards."
        )

    return result


@app.get(
    "/forecast/{hospital_id}/{ward}",
    response_model=list[ForecastRecord],
    tags=["Forecasting"],
    summary="Get 30-day SARIMAX forecast"
)
def forecast(
    hospital_id: str,
    ward: str
):

    result = get_forecast(
        hospital_id,
        ward
    )

    if not result:
        raise HTTPException(
            status_code=404,
            detail="No forecast found for this hospital and ward."
        )

    return result


@app.get(
    "/xgb-forecast/{hospital_id}/{ward}",
    response_model=list[XGBoostForecastRecord],
    tags=["Forecasting"],
    summary="Get 30-day XGBoost forecast"
)
def xgb_forecast(
    hospital_id: str,
    ward: str
):

    result = get_xgb_forecast(
        hospital_id,
        ward
    )

    if not result:
        raise HTTPException(
            status_code=404,
            detail="No XGBoost forecast found for this hospital and ward."
        )

    return result


@app.get(
    "/staffing/{hospital_id}/{ward}",
    response_model=list[StaffingRecord],
    tags=["Workforce"],
    summary="Get staffing-risk forecast"
)
def staffing(
    hospital_id: str,
    ward: str
):

    result = get_staffing_risk(
        hospital_id,
        ward
    )

    if not result:
        raise HTTPException(
            status_code=404,
            detail="No staffing data found."
        )

    return result


@app.get(
    "/occupancy/{hospital_id}/{ward}",
    response_model=list[OccupancyRecord],
    tags=["Operations"],
    summary="Get recent occupancy history"
)
def occupancy(
    hospital_id: str,
    ward: str
):

    result = get_occupancy(
        hospital_id,
        ward
    )

    if not result:
        raise HTTPException(
            status_code=404,
            detail="No occupancy data found."
        )

    return result


@app.get(
    "/capacity-risk/{hospital_id}/{ward}",
    response_model=list[CapacityRiskRecord],
    tags=["Operations"],
    summary="Get forecast capacity risk"
)
def capacity_risk(
    hospital_id: str,
    ward: str
):

    result = get_capacity_risk(
        hospital_id,
        ward
    )

    if not result:
        raise HTTPException(
            status_code=404,
            detail="No capacity-risk data found."
        )

    return result


@app.get(
    "/patient-flow/{hospital_id}/{ward}",
    response_model=list[PatientFlowRecord],
    tags=["Operations"],
    summary="Get recent patient-flow activity"
)
def patient_flow(
    hospital_id: str,
    ward: str
):

    result = get_patient_flow(
        hospital_id,
        ward
    )

    if not result:
        raise HTTPException(
            status_code=404,
            detail="No patient-flow data found."
        )

    return result


@app.get(
    "/dashboard-summary/{hospital_id}/{ward}",
    response_model=DashboardSummary,
    tags=["Operations"],
    summary="Get executive dashboard summary"
)
def dashboard_summary(
    hospital_id: str,
    ward: str
):

    result = get_dashboard_summary(
        hospital_id,
        ward
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Dashboard summary unavailable."
        )

    return result