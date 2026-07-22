from datetime import date
from pydantic import BaseModel


class HealthResponse(BaseModel):
    service: str
    status: str
    version: str


class ForecastRecord(BaseModel):
    hospital_id: str
    ward: str
    date: date
    forecast_occupied_beds: float

class XGBoostForecastRecord(BaseModel):
    hospital_id: str
    ward: str
    date: date
    predicted_occupied_beds: float

class StaffingRecord(BaseModel):
    hospital_id: str
    ward: str
    date: date
    predicted_occupied_beds: float
    planned_staff: float | None = None
    actual_staff: float | None = None
    safe_ratio_met: str | None = None
    staffing_ratio: float | None = None
    staffing_risk: str | None = None


class OccupancyRecord(BaseModel):
    date: date
    total_beds: float | None = None
    staffed_beds: float | None = None
    occupied_beds: float | None = None
    occupancy_rate: float | None = None
    available_beds: float | None = None


class CapacityRiskRecord(BaseModel):
    date: date
    forecast_occupied_beds: float
    staffed_beds: float | None = None
    total_beds: float | None = None
    occupancy_rate: float | None = None
    risk_level: str


class PatientFlowRecord(BaseModel):
    date: date
    daily_admissions: float | None = None
    daily_discharges: float | None = None
    daily_ed_arrivals: float | None = None
    net_flow: float | None = None


class DashboardSummary(BaseModel):
    hospital_id: str
    ward: str
    forecast_beds: float
    total_beds: float
    staffed_beds: float
    occupancy_rate: float
    available_beds: float
    capacity_risk: str
    staffing_risk: str
    forecast_days: int