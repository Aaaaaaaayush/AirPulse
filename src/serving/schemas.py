"""
AirPulse — FastAPI Pydantic Request & Response Schemas.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    model_name: str = Field(..., json_schema_extra={"example": "airpulse-forecaster"})
    model_version: str = Field(..., json_schema_extra={"example": "2"})
    model_stage: str = Field(..., json_schema_extra={"example": "Production"})
    run_id: Optional[str] = Field(None, json_schema_extra={"example": "42aa50d09c4f4d2da0c89a92f077ef4e"})


class CityInfo(BaseModel):
    id: str = Field(..., json_schema_extra={"example": "mumbai"})
    name: str = Field(..., json_schema_extra={"example": "Mumbai"})
    lat: float = Field(..., json_schema_extra={"example": 19.0760})
    lon: float = Field(..., json_schema_extra={"example": 72.8777})
    timezone: str = Field(..., json_schema_extra={"example": "Asia/Kolkata"})


class HourlyForecastPoint(BaseModel):
    timestamp: str = Field(..., json_schema_extra={"example": "2026-08-12T18:00:00Z"})
    aqi: float = Field(..., json_schema_extra={"example": 124.5})
    category: str = Field(..., json_schema_extra={"example": "Moderate"})
    color: str = Field(..., json_schema_extra={"example": "#F59E0B"})
    temperature_2m: float = Field(..., json_schema_extra={"example": 28.5})
    relative_humidity_2m: float = Field(..., json_schema_extra={"example": 75.0})
    wind_speed_10m: float = Field(..., json_schema_extra={"example": 14.2})
    precipitation: float = Field(..., json_schema_extra={"example": 0.0})


class ForecastResponse(BaseModel):
    city: str = Field(..., json_schema_extra={"example": "mumbai"})
    city_display: str = Field(..., json_schema_extra={"example": "Mumbai"})
    fetched_at: str = Field(..., json_schema_extra={"example": "2026-08-12T16:00:00Z"})
    current_aqi: float = Field(..., json_schema_extra={"example": 118.2})
    current_category: str = Field(..., json_schema_extra={"example": "Moderate"})
    current_color: str = Field(..., json_schema_extra={"example": "#F59E0B"})
    forecast: List[HourlyForecastPoint]


class CustomPredictionRequest(BaseModel):
    city: str = Field("mumbai", json_schema_extra={"example": "mumbai"})
    aqi_lag_1h: float = Field(120.0, json_schema_extra={"example": 120.0})
    aqi_lag_2h: float = Field(118.0, json_schema_extra={"example": 118.0})
    aqi_lag_24h: float = Field(125.0, json_schema_extra={"example": 125.0})
    temperature_2m: float = Field(30.0, json_schema_extra={"example": 30.0})
    relative_humidity_2m: float = Field(70.0, json_schema_extra={"example": 70.0})
    wind_speed_10m: float = Field(12.0, json_schema_extra={"example": 12.0})
    precipitation: float = Field(0.0, json_schema_extra={"example": 0.0})


class CustomPredictionResponse(BaseModel):
    predicted_aqi: float = Field(..., json_schema_extra={"example": 122.4})
    category: str = Field(..., json_schema_extra={"example": "Moderate"})
    color: str = Field(..., json_schema_extra={"example": "#F59E0B"})
