"""
AirPulse — FastAPI Production Serving Application.

Provides real-time AQI forecasting endpoints and serves the static Web Dashboard UI.
Loads Production model from MLflow Registry with local artifact fallback.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

from src.config import CITIES, MLFLOW_TRACKING_URI, MODEL_REGISTRY_NAME
from src.features.dataset_loader import load_dataset
from src.features.feature_builder import build_features, get_feature_columns
from src.serving.schemas import (
    CityInfo,
    CustomPredictionRequest,
    CustomPredictionResponse,
    ForecastResponse,
    HealthResponse,
    HourlyForecastPoint,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("airpulse.serving")

# Global Model State
class ModelManager:
    def __init__(self):
        self.model = None
        self.version = "1"
        self.stage = "LocalFallback"
        self.run_id = "N/A"
        self.feature_names: List[str] = []

    def load_model(self):
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient()

            # Attempt to find Production stage or latest registered model
            versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
            if versions:
                # Find Production stage or highest version
                prod_ver = next((v for v in versions if v.current_stage == "Production"), versions[0])
                self.version = str(prod_ver.version)
                self.stage = prod_ver.current_stage
                self.run_id = prod_ver.run_id

                model_uri = f"models:/{MODEL_REGISTRY_NAME}/{self.version}"
                logger.info("Loading MLflow model from %s (stage=%s)...", model_uri, self.stage)
                self.model = mlflow.lightgbm.load_model(model_uri)
                logger.info("Successfully loaded MLflow model version %s!", self.version)
                return
        except Exception as exc:
            logger.warning("Could not load from MLflow registry (%s). Falling back to trained model...", exc)

        # Local Fallback
        from src.training.baseline_model import AirPulseBaselineModel
        from src.training.train import run_training_pipeline
        result = run_training_pipeline()
        self.run_id = result["run_id"]
        self.version = "1"
        self.stage = "LocalTrained"
        model_uri = f"runs:/{self.run_id}/model"
        self.model = mlflow.lightgbm.load_model(model_uri)
        logger.info("Successfully loaded fallback LightGBM model from run %s!", self.run_id)


model_mgr = ModelManager()


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    model_mgr.load_model()
    yield

# Initialize FastAPI App
app = FastAPI(
    title="AirPulse AQI Forecasting API",
    description="Self-improving AI-powered Air Quality Index forecasting service across Indian cities.",
    version="1.0.0",
    lifespan=lifespan,
)


# Helper: AQI Severity & Colors (Indian CPCB AQI standard)
def get_aqi_category(aqi: float) -> tuple[str, str]:
    if aqi <= 50:
        return "Good", "#10B981"
    elif aqi <= 100:
        return "Satisfactory", "#84CC16"
    elif aqi <= 200:
        return "Moderate", "#F59E0B"
    elif aqi <= 300:
        return "Poor", "#F97316"
    elif aqi <= 400:
        return "Very Poor", "#EF4444"
    else:
        return "Severe", "#991B1B"


# --- API Routes ---

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    if model_mgr.model is None:
        model_mgr.load_model()
    return HealthResponse(
        status="healthy",
        model_name=MODEL_REGISTRY_NAME,
        model_version=model_mgr.version,
        model_stage=model_mgr.stage,
        run_id=model_mgr.run_id,
    )


@app.get("/api/cities", response_model=List[CityInfo], tags=["Metadata"])
def get_supported_cities():
    return [
        CityInfo(
            id=key,
            name=key.title(),
            lat=cfg["lat"],
            lon=cfg["lon"],
            timezone=cfg["timezone"],
        )
        for key, cfg in CITIES.items()
    ]


@app.get("/api/forecast", response_model=ForecastResponse, tags=["Forecasting"])
def get_city_forecast(city: str = Query("mumbai", description="Target city name")):
    if model_mgr.model is None:
        model_mgr.load_model()
    city_lower = city.lower().strip()
    if city_lower not in CITIES:
        raise HTTPException(status_code=400, detail=f"City '{city}' not supported. Choose from {list(CITIES.keys())}")

    # Load dataset & build features
    df_raw = load_dataset()
    target_col = "target_aqi_lead24h"
    df_features = build_features(df_raw, forecast_horizon=24)

    # Filter city data
    city_df = df_features[df_features["city"] == city_lower].sort_values("timestamp").reset_index(drop=True)
    if city_df.empty:
        raise HTTPException(status_code=404, detail=f"No feature records available for city '{city}'")

    # Predict on available feature rows
    feature_cols = get_feature_columns(city_df, target_col)
    X_city = city_df[feature_cols]
    preds = model_mgr.model.predict(X_city)

    # Take the latest record as current
    latest_row = city_df.iloc[-1]
    curr_aqi = round(float(latest_row["aqi"]), 1)
    curr_cat, curr_col = get_aqi_category(curr_aqi)

    # Build 24-hour forecast trajectory points
    forecast_points = []
    base_time = pd.to_datetime(latest_row["timestamp"])

    # Extract recent predictions window
    n_points = min(24, len(preds))
    recent_preds = preds[-n_points:]
    recent_rows = city_df.iloc[-n_points:]

    for idx in range(n_points):
        point_time = (base_time + timedelta(hours=idx + 1)).strftime("%Y-%m-%dT%H:00:00Z")
        p_aqi = max(10.0, round(float(recent_preds[idx]), 1))
        p_cat, p_col = get_aqi_category(p_aqi)
        r = recent_rows.iloc[idx]

        forecast_points.append(
            HourlyForecastPoint(
                timestamp=point_time,
                aqi=p_aqi,
                category=p_cat,
                color=p_col,
                temperature_2m=round(float(r["temperature_2m"]), 1),
                relative_humidity_2m=round(float(r["relative_humidity_2m"]), 1),
                wind_speed_10m=round(float(r["wind_speed_10m"]), 1),
                precipitation=round(float(r["precipitation"]), 2),
            )
        )

    return ForecastResponse(
        city=city_lower,
        city_display=city_lower.title(),
        fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        current_aqi=curr_aqi,
        current_category=curr_cat,
        current_color=curr_col,
        forecast=forecast_points,
    )


# Mount Static Files for Dashboard UI
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/", response_class=FileResponse, tags=["UI"])
def serve_dashboard():
    index_file = static_path / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    raise HTTPException(status_code=404, detail="Dashboard UI index.html not found")
