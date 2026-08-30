"""
Unit and Integration Tests for FastAPI Serving Endpoints.
"""

import lightgbm as lgb
import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport
from src.serving.app import app, model_mgr


@pytest.fixture(autouse=True)
def setup_test_model():
    """Ensure API tests execute with an in-memory LightGBM model for speed and isolation."""
    if model_mgr.model is None:
        X_dummy = np.random.rand(10, 37)
        y_dummy = np.random.rand(10) * 100
        train_data = lgb.Dataset(X_dummy, label=y_dummy)
        model_mgr.model = lgb.train({"verbosity": -1}, train_data, num_boost_round=5)
        model_mgr.version = "1"
        model_mgr.stage = "TestFixture"


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_name" in data
        assert "model_version" in data


@pytest.mark.asyncio
async def test_get_cities_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/cities")
        assert response.status_code == 200
        cities = response.json()
        assert len(cities) == 5
        city_ids = [c["id"] for c in cities]
        assert "mumbai" in city_ids
        assert "delhi" in city_ids


@pytest.mark.asyncio
async def test_get_forecast_endpoint_valid_city():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/forecast?city=mumbai")
        assert response.status_code == 200
        data = response.json()
        assert data["city"] == "mumbai"
        assert data["current_aqi"] > 0
        assert "current_category" in data
        assert len(data["forecast"]) > 0
        first_point = data["forecast"][0]
        assert "aqi" in first_point
        assert "temperature_2m" in first_point


@pytest.mark.asyncio
async def test_get_forecast_endpoint_invalid_city():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/forecast?city=invalid_city")
        assert response.status_code == 400
