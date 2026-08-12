"""
Unit and Integration Tests for FastAPI Serving Endpoints.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from src.serving.app import app


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
