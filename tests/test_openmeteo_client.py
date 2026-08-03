"""
Tests for Open-Meteo API client and hour-splitter function.
"""

import pytest
import respx
import httpx
from src.ingestion.openmeteo_client import (
    fetch_city_weather,
    split_by_hour,
    OpenMeteoError,
)

SAMPLE_WEATHER_RESPONSE = {
    "latitude": 19.076,
    "longitude": 72.8777,
    "timezone": "Asia/Kolkata",
    "hourly_units": {
        "time": "iso8601",
        "temperature_2m": "°C",
        "relative_humidity_2m": "%",
        "wind_speed_10m": "km/h",
        "wind_direction_10m": "°",
        "precipitation": "mm",
    },
    "hourly": {
        "time": ["2026-08-03T10:00", "2026-08-03T11:00"],
        "temperature_2m": [28.4, 28.8],
        "relative_humidity_2m": [79, 79],
        "wind_speed_10m": [13.2, 14.2],
        "wind_direction_10m": [282, 277],
        "precipitation": [0.4, 0.2],
    },
}


@pytest.mark.asyncio
@respx.mock
async def test_fetch_city_weather_success():
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=SAMPLE_WEATHER_RESPONSE)
    )

    async with httpx.AsyncClient() as client:
        data = await fetch_city_weather(client, "mumbai", 19.076, 72.8777)

    assert data["city"] == "mumbai"
    assert "fetched_at" in data
    assert len(data["hourly"]["time"]) == 2
    assert data["hourly"]["temperature_2m"][0] == 28.4


@pytest.mark.asyncio
@respx.mock
async def test_fetch_city_weather_failure():
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(400, text="Bad Request")
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(OpenMeteoError):
            await fetch_city_weather(client, "mumbai", 19.076, 72.8777)


def test_split_by_hour():
    weather_data = {
        "city": "mumbai",
        "fetched_at": "2026-08-03T12:00:00Z",
        **SAMPLE_WEATHER_RESPONSE,
    }

    hourly_docs = split_by_hour(weather_data)

    assert len(hourly_docs) == 2
    assert "2026-08-03T10:00" in hourly_docs
    assert "2026-08-03T11:00" in hourly_docs

    doc1 = hourly_docs["2026-08-03T10:00"]
    assert doc1["city"] == "mumbai"
    assert doc1["temperature_2m"] == 28.4
    assert doc1["precipitation"] == 0.4
    assert doc1["units"]["temperature_2m"] == "°C"
