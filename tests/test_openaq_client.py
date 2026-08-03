"""
Tests for OpenAQ v3 API client.
"""

import pytest
import respx
import httpx
from src.ingestion.openaq_client import (
    fetch_locations,
    fetch_sensor_measurements,
    fetch_city_air_quality,
    OpenAQError,
)

SAMPLE_LOCATIONS_RESPONSE = {
    "meta": {"found": 1},
    "results": [
        {
            "id": 100,
            "name": "Bandra, Mumbai",
            "locality": "Mumbai",
            "sensors": [
                {"id": 1001, "name": "PM2.5"},
                {"id": 1002, "name": "PM10"},
            ],
        }
    ],
}

SAMPLE_MEASUREMENTS_RESPONSE = {
    "meta": {"found": 1},
    "results": [
        {
            "period": {"datetimeFrom": {"utc": "2026-08-03T10:00:00Z"}},
            "value": 45.2,
            "parameter": {"name": "pm25", "units": "µg/m³"},
        }
    ],
}


@pytest.mark.asyncio
@respx.mock
async def test_fetch_locations_success():
    respx.get("https://api.openaq.org/v3/locations").mock(
        return_value=httpx.Response(200, json=SAMPLE_LOCATIONS_RESPONSE)
    )

    async with httpx.AsyncClient() as client:
        locations = await fetch_locations(client, 19.076, 72.8777)

    assert len(locations) == 1
    assert locations[0]["id"] == 100
    assert len(locations[0]["sensors"]) == 2


@pytest.mark.asyncio
@respx.mock
async def test_fetch_sensor_measurements_success():
    respx.get("https://api.openaq.org/v3/sensors/1001/measurements/hourly").mock(
        return_value=httpx.Response(200, json=SAMPLE_MEASUREMENTS_RESPONSE)
    )

    async with httpx.AsyncClient() as client:
        measurements = await fetch_sensor_measurements(client, 1001)

    assert len(measurements) == 1
    assert measurements[0]["value"] == 45.2


@pytest.mark.asyncio
@respx.mock
async def test_fetch_city_air_quality():
    respx.get("https://api.openaq.org/v3/locations").mock(
        return_value=httpx.Response(200, json=SAMPLE_LOCATIONS_RESPONSE)
    )
    respx.get("https://api.openaq.org/v3/sensors/1001/measurements/hourly").mock(
        return_value=httpx.Response(200, json=SAMPLE_MEASUREMENTS_RESPONSE)
    )
    respx.get("https://api.openaq.org/v3/sensors/1002/measurements/hourly").mock(
        return_value=httpx.Response(200, json=SAMPLE_MEASUREMENTS_RESPONSE)
    )

    async with httpx.AsyncClient() as client:
        result = await fetch_city_air_quality(client, "mumbai", 19.076, 72.8777)

    assert result["city"] == "mumbai"
    assert len(result["locations"]) == 1
    assert "1001" in result["measurements"]
    assert "1002" in result["measurements"]
