"""
AirPulse — Open-Meteo API client.

Fetches hourly weather data for configured cities.
No API key required — Open-Meteo is free for non-commercial use
(10,000 calls/day limit; we use ~120/day at hourly cadence × 5 cities).

The response includes past_days of actuals + forecast_days of predictions,
giving us a sliding window of recent history + future weather.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from src.config import (
    OPENMETEO_BASE_URL,
    OPENMETEO_FORECAST_DAYS,
    OPENMETEO_HOURLY_VARS,
    OPENMETEO_PAST_DAYS,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0
_REQUEST_TIMEOUT = 30.0


class OpenMeteoError(Exception):
    """Raised when Open-Meteo returns a non-recoverable error."""


# ---------------------------------------------------------------------------
# Low-level HTTP
# ---------------------------------------------------------------------------

async def _get_with_retry(
    client: httpx.AsyncClient,
    params: dict,
) -> dict[str, Any]:
    """GET with exponential backoff on transient errors."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = await client.get(
                OPENMETEO_BASE_URL, params=params, timeout=_REQUEST_TIMEOUT
            )

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429 or resp.status_code >= 500:
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    "Open-Meteo %s (attempt %d/%d), retrying in %.1fs …",
                    resp.status_code, attempt, _MAX_RETRIES, wait,
                )
                await asyncio.sleep(wait)
                continue

            raise OpenMeteoError(
                f"Open-Meteo API error {resp.status_code}: {resp.text[:300]}"
            )

        except httpx.RequestError as exc:
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                "Open-Meteo request error (attempt %d/%d): %s",
                attempt, _MAX_RETRIES, exc,
            )
            await asyncio.sleep(wait)

    raise OpenMeteoError(
        f"Open-Meteo request failed after {_MAX_RETRIES} retries"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def fetch_city_weather(
    client: httpx.AsyncClient,
    city_name: str,
    lat: float,
    lon: float,
    past_days: int = OPENMETEO_PAST_DAYS,
    forecast_days: int = OPENMETEO_FORECAST_DAYS,
) -> dict[str, Any]:
    """
    Fetch hourly weather for a city.

    Returns the raw Open-Meteo JSON enriched with city metadata:
        {
            "city": "mumbai",
            "fetched_at": "2026-08-03T14:00:00Z",
            "latitude": 19.086,
            "longitude": 72.853,
            "timezone": "Asia/Kolkata",
            "hourly_units": { ... },
            "hourly": {
                "time": ["2026-08-02T00:00", ...],
                "temperature_2m": [27.9, ...],
                "relative_humidity_2m": [81, ...],
                ...
            }
        }
    """
    logger.info("Fetching weather for %s …", city_name)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(OPENMETEO_HOURLY_VARS),
        "timezone": "auto",
        "past_days": past_days,
        "forecast_days": forecast_days,
    }

    data = await _get_with_retry(client, params)

    # Enrich with our city label and fetch timestamp
    data["city"] = city_name
    data["fetched_at"] = datetime.now(timezone.utc).isoformat()

    n_hours = len(data.get("hourly", {}).get("time", []))
    logger.info("%s: received %d hourly data points", city_name, n_hours)

    return data


def split_by_hour(weather_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Split a full Open-Meteo response into per-hour JSON documents.

    Returns a dict keyed by ISO timestamp (YYYY-MM-DDTHH:00):
        {
            "2026-08-03T14:00": {
                "city": "mumbai",
                "time": "2026-08-03T14:00",
                "temperature_2m": 29.2,
                "relative_humidity_2m": 74,
                ...
            },
            ...
        }
    """
    hourly = weather_data.get("hourly", {})
    times = hourly.get("time", [])
    city = weather_data.get("city", "unknown")
    fetched_at = weather_data.get("fetched_at", "")

    result: dict[str, dict[str, Any]] = {}

    for i, t in enumerate(times):
        hour_doc: dict[str, Any] = {
            "city": city,
            "time": t,
            "fetched_at": fetched_at,
        }
        # Copy each weather variable for this hour
        for var in OPENMETEO_HOURLY_VARS:
            values = hourly.get(var, [])
            if i < len(values):
                hour_doc[var] = values[i]

        # Include units for self-describing data
        units = weather_data.get("hourly_units", {})
        hour_doc["units"] = {
            var: units.get(var, "") for var in OPENMETEO_HOURLY_VARS
        }

        result[t] = hour_doc

    return result
