"""
AirPulse — OpenAQ v3 API client.

Fetches air-quality measurements for configured cities.

Data flow:
    1. GET /v3/locations  (radius query around city centre)
       → returns monitoring stations and their sensors
    2. GET /v3/sensors/{id}/measurements/hourly
       → returns hourly-aggregated pollutant readings per sensor

All responses are returned as raw dicts — no transformation here.
Transformation is Phase 2's job (ELT pattern).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from src.config import (
    OPENAQ_API_KEY,
    OPENAQ_BASE_URL,
    OPENAQ_PAGE_LIMIT,
    OPENAQ_RADIUS_M,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry / rate-limit helpers
# ---------------------------------------------------------------------------
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0        # exponential backoff: 2s, 4s, 8s
_REQUEST_TIMEOUT = 30.0    # seconds


class OpenAQError(Exception):
    """Raised when the OpenAQ API returns a non-recoverable error."""


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------

def _headers() -> dict[str, str]:
    """Build request headers including the API key."""
    h = {"Accept": "application/json"}
    if OPENAQ_API_KEY:
        h["X-API-Key"] = OPENAQ_API_KEY
    return h


async def _get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    params: dict | None = None,
) -> dict[str, Any]:
    """
    GET with exponential backoff on 429 (rate-limited) and 5xx errors.
    Returns the parsed JSON response dict.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = await client.get(
                url, params=params, headers=_headers(), timeout=_REQUEST_TIMEOUT
            )

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429 or resp.status_code >= 500:
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    "OpenAQ %s (attempt %d/%d), retrying in %.1fs …",
                    resp.status_code, attempt, _MAX_RETRIES, wait,
                )
                await asyncio.sleep(wait)
                continue

            # Non-retryable error (401, 403, 404, 422, etc.)
            raise OpenAQError(
                f"OpenAQ API error {resp.status_code}: {resp.text[:300]}"
            )

        except httpx.RequestError as exc:
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                "OpenAQ request error (attempt %d/%d): %s, retrying in %.1fs …",
                attempt, _MAX_RETRIES, exc, wait,
            )
            await asyncio.sleep(wait)

    raise OpenAQError(f"OpenAQ request failed after {_MAX_RETRIES} retries: {url}")


# ---------------------------------------------------------------------------
# Paginated fetch helper
# ---------------------------------------------------------------------------

async def _get_all_pages(
    client: httpx.AsyncClient,
    url: str,
    params: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch all pages from a paginated OpenAQ endpoint.
    Returns a flat list of result dicts.
    """
    params = dict(params or {})
    params.setdefault("limit", OPENAQ_PAGE_LIMIT)
    params.setdefault("page", 1)

    all_results: list[dict[str, Any]] = []

    while True:
        data = await _get_with_retry(client, url, params)
        results = data.get("results", [])
        all_results.extend(results)

        # Check if there are more pages
        meta = data.get("meta", {})
        found = meta.get("found", len(results))
        if len(all_results) >= found or len(results) < params["limit"]:
            break

        params["page"] += 1

    return all_results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def fetch_locations(
    client: httpx.AsyncClient,
    lat: float,
    lon: float,
    radius_m: int = OPENAQ_RADIUS_M,
) -> list[dict[str, Any]]:
    """
    Find air-quality monitoring stations near (lat, lon).

    Returns a list of location dicts, each containing sensor info.
    """
    url = f"{OPENAQ_BASE_URL}/locations"
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": radius_m,
        "limit": OPENAQ_PAGE_LIMIT,
    }
    locations = await _get_all_pages(client, url, params)
    logger.info(
        "Found %d locations within %d m of (%.4f, %.4f)",
        len(locations), radius_m, lat, lon,
    )
    return locations


async def fetch_sensor_measurements(
    client: httpx.AsyncClient,
    sensor_id: int,
    datetime_from: datetime | None = None,
    datetime_to: datetime | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch hourly-aggregated measurements for a single sensor.

    If no time range is given, defaults to the last 24 hours.
    """
    if datetime_from is None:
        datetime_to = datetime.now(timezone.utc)
        datetime_from = datetime_to - timedelta(hours=24)

    url = f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/measurements/hourly"
    params = {
        "datetime_from": datetime_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "datetime_to": datetime_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": OPENAQ_PAGE_LIMIT,
    }
    measurements = await _get_all_pages(client, url, params)
    logger.debug(
        "Sensor %d: fetched %d hourly measurements", sensor_id, len(measurements)
    )
    return measurements


async def fetch_city_air_quality(
    client: httpx.AsyncClient,
    city_name: str,
    lat: float,
    lon: float,
    datetime_from: datetime | None = None,
    datetime_to: datetime | None = None,
) -> dict[str, Any]:
    """
    High-level: fetch all air-quality data for a city.

    Returns a dict with:
        {
            "city": "mumbai",
            "fetched_at": "2026-08-03T14:00:00Z",
            "locations": [ ... raw location dicts ... ],
            "measurements": {
                <sensor_id>: [ ... hourly measurement dicts ... ]
            }
        }
    """
    logger.info("Fetching air quality for %s …", city_name)

    # Step 1: find stations near the city
    locations = await fetch_locations(client, lat, lon)

    if not locations:
        logger.warning("No OpenAQ locations found for %s", city_name)
        return {
            "city": city_name,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "locations": [],
            "measurements": {},
        }

    # Step 2: collect sensor IDs from all locations
    sensor_ids: list[int] = []
    for loc in locations:
        for sensor in loc.get("sensors", []):
            sid = sensor.get("id")
            if sid:
                sensor_ids.append(sid)

    logger.info(
        "%s: found %d sensors across %d locations",
        city_name, len(sensor_ids), len(locations),
    )

    # Step 3: fetch measurements for each sensor (with concurrency limit)
    semaphore = asyncio.Semaphore(5)  # max 5 concurrent sensor fetches

    async def _fetch_one(sid: int) -> tuple[int, list[dict]]:
        async with semaphore:
            measurements = await fetch_sensor_measurements(
                client, sid, datetime_from, datetime_to
            )
            return sid, measurements

    tasks = [_fetch_one(sid) for sid in sensor_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    measurements_by_sensor: dict[int, list[dict]] = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error("Sensor fetch failed: %s", result)
            continue
        sid, meas = result
        if meas:
            measurements_by_sensor[sid] = meas

    return {
        "city": city_name,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "locations": locations,
        "measurements": {
            str(k): v for k, v in measurements_by_sensor.items()
        },
    }
