"""
AirPulse — Ingestion orchestrator.

Main entry point that ties together the API clients and S3 uploader.
Fetches air-quality + weather data for all configured cities, partitions
by hour, and uploads to S3.

Usage:
    # Dry-run (local save, no S3/API keys needed for weather)
    python -m src.ingestion.run_ingestion --dry-run

    # Specific cities only
    python -m src.ingestion.run_ingestion --cities mumbai,delhi

    # Full run (requires .env with API keys + AWS credentials)
    python -m src.ingestion.run_ingestion
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone

import httpx

from src.config import CITIES, OPENAQ_API_KEY
from src.ingestion.openaq_client import fetch_city_air_quality
from src.ingestion.openmeteo_client import fetch_city_weather, split_by_hour
from src.ingestion.s3_uploader import S3Uploader

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("airpulse.ingestion")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

async def ingest_city(
    client: httpx.AsyncClient,
    city_name: str,
    city_cfg: dict,
    uploader: S3Uploader,
    skip_openaq: bool = False,
) -> dict:
    """
    Ingest all data for a single city.

    Returns a summary dict with counts.
    """
    lat, lon = city_cfg["lat"], city_cfg["lon"]
    summary = {"city": city_name, "openaq_objects": 0, "openmeteo_objects": 0, "errors": []}

    # --- Open-Meteo (weather) ---
    try:
        weather = await fetch_city_weather(client, city_name, lat, lon)
        hourly_docs = split_by_hour(weather)

        for timestamp, doc in hourly_docs.items():
            # Parse "2026-08-03T14:00" → date="2026-08-03", hour="14"
            dt = datetime.fromisoformat(timestamp)
            date_str = dt.strftime("%Y-%m-%d")
            hour_str = dt.strftime("%H")
            uploader.upload_json("openmeteo", city_name, date_str, hour_str, doc)
            summary["openmeteo_objects"] += 1

    except Exception as exc:
        logger.error("%s: Open-Meteo failed: %s", city_name, exc)
        summary["errors"].append(f"openmeteo: {exc}")

    # --- OpenAQ (air quality) ---
    if skip_openaq:
        logger.info("%s: skipping OpenAQ (no API key)", city_name)
    else:
        try:
            aq_data = await fetch_city_air_quality(client, city_name, lat, lon)

            # Upload the full city response as a single document per fetch hour
            now = datetime.now(timezone.utc)
            date_str = now.strftime("%Y-%m-%d")
            hour_str = now.strftime("%H")
            uploader.upload_json("openaq", city_name, date_str, hour_str, aq_data)
            summary["openaq_objects"] += 1

        except Exception as exc:
            logger.error("%s: OpenAQ failed: %s", city_name, exc)
            summary["errors"].append(f"openaq: {exc}")

    return summary


async def run_pipeline(
    cities: dict[str, dict],
    dry_run: bool = False,
) -> list[dict]:
    """
    Run the full ingestion pipeline for all specified cities.
    """
    uploader = S3Uploader(dry_run=dry_run)
    skip_openaq = not OPENAQ_API_KEY

    if skip_openaq:
        logger.warning(
            "OPENAQ_API_KEY not set — OpenAQ ingestion will be skipped. "
            "Set it in .env to enable air-quality data."
        )

    async with httpx.AsyncClient() as client:
        tasks = [
            ingest_city(client, name, cfg, uploader, skip_openaq)
            for name, cfg in cities.items()
        ]
        summaries = await asyncio.gather(*tasks)

    # Print summary
    logger.info("=" * 60)
    logger.info("Ingestion complete — %s", uploader)
    logger.info("=" * 60)
    for s in summaries:
        errors_str = f"  ERRORS: {s['errors']}" if s["errors"] else ""
        logger.info(
            "  %-12s  openmeteo=%d  openaq=%d%s",
            s["city"], s["openmeteo_objects"], s["openaq_objects"], errors_str,
        )

    return summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AirPulse — Ingest air-quality and weather data",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Save to local data/ directory instead of S3",
    )
    parser.add_argument(
        "--cities",
        type=str,
        default=None,
        help="Comma-separated list of cities (default: all)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Filter cities if specified
    if args.cities:
        selected = [c.strip().lower() for c in args.cities.split(",")]
        cities = {k: v for k, v in CITIES.items() if k in selected}
        unknown = set(selected) - set(cities.keys())
        if unknown:
            logger.error("Unknown cities: %s. Available: %s", unknown, list(CITIES.keys()))
            sys.exit(1)
    else:
        cities = CITIES

    logger.info(
        "Starting ingestion for %d cities: %s (dry_run=%s)",
        len(cities), list(cities.keys()), args.dry_run,
    )

    asyncio.run(run_pipeline(cities, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
