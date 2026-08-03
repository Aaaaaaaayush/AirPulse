"""
AirPulse — Central configuration.

Single source of truth for city coordinates, API endpoints, S3 paths,
and environment-loaded secrets.  Every other module imports from here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root (two levels up from src/config.py)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Target cities — coordinates are city-centre approximations.
# OpenAQ queries use a radius search around these points.
# ---------------------------------------------------------------------------
CITIES: dict[str, dict] = {
    "mumbai": {
        "lat": 19.0760,
        "lon": 72.8777,
        "timezone": "Asia/Kolkata",
    },
    "delhi": {
        "lat": 28.6139,
        "lon": 77.2090,
        "timezone": "Asia/Kolkata",
    },
    "bangalore": {
        "lat": 12.9716,
        "lon": 77.5946,
        "timezone": "Asia/Kolkata",
    },
    "chennai": {
        "lat": 13.0827,
        "lon": 80.2707,
        "timezone": "Asia/Kolkata",
    },
    "kolkata": {
        "lat": 22.5726,
        "lon": 88.3639,
        "timezone": "Asia/Kolkata",
    },
}

# ---------------------------------------------------------------------------
# OpenAQ v3 settings
# ---------------------------------------------------------------------------
OPENAQ_BASE_URL = "https://api.openaq.org/v3"
OPENAQ_API_KEY: str = os.getenv("OPENAQ_API_KEY", "")
OPENAQ_RADIUS_M: int = 25_000          # 25 km radius around city centre
OPENAQ_PAGE_LIMIT: int = 100           # max results per page
OPENAQ_RATE_LIMIT_RPM: int = 60        # free-tier rate limit (requests/min)

# ---------------------------------------------------------------------------
# Open-Meteo settings
# ---------------------------------------------------------------------------
OPENMETEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]
OPENMETEO_PAST_DAYS: int = 1           # include yesterday's actuals
OPENMETEO_FORECAST_DAYS: int = 3       # 3-day forecast (covers 48 h window)

# ---------------------------------------------------------------------------
# AWS / S3 settings
# ---------------------------------------------------------------------------
AWS_REGION: str = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
S3_BUCKET_RAW: str = os.getenv("S3_BUCKET_RAW", "airpulse-raw")
S3_PREFIX_OPENAQ = "openaq"
S3_PREFIX_OPENMETEO = "openmeteo"

# ---------------------------------------------------------------------------
# Local data cache (for --dry-run or offline testing)
# ---------------------------------------------------------------------------
LOCAL_DATA_DIR = _PROJECT_ROOT / "data"
