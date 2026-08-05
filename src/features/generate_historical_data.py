"""
AirPulse — Historical Data Generator.

Generates 90 days of realistic hourly weather and air quality data for our 5 target
Indian cities (Mumbai, Delhi, Bangalore, Chennai, Kolkata).

Saves partitioned JSON files into `data/` matching the exact schema of our live
Open-Meteo & OpenAQ ingestion pipeline so the feature engineering pipeline can
transparently consume both simulated and real live S3 data.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

from src.config import CITIES, LOCAL_DATA_DIR, OPENMETEO_HOURLY_VARS


def _generate_city_series(city: str, n_hours: int = 90 * 24, seed: int = 42) -> list[dict]:
    """Generate n_hours of hourly records for a specific city."""
    rng = np.random.default_rng(seed + hash(city) % 1000)

    # Base characteristics per city
    city_baselines = {
        "delhi": {"aqi_mean": 210, "aqi_std": 55, "temp_mean": 28, "humidity_mean": 60, "wind_mean": 10},
        "mumbai": {"aqi_mean": 130, "aqi_std": 35, "temp_mean": 30, "humidity_mean": 75, "wind_mean": 18},
        "kolkata": {"aqi_mean": 160, "aqi_std": 40, "temp_mean": 29, "humidity_mean": 70, "wind_mean": 12},
        "chennai": {"aqi_mean": 95, "aqi_std": 25, "temp_mean": 31, "humidity_mean": 78, "wind_mean": 16},
        "bangalore": {"aqi_mean": 80, "aqi_std": 20, "temp_mean": 24, "humidity_mean": 65, "wind_mean": 14},
    }
    base = city_baselines.get(city, city_baselines["mumbai"])

    start_time = datetime.now(timezone.utc) - timedelta(hours=n_hours)
    records = []

    for h in range(n_hours):
        dt = start_time + timedelta(hours=h)
        hour = dt.hour
        day_of_week = dt.weekday()

        # Diurnal temperature cycle (peak at 14:00, lowest at 05:00)
        temp_cycle = 4.0 * np.sin((hour - 8) * np.pi / 12)
        temp = max(10.0, base["temp_mean"] + temp_cycle + rng.normal(0, 1.5))

        # Humidity anti-correlated with temperature
        humidity = np.clip(base["humidity_mean"] - 2.5 * temp_cycle + rng.normal(0, 3.0), 20.0, 98.0)

        # Wind speed with random gusts
        wind_speed = max(1.0, base["wind_mean"] + rng.normal(0, 3.5))
        wind_dir = float((220 + rng.normal(0, 30)) % 360)

        # Occasional rain
        precip = float(max(0.0, rng.choice([0.0, 0.0, 0.0, 0.0, 0.0, 1.2, 4.5], p=[0.8, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01])))

        # AQI calculation with traffic peaks (08:00 and 19:00) & weather dispersion effect
        traffic_peak = 25.0 * np.exp(-((hour - 9) ** 2) / 8) + 30.0 * np.exp(-((hour - 19) ** 2) / 8)
        wind_dispersion = -1.2 * (wind_speed - base["wind_mean"])
        rain_washout = -20.0 if precip > 0.5 else 0.0

        aqi_val = max(15.0, base["aqi_mean"] + traffic_peak + wind_dispersion + rain_washout + rng.normal(0, 12.0))
        pm25_val = max(5.0, aqi_val * 0.6 + rng.normal(0, 4.0))
        pm10_val = max(10.0, aqi_val * 1.1 + rng.normal(0, 8.0))

        records.append({
            "timestamp": dt.strftime("%Y-%m-%dT%H:00:00Z"),
            "city": city,
            "temperature_2m": round(float(temp), 2),
            "relative_humidity_2m": round(float(humidity), 2),
            "wind_speed_10m": round(float(wind_speed), 2),
            "wind_direction_10m": round(float(wind_dir), 2),
            "precipitation": round(float(precip), 2),
            "aqi": round(float(aqi_val), 2),
            "pm25": round(float(pm25_val), 2),
            "pm10": round(float(pm10_val), 2),
        })

    return records


def generate_and_save_historical_data(days: int = 90) -> list[Path]:
    """Generates historical records for all cities and saves to data/historical/."""
    out_dir = LOCAL_DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_hours = days * 24
    saved_files = []

    for city in CITIES:
        records = _generate_city_series(city, n_hours=n_hours)
        file_path = out_dir / f"{city}_historical_{days}d.json"
        file_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        saved_files.append(file_path)
        print(f"Generated {len(records)} records for {city} -> {file_path}")

    return saved_files


if __name__ == "__main__":
    generate_and_save_historical_data()
