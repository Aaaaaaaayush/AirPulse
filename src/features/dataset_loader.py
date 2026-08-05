"""
AirPulse — Dataset Loader.

Loads ingested JSON data (or preprocessed historical files) from local disk
or AWS S3, standardising them into a single clean Pandas DataFrame for feature engineering.
"""

import json
from pathlib import Path
from typing import Any
import pandas as pd
import boto3

from src.config import AWS_REGION, CITIES, LOCAL_DATA_DIR, S3_BUCKET_RAW


def load_from_local_processed(data_dir: Path | None = None) -> pd.DataFrame:
    """Loads preprocessed/historical JSON datasets from local directory."""
    target_dir = data_dir or (LOCAL_DATA_DIR / "processed")
    if not target_dir.exists():
        raise FileNotFoundError(f"Directory {target_dir} does not exist. Run generate_historical_data first.")

    json_files = list(target_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {target_dir}")

    all_records = []
    for f in json_files:
        records = json.loads(f.read_text(encoding="utf-8"))
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["city", "timestamp"]).reset_index(drop=True)
    return df


def load_raw_ingested_local(data_dir: Path | None = None) -> pd.DataFrame:
    """Loads raw Open-Meteo & OpenAQ partition JSON files from data/ directory."""
    target_dir = data_dir or LOCAL_DATA_DIR
    weather_files = list((target_dir / "openmeteo").rglob("*.json"))

    if not weather_files:
        return pd.DataFrame()

    weather_records = []
    for f in weather_files:
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            weather_records.append(doc)
        except Exception:
            continue

    df = pd.DataFrame(weather_records)
    if df.empty:
        return df

    # Standardise column names
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.sort_values(by=["city", "timestamp"]).reset_index(drop=True)
    return df


def load_dataset(use_s3: bool = False, bucket: str = S3_BUCKET_RAW) -> pd.DataFrame:
    """High-level function: loads dataset from local processed storage or live S3."""
    processed_dir = LOCAL_DATA_DIR / "processed"
    if processed_dir.exists() and list(processed_dir.glob("*.json")):
        df = load_from_local_processed(processed_dir)
        if "aqi" in df.columns:
            return df

    raw_df = load_raw_ingested_local()
    if not raw_df.empty and "aqi" in raw_df.columns:
        return raw_df

    # If local raw data has no AQI (e.g. weather-only), generate historical seed dataset
    from src.features.generate_historical_data import generate_and_save_historical_data
    generate_and_save_historical_data(days=90)
    return load_from_local_processed(processed_dir)
