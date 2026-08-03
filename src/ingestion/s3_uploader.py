"""
AirPulse — S3 uploader with Hive-style partitioning.

Writes raw JSON documents to S3 using partition paths like:
    s3://airpulse-raw/openaq/city=mumbai/date=2026-08-03/hour=14.json
    s3://airpulse-raw/openmeteo/city=mumbai/date=2026-08-03/hour=14.json

Why Hive-style partitioning?
    Tools like Athena, Spark, and even simple Python scripts can filter
    by city/date/hour from the path alone — no need to open every file.
    It's the standard convention in data-lake architectures.

Idempotency:
    Same city + date + hour always produces the same S3 key.
    Re-running overwrites the object — no duplicates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from src.config import AWS_REGION, LOCAL_DATA_DIR, S3_BUCKET_RAW

logger = logging.getLogger(__name__)


def _build_s3_key(
    source: str,
    city: str,
    date_str: str,
    hour_str: str,
) -> str:
    """
    Build a Hive-style S3 key.

    Example:
        _build_s3_key("openaq", "mumbai", "2026-08-03", "14")
        → "openaq/city=mumbai/date=2026-08-03/hour=14.json"
    """
    return f"{source}/city={city}/date={date_str}/hour={hour_str}.json"


class S3Uploader:
    """Upload JSON documents to S3 or local filesystem (dry-run mode)."""

    def __init__(
        self,
        bucket: str = S3_BUCKET_RAW,
        region: str = AWS_REGION,
        dry_run: bool = False,
    ):
        self.bucket = bucket
        self.dry_run = dry_run
        self._client = None if dry_run else boto3.client("s3", region_name=region)
        self._upload_count = 0
        self._error_count = 0

    def upload_json(
        self,
        source: str,
        city: str,
        date_str: str,
        hour_str: str,
        data: dict[str, Any],
    ) -> str:
        """
        Upload a single JSON document to S3 (or save locally in dry-run).

        Args:
            source:   "openaq" or "openmeteo"
            city:     lowercase city name
            date_str: "YYYY-MM-DD"
            hour_str: "HH" (zero-padded, 24h format)
            data:     the raw JSON-serialisable dict

        Returns:
            The S3 key (or local path in dry-run mode).
        """
        key = _build_s3_key(source, city, date_str, hour_str)
        body = json.dumps(data, ensure_ascii=False, indent=2)

        if self.dry_run:
            return self._save_local(key, body)

        return self._upload_to_s3(key, body)

    def _upload_to_s3(self, key: str, body: str) -> str:
        """Put object to S3."""
        try:
            self._client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body.encode("utf-8"),
                ContentType="application/json",
            )
            self._upload_count += 1
            logger.debug("Uploaded s3://%s/%s", self.bucket, key)
            return f"s3://{self.bucket}/{key}"

        except ClientError as exc:
            self._error_count += 1
            logger.error("S3 upload failed for %s: %s", key, exc)
            raise

    def _save_local(self, key: str, body: str) -> str:
        """Save to local data/ directory (dry-run mode)."""
        local_path = LOCAL_DATA_DIR / key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(body, encoding="utf-8")
        self._upload_count += 1
        logger.debug("Saved locally: %s", local_path)
        return str(local_path)

    @property
    def stats(self) -> dict[str, int]:
        """Return upload statistics."""
        return {
            "uploaded": self._upload_count,
            "errors": self._error_count,
        }

    def __repr__(self) -> str:
        mode = "DRY-RUN (local)" if self.dry_run else f"s3://{self.bucket}"
        return f"S3Uploader({mode}, uploaded={self._upload_count})"
