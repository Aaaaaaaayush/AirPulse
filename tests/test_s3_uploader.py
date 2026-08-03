"""
Tests for S3 uploader and Hive partition key generation.
"""

import json
import boto3
import pytest
from moto import mock_aws
from src.ingestion.s3_uploader import S3Uploader, _build_s3_key


def test_build_s3_key():
    key = _build_s3_key("openaq", "mumbai", "2026-08-03", "14")
    assert key == "openaq/city=mumbai/date=2026-08-03/hour=14.json"


def test_s3_uploader_dry_run(tmp_path, monkeypatch):
    # Redirect LOCAL_DATA_DIR to tmp_path
    monkeypatch.setattr("src.ingestion.s3_uploader.LOCAL_DATA_DIR", tmp_path)

    uploader = S3Uploader(dry_run=True)
    data = {"temp": 28.5, "city": "mumbai"}

    path_str = uploader.upload_json("openmeteo", "mumbai", "2026-08-03", "14", data)
    assert "city=mumbai" in path_str

    # Verify file content
    saved_file = tmp_path / "openmeteo" / "city=mumbai" / "date=2026-08-03" / "hour=14.json"
    assert saved_file.exists()

    content = json.loads(saved_file.read_text(encoding="utf-8"))
    assert content["temp"] == 28.5
    assert uploader.stats["uploaded"] == 1


@mock_aws
def test_s3_uploader_real_s3():
    # Setup mock S3 bucket
    s3 = boto3.client("s3", region_name="ap-south-1")
    s3.create_bucket(
        Bucket="airpulse-raw",
        CreateBucketConfiguration={"LocationConstraint": "ap-south-1"},
    )

    uploader = S3Uploader(bucket="airpulse-raw", region="ap-south-1", dry_run=False)
    data = {"aqi": 120, "city": "delhi"}

    uri = uploader.upload_json("openaq", "delhi", "2026-08-03", "15", data)
    assert uri == "s3://airpulse-raw/openaq/city=delhi/date=2026-08-03/hour=15.json"

    # Read back from S3
    obj = s3.get_object(Bucket="airpulse-raw", Key="openaq/city=delhi/date=2026-08-03/hour=15.json")
    body = json.loads(obj["Body"].read().decode("utf-8"))

    assert body["aqi"] == 120
    assert uploader.stats["uploaded"] == 1
