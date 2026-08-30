"""
Unit and Integration Tests for Evidently AI Drift Detector and Monitoring Endpoints.
"""

import pytest
import pandas as pd
import numpy as np
from httpx import AsyncClient, ASGITransport

from src.serving.app import app
from src.monitoring.drift_detector import evaluate_data_drift, get_drift_report_path


def test_get_drift_report_path():
    path = get_drift_report_path()
    assert path.name == "drift_report.html"


def test_evaluate_data_drift_basic():
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "temperature": np.random.normal(25, 2, 100),
        "humidity": np.random.normal(60, 5, 100),
    })
    curr_df = pd.DataFrame({
        "temperature": np.random.normal(25, 2, 100),
        "humidity": np.random.normal(60, 5, 100),
    })

    metrics = evaluate_data_drift(reference_df=ref_df, current_df=curr_df)
    assert "dataset_drift" in metrics
    assert "drift_share" in metrics
    assert "report_url" in metrics
    assert metrics["report_url"] == "/reports/drift_report.html"


@pytest.mark.asyncio
async def test_get_drift_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/drift")
        assert response.status_code == 200
        data = response.json()
        assert "dataset_drift" in data
        assert "drift_share" in data
        assert "report_url" in data
