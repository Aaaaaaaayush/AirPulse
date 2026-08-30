"""
AirPulse — Evidently AI Data & Prediction Drift Detector.

Monitors statistical feature distribution shifts (Kolmogorov-Smirnov / Wasserstein)
between reference training data and live inference data, generates interactive HTML reports,
and triggers automated retraining pipelines when drift threshold is exceeded.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config import LOCAL_DATA_DIR, MODEL_REGISTRY_NAME, MLFLOW_TRACKING_URI
from src.features.dataset_loader import load_dataset
from src.features.feature_builder import build_features, get_feature_columns

logger = logging.getLogger(__name__)


def get_drift_report_path() -> Path:
    """Returns absolute path to generated drift HTML report."""
    report_dir = LOCAL_DATA_DIR.parent / "data" / "artifacts"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / "drift_report.html"


def evaluate_data_drift(
    reference_df: pd.DataFrame | None = None,
    current_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Computes data drift between reference (baseline) and current dataset using Evidently AI.
    Generates data/artifacts/drift_report.html.
    """
    target_col = "target_aqi_lead24h"

    # Prepare datasets if not provided
    if reference_df is None or current_df is None:
        df_raw = load_dataset()
        df_feat = build_features(df_raw, forecast_horizon=24)
        feature_cols = get_feature_columns(df_feat, target_col)

        # Split 70/30 chronologically as reference vs current for demonstration
        split_idx = int(len(df_feat) * 0.7)
        reference_df = df_feat.iloc[:split_idx][feature_cols].copy()
        current_df = df_feat.iloc[split_idx:][feature_cols].copy()
    else:
        feature_cols = [c for c in reference_df.columns if c not in {"timestamp", target_col, "city"}]
        reference_df = reference_df[feature_cols].copy()
        current_df = current_df[feature_cols].copy()

    # Ensure numeric types
    for col in reference_df.columns:
        if reference_df[col].dtype.name == "category" or reference_df[col].dtype == "object":
            reference_df[col] = reference_df[col].astype("category").cat.codes
            current_df[col] = current_df[col].astype("category").cat.codes

    # Run Evidently AI DataDriftPreset report
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_df, current_data=current_df)

    # Save interactive HTML report
    html_path = get_drift_report_path()
    snapshot.save_html(str(html_path))
    logger.info("Evidently AI drift report saved to %s", html_path)

    # Extract JSON summary dict from snapshot
    snapshot_dict = snapshot.dict()
    drift_metrics = {}

    metrics_list = snapshot_dict.get("metrics", [])
    for metric in metrics_list:
        result = metric.get("result", {})
        if "dataset_drift" in result or "number_of_drifted_columns" in result:
            drift_share = float(result.get("share_of_drifted_columns", 0.0))
            n_drifted = int(result.get("number_of_drifted_columns", 0))
            n_columns = int(result.get("number_of_columns", len(feature_cols)))
            has_drift = bool(result.get("dataset_drift", drift_share > 0.3))

            drift_metrics = {
                "dataset_drift": has_drift,
                "drift_share": round(drift_share, 4),
                "number_of_drifted_features": n_drifted,
                "total_features": n_columns,
                "report_url": "/reports/drift_report.html",
            }
            break

    if not drift_metrics:
        drift_metrics = {
            "dataset_drift": False,
            "drift_share": 0.0,
            "number_of_drifted_features": 0,
            "total_features": len(feature_cols),
            "report_url": "/reports/drift_report.html",
        }

    return drift_metrics


def check_and_trigger_retrain(drift_threshold: float = 0.30) -> dict[str, Any]:
    """
    Evaluates dataset drift against threshold. If drift exceeds threshold,
    triggers automated model retraining pipeline.
    """
    metrics = evaluate_data_drift()
    drift_share = metrics.get("drift_share", 0.0)

    retrain_triggered = False
    retrain_summary = None

    if drift_share >= drift_threshold or metrics.get("dataset_drift", False):
        logger.warning(
            "Drift threshold exceeded (drift_share=%.2f >= %.2f)! Triggering automated retraining...",
            drift_share, drift_threshold
        )
        try:
            from src.training.train import run_training_pipeline
            retrain_summary = run_training_pipeline()
            retrain_triggered = True

            # Promote new version to Production in MLflow
            import mlflow
            from mlflow.tracking import MlflowClient
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
            if versions:
                latest_ver = max(int(v.version) for v in versions)
                client.transition_model_version_stage(
                    name=MODEL_REGISTRY_NAME, version=latest_ver, stage="Production"
                )
                logger.info("Promoted newly retrained model version %d to Production stage!", latest_ver)
        except Exception as exc:
            logger.error("Automated retraining failed: %s", exc)
            retrain_summary = {"error": str(exc)}
    else:
        logger.info("Drift within acceptable bounds (drift_share=%.2f < %.2f). No retraining needed.", drift_share, drift_threshold)

    return {
        "drift_metrics": metrics,
        "retrain_triggered": retrain_triggered,
        "retrain_summary": retrain_summary,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = check_and_trigger_retrain(drift_threshold=0.30)
    print("Drift Check Results:", json.dumps(results, indent=2))
