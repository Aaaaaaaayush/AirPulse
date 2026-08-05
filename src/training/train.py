"""
AirPulse — Main Training & MLflow Tracking Pipeline.

Orchestrates data loading, feature engineering, LightGBM model training,
metric evaluation, artifact logging, and MLflow model registration.

Usage:
    python -m src.training.train [--horizon 24] [--trees 200] [--lr 0.05]
"""

import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import pandas as pd

from src.config import MLFLOW_TRACKING_URI
from src.features.dataset_loader import load_dataset
from src.features.feature_builder import build_features, split_chronological
from src.features.generate_historical_data import generate_and_save_historical_data
from src.training.baseline_model import AirPulseBaselineModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("airpulse.training")


def run_training_pipeline(
    forecast_horizon: int = 24,
    n_estimators: int = 250,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = 6,
    experiment_name: str = "AirPulse-AQI-Forecast",
    model_name: str = "airpulse-forecaster",
) -> dict:
    """Runs the full Phase 2 baseline model training and MLflow tracking pipeline."""
    logger.info("Initializing MLflow tracking URI: %s", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    # 1. Dataset Loading
    try:
        df_raw = load_dataset()
    except Exception as exc:
        logger.info("Dataset not found (%s). Generating 90-day historical seed data...", exc)
        generate_and_save_historical_data(days=90)
        df_raw = load_dataset()

    logger.info("Loaded raw dataset: %d records across %d cities", len(df_raw), df_raw["city"].nunique())

    # 2. Feature Building
    logger.info("Building features (horizon=%dh)...", forecast_horizon)
    target_col = f"target_aqi_lead{forecast_horizon}h"
    df_features = build_features(df_raw, forecast_horizon=forecast_horizon)
    logger.info("Feature engineering complete: %d rows, %d columns", *df_features.shape)

    # 3. Chronological Train/Val/Test Split
    X_train, y_train, X_val, y_val, X_test, y_test = split_chronological(df_features, target_col)
    logger.info("Split sizes — Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))

    # 4. Start MLflow Run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("Started MLflow run ID: %s", run_id)

        # Log Hyperparameters
        params = {
            "forecast_horizon": forecast_horizon,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "n_features": X_train.shape[1],
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
        }
        mlflow.log_params(params)

        # 5. Train LightGBM Model
        logger.info("Training LightGBM model...")
        model_wrapper = AirPulseBaselineModel(params={
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
        })
        model_wrapper.fit(X_train, y_train, X_val, y_val)

        # 6. Evaluation
        val_metrics = model_wrapper.evaluate(X_val, y_val)
        test_metrics = model_wrapper.evaluate(X_test, y_test)

        logger.info("Validation Metrics — MAE: %.2f, RMSE: %.2f, R2: %.4f", val_metrics["mae"], val_metrics["rmse"], val_metrics["r2"])
        logger.info("Test Metrics       — MAE: %.2f, RMSE: %.2f, R2: %.4f", test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"])

        # Log Metrics
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # 7. Feature Importance & Artifacts
        df_imp = model_wrapper.get_feature_importances()
        logger.info("Top 5 Features:\n%s", df_imp.head(5).to_string(index=False))

        # Save feature importance CSV & plot
        artifact_dir = Path("data/artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        csv_path = artifact_dir / "feature_importance.csv"
        df_imp.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))

        # Feature Importance Plot
        plt.figure(figsize=(10, 6))
        plt.barh(df_imp["feature"].head(12)[::-1], df_imp["importance"].head(12)[::-1], color="#3B82F6")
        plt.title(f"AirPulse Baseline Feature Importance ({forecast_horizon}h AQI Forecast)")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plot_path = artifact_dir / "feature_importance.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(str(plot_path))

        # 8. Log and Register Model
        logger.info("Logging model artifact to MLflow registry '%s'...", model_name)
        mlflow.lightgbm.log_model(
            lgb_model=model_wrapper.model,
            artifact_path="model",
            registered_model_name=model_name,
        )

        logger.info("Pipeline complete! Run ID: %s", run_id)
        return {
            "run_id": run_id,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="AirPulse Phase 2 Baseline Training Pipeline")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon in hours")
    parser.add_argument("--trees", type=int, default=250, help="Number of LightGBM trees")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training_pipeline(
        forecast_horizon=args.horizon,
        n_estimators=args.trees,
        learning_rate=args.lr,
    )
