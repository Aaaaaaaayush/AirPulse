"""
Unit tests for Baseline LightGBM Model & Evaluation Metrics.
"""

import numpy as np
import pandas as pd
from src.features.feature_builder import build_features, split_chronological
from src.features.generate_historical_data import _generate_city_series_fallback
from src.training.baseline_model import AirPulseBaselineModel, calculate_metrics


def test_calculate_metrics():
    y_true = np.array([100.0, 150.0, 200.0])
    y_pred = np.array([110.0, 140.0, 190.0])

    metrics = calculate_metrics(y_true, y_pred)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "mape" in metrics
    assert metrics["mae"] == 10.0
    assert metrics["r2"] > 0.9


def test_baseline_model_fit_predict():
    records = []
    records.extend(_generate_city_series_fallback("mumbai", n_hours=150, seed=1))
    records.extend(_generate_city_series_fallback("delhi", n_hours=150, seed=2))
    df_raw = pd.DataFrame(records)

    df_feat = build_features(df_raw, forecast_horizon=24)
    target_col = "target_aqi_lead24h"

    X_train, y_train, X_val, y_val, X_test, y_test = split_chronological(df_feat, target_col)

    model = AirPulseBaselineModel(params={"n_estimators": 50, "random_state": 42})
    model.fit(X_train, y_train, X_val, y_val)

    preds = model.predict(X_test)
    assert len(preds) == len(X_test)

    eval_res = model.evaluate(X_test, y_test)
    assert eval_res["mae"] > 0
    assert eval_res["r2"] <= 1.0

    df_imp = model.get_feature_importances()
    assert not df_imp.empty
    assert "feature" in df_imp.columns
    assert "importance" in df_imp.columns
