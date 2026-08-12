"""
Unit tests for Phase 2 Feature Builder & Chronological Split.
"""

import pandas as pd
import pytest
from src.features.feature_builder import build_features, get_feature_columns, split_chronological
from src.features.generate_historical_data import _generate_city_series_fallback


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    records = []
    records.extend(_generate_city_series_fallback("mumbai", n_hours=120, seed=1))
    records.extend(_generate_city_series_fallback("delhi", n_hours=120, seed=2))
    df = pd.DataFrame(records)
    return df


def test_build_features_shape_and_columns(sample_dataset):
    df_feat = build_features(sample_dataset, forecast_horizon=24, lags=[1, 2, 3, 6, 12, 24])

    assert not df_feat.empty
    assert "target_aqi_lead24h" in df_feat.columns
    assert "aqi_lag_1h" in df_feat.columns
    assert "aqi_roll_mean_6h" in df_feat.columns
    assert "hour_sin" in df_feat.columns
    assert "hour_cos" in df_feat.columns
    assert df_feat["city"].dtype.name == "category"


def test_split_chronological_no_overlap(sample_dataset):
    df_feat = build_features(sample_dataset, forecast_horizon=24)
    target_col = "target_aqi_lead24h"

    X_tr, y_tr, X_val, y_val, X_test, y_test = split_chronological(
        df_feat, target_col, train_ratio=0.8, val_ratio=0.1
    )

    assert len(X_tr) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0
    assert len(X_tr) + len(X_val) + len(X_test) == len(df_feat)

    feature_cols = get_feature_columns(df_feat, target_col)
    assert list(X_tr.columns) == feature_cols
