"""
AirPulse — Feature Builder.

Transforms raw AQI & weather time series into ML-ready features.
Enforces strict chronological train/validation/test splits to prevent data leakage.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def build_features(
    df: pd.DataFrame,
    forecast_horizon: int = 24,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Builds lag features, rolling statistics, cyclical time encodings, and
    target variable for multi-city AQI forecasting.

    Args:
        df: Input DataFrame with columns: ['timestamp', 'city', 'aqi', 'temperature_2m', ...]
        forecast_horizon: Hours ahead to forecast (default: 24)
        lags: List of lag hours to compute (default: [1, 2, 3, 6, 12, 24, 48])

    Returns:
        DataFrame with feature columns and target_lead_{horizon} column.
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48]

    # Ensure sorted by city and timestamp
    data = df.sort_values(by=["city", "timestamp"]).copy()

    # 1. Cyclical Time Encodings
    dt = pd.to_datetime(data["timestamp"])
    data["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24.0)
    data["day_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7.0)
    data["day_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7.0)

    # 2. Per-City Grouped Lag & Rolling Features
    feature_dfs = []
    for city_name, group in data.groupby("city", sort=False):
        g = group.copy()

        # Lag features
        for lag in lags:
            g[f"aqi_lag_{lag}h"] = g["aqi"].shift(lag)
            g[f"temp_lag_{lag}h"] = g["temperature_2m"].shift(lag)

        # Rolling window features
        for w in [6, 12, 24]:
            g[f"aqi_roll_mean_{w}h"] = g["aqi"].shift(1).rolling(window=w).mean()
            g[f"aqi_roll_std_{w}h"] = g["aqi"].shift(1).rolling(window=w).std()
            g[f"aqi_roll_max_{w}h"] = g["aqi"].shift(1).rolling(window=w).max()
            g[f"aqi_roll_min_{w}h"] = g["aqi"].shift(1).rolling(window=w).min()

        # Lead Target Variable (AQI 24h in the future)
        g[f"target_aqi_lead{forecast_horizon}h"] = g["aqi"].shift(-forecast_horizon)

        feature_dfs.append(g)

    result_df = pd.concat(feature_dfs, axis=0).reset_index(drop=True)

    # Drop NaNs created by lagging & leading
    result_df = result_df.dropna().reset_index(drop=True)

    # Ensure city is a categorical feature for LightGBM
    result_df["city"] = result_df["city"].astype("category")

    return result_df


def get_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """Returns list of feature column names excluding timestamps and target."""
    exclude = {"timestamp", target_col, "pm25", "pm10"}
    return [col for col in df.columns if col not in exclude]


def split_chronological(
    df: pd.DataFrame,
    target_col: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Chronologically splits dataset per city to ensure zero data leakage.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    feature_cols = get_feature_columns(df, target_col)

    train_parts, val_parts, test_parts = [], [], []

    for _, group in df.groupby("city", observed=False, sort=False):
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train : n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val :])

    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
    val_df = pd.concat(val_parts, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0).reset_index(drop=True)

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    return X_train, y_train, X_val, y_val, X_test, y_test
