"""
AirPulse — Baseline Model Wrapper.

Wraps LightGBM Regressor for 24h AQI forecasting with evaluation metrics
(MAE, RMSE, R2, MAPE) and feature importance calculation.
"""

from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> Dict[str, float]:
    """Computes MAE, RMSE, R2 score, and MAPE."""
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))

    # Avoid zero-division in MAPE
    non_zero = y_t != 0
    mape = float(np.mean(np.abs((y_t[non_zero] - y_p[non_zero]) / y_t[non_zero])) * 100.0)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mape": round(mape, 4),
    }


class AirPulseBaselineModel:
    """LightGBM Baseline Regressor for AQI Forecasting."""

    def __init__(self, params: Dict[str, Any] | None = None):
        default_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 250,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "random_state": 42,
            "verbose": -1,
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = lgb.LGBMRegressor(**self.params)
        self.feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "AirPulseBaselineModel":
        """Fits LightGBM regressor with optional early stopping on validation set."""
        self.feature_names = list(X_train.columns)

        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)] if X_val is not None else []

        eval_X = X_val if X_val is not None else None
        eval_y = y_val if y_val is not None else None

        self.model.fit(
            X_train,
            y_train,
            eval_X=eval_X,
            eval_y=eval_y,
            callbacks=callbacks,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates predictions."""
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Generates predictions and evaluates performance metrics."""
        preds = self.predict(X)
        return calculate_metrics(y, preds)

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns DataFrame of feature importances sorted descending."""
        if not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame()

        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        })
        return df_imp.sort_values(by="importance", ascending=False).reset_index(drop=True)
