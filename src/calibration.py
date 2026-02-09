from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split


class SklearnCompatibleCatBoost(BaseEstimator, ClassifierMixin):
    """
    Early Stopping desteği olan ve Scikit-learn CalibratedClassifierCV
    ile uyumlu CatBoost sarmalayıcısı.
    """

    def __init__(self, model: CatBoostClassifier) -> None:
        self.model = model
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> "SklearnCompatibleCatBoost":
        # CalibratedClassifierCV'den gelen veriyi kendi içinde split ediyoruz
        # Böylece modelin overfit olmasını engelleyecek bir eval_set oluşturuyoruz
        x_train_intern, x_val_intern, y_train_intern, y_val_intern = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model.fit(
            x_train_intern,
            y_train_intern,
            eval_set=(x_val_intern, y_val_intern),
            early_stopping_rounds=10,
            verbose=False,
        )

        self.classes_ = self.model.classes_
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.model.predict_proba(X)

    def __sklearn_tags__(self) -> Any:
        from sklearn.utils._tags import ClassifierTags

        return ClassifierTags()


def isotonic_calibrator_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: Dict[str, Any],
    method: str = "isotonic",
    n_splits: int = 3,
) -> CalibratedClassifierCV:
    """
    CatBoost modelini early stopping kullanarak eğitir ve kalibre eder.
    """
    params = best_params.copy()
    if "iterations" not in params and "n_estimators" not in params:
        params["iterations"] = 2000

    base_cb = CatBoostClassifier(**params, verbose=0, allow_writing_files=False)

    wrapped_model = SklearnCompatibleCatBoost(base_cb)

    calibrated_model = CalibratedClassifierCV(
        estimator=wrapped_model, method=method, cv=n_splits
    )

    calibrated_model.fit(X, y)
    return calibrated_model
