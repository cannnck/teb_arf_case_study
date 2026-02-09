from typing import Any, Dict, Union

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from optuna.trial import Trial
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold


def objective(
    trial: Trial, X: pd.DataFrame, y: pd.Series, n_splits: int
) -> Union[float, int]:
    param_grid: Dict[str, Any] = {
        "iterations": trial.suggest_int("iterations", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 2, 4),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2, 15.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-2, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", 1.0, (len(y) - sum(y)) / sum(y)
        ),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
        ),
        "eval_metric": "PRAUC",
        "random_state": 42,
        "verbose": 0,
        "allow_writing_files": False,
    }

    # Bernoulli veya MVS seçilirse subsample parametresi gerekir
    if param_grid["bootstrap_type"] in ["Bernoulli", "MVS"]:
        param_grid["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_scores_val: list[float] = []
    cv_scores_train: list[float] = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**param_grid)

        # Early stopping kullanarak optimizasyonu hızlandırıyoruz
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=10,
            verbose=False,
        )

        val_preds: np.ndarray = model.predict_proba(X_val)[:, 1]
        train_preds: np.ndarray = model.predict_proba(X_train)[:, 1]

        cv_scores_val.append(average_precision_score(y_val, val_preds))
        cv_scores_train.append(average_precision_score(y_train, train_preds))

    val_mean: float = float(np.mean(cv_scores_val))
    train_mean: float = float(np.mean(cv_scores_train))

    gap: float = train_mean - val_mean
    if gap > 0.03:
        return 0  # Kesinlikle overfit istemiyoruz!

    return val_mean


def run_optuna_study(
    X: pd.DataFrame, y: pd.Series, n_trials: int = 50, n_splits: int = 3
) -> Dict[str, Any]:
    study = optuna.create_study(
        direction="maximize",
        study_name="CatBoost_Optimization",
        sampler=optuna.samplers.TPESampler(
            seed=42, n_startup_trials=10, multivariate=True, group=True
        ),
    )

    def func(trial: Trial) -> Union[float, int]:
        return objective(trial, X, y, n_splits)

    print(f"Optuna {n_trials} deneme için CatBoost üzerinde başlatılıyor...")
    study.optimize(func, n_trials=n_trials)

    print("\n--- OPTIMIZASYON TAMAMLANDI ---")
    print(f"En İyi PR-AUC: {study.best_value:.4f}")

    return study.best_params
