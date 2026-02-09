from typing import List, Tuple

import catboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def calculate_iv_summary(
    df: pd.DataFrame, target: pd.Series, features: List[str], n_bins: int = 10
) -> pd.DataFrame:
    """
    Değişkenlerin IV değerlerini hesaplar ve bir rapor oluşturur.
    """
    iv_results = []

    for col in features:
        temp_df = df[[col]].copy()
        temp_df["Target"] = target

        if temp_df[col].dtype in [np.float64, np.int64]:
            try:
                temp_df["bin"] = pd.qcut(temp_df[col], q=n_bins, duplicates="drop")
            except ValueError:
                temp_df["bin"] = pd.cut(temp_df[col], bins=n_bins)
        else:
            temp_df["bin"] = temp_df[col].fillna("Missing")

        stats = temp_df.groupby("bin", observed=True)["Target"].agg(["count", "sum"])
        stats.columns = ["Total", "Events"]
        stats["Non-Events"] = stats["Total"] - stats["Events"]

        stats["Events"] = stats["Events"].replace(0, 0.5)
        stats["Non-Events"] = stats["Non-Events"].replace(0, 0.5)

        dist_event = stats["Events"] / stats["Events"].sum()
        dist_non_event = stats["Non-Events"] / stats["Non-Events"].sum()

        woe = np.log(dist_event / dist_non_event)
        iv_bin = (dist_event - dist_non_event) * woe

        total_iv = iv_bin.sum()
        iv_results.append({"Variable": col, "IV": total_iv})

    return pd.DataFrame(iv_results).sort_values(by="IV", ascending=False)


def select_features_by_iv(
    df: pd.DataFrame,
    target: pd.Series,
    features: List[str],
    threshold: float = 0.02,
    max_threshold: float = 0.5,
) -> Tuple[List[str], pd.DataFrame]:
    """
    IV değerine göre değişken seçimi yapar ve Predictive Power etiketlerini atar.
    """
    iv_df = calculate_iv_summary(df, target, features)

    # Koşulları tanımlayalım
    conditions = [
        (iv_df["IV"] < 0.02),
        (iv_df["IV"] >= 0.02) & (iv_df["IV"] < 0.1),
        (iv_df["IV"] >= 0.1) & (iv_df["IV"] < 0.3),
        (iv_df["IV"] >= 0.3) & (iv_df["IV"] < 0.5),
        (iv_df["IV"] >= 0.5),
    ]
    choices = ["Useless", "Weak", "Medium", "Strong", "Suspicious (Leakage?)"]

    iv_df["Predictive_Power"] = np.select(conditions, choices, default="Unknown")

    selected_features = iv_df[
        (iv_df["IV"] >= threshold) & (iv_df["IV"] < max_threshold)
    ]["Variable"].tolist()

    print("--- IV Eleme Özeti ---")
    print(f"Toplam Değişken: {len(features)}")
    print(f"Seçilen Değişken: {len(selected_features)}")
    print(f"Elenen Değişken: {len(features) - len(selected_features)}")

    return selected_features, iv_df


def perform_nan_safe_boruta(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    model_name: str,
    n_iterations: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    NaN içeren verilerde XGBoost kullanarak Boruta mantığıyla
    özellik seçimi yapar. Imputation gerektirmez.
    """
    np.random.seed(random_state)
    x = df[features].copy()
    y = df[target_col]

    feature_hits = np.zeros(len(features))

    counts = y.value_counts()
    scale_pos_weight = counts[0] / counts[1]

    if model_name == "XGBoost":
        model = xgb.XGBClassifier(
            n_jobs=-1,
            random_state=random_state,
            tree_method="hist",  # Hız için
            enable_categorical=True,
            max_depth=3,
            importance_type="gain",
            scale_pos_weight=scale_pos_weight,
        )

    elif model_name == "CatBoost":
        model = catboost.CatBoostClassifier(
            verbose=0,
            scale_pos_weight=scale_pos_weight,
            allow_writing_files=False,
            random_state=42,
            max_depth=3,
        )
    else:
        print("XGBoost ya da CatBoost seçiniz!")
        raise

    print(f"İşlem başlatılıyor: {n_iterations} iterasyon...")

    for i in range(n_iterations):
        # Shadow Features Oluşturma (Her sütunu kendi içinde karıştır)
        x_shadow = x.apply(lambda col: col.sample(frac=1).values)
        x_shadow.columns = [f"shadow_{c}" for c in x.columns]

        # Gerçek ve Shadow veriyi birleştir
        x_combined = pd.concat([x, x_shadow], axis=1)

        model.fit(x_combined, y)

        importances = model.feature_importances_
        real_importances = importances[: len(features)]
        shadow_importances = importances[len(features) :]

        shadow_threshold = shadow_importances.mean()

        feature_hits += real_importances > shadow_threshold

        if (i + 1) % 10 == 0:
            print(f"İterasyon {i + 1} tamamlandı.")

    report = pd.DataFrame(
        {
            "Feature": features,
            "Hits": feature_hits,
            "Hit_Ratio": feature_hits / n_iterations,
        }
    ).sort_values(by="Hits", ascending=False)

    report["Status"] = np.where(
        report["Hit_Ratio"] > 0.5,
        "Confirmed",
        np.where(report["Hit_Ratio"] > 0.2, "Tentative", "Rejected"),
    )

    return report


def custom_rfe_with_early_stopping(X, y, step=1, n_splits=3):
    features = list(X.columns)
    history = []

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pos_weight = (len(y) - sum(y)) / sum(y)

    while len(features) > 1:
        fold_scores_auc = []
        fold_scores_pr_auc = []
        feature_importances = pd.Series(0, index=features)

        for train_idx, val_idx in cv.split(X[features], y):
            X_tr, X_val = X[features].iloc[train_idx], X[features].iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBClassifier(
                n_estimators=1000,
                n_jobs=-1,
                scale_pos_weight=pos_weight,
                tree_method="hist",
                random_state=42,
                eval_metric="aucpr",
                max_depth=3,
                objective="binary:logistic",
                learning_rate=0.05,
                early_stopping_rounds=10,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            preds = model.predict_proba(X_val)[:, 1]
            fold_scores_auc.append(roc_auc_score(y_val, preds))
            fold_scores_pr_auc.append(average_precision_score(y_val, preds))
            feature_importances += model.feature_importances_

        mean_gini = (2 * np.mean(fold_scores_auc)) - 1
        mean_pr_auc = np.mean(fold_scores_pr_auc)
        history.append(
            {
                "n_features": len(features),
                "gini": mean_gini,
                "pr_auc": mean_pr_auc,
                "features": features.copy(),
            }
        )

        low_importance_vars = feature_importances.nsmallest(step).index.tolist()
        for var in low_importance_vars:
            features.remove(var)

        print(
            f"Features: {len(features) + step} -> Gini: {mean_gini:.4f} -> PR_AUC {mean_pr_auc:.4f} | Removed: {low_importance_vars}"
        )

    return pd.DataFrame(history)


def select_features_with_beeswarm(X, y, n_splits=3):
    """
    SHAP değerlerini hesaplar, önem sırasına dizer ve
    63 değişkenin tamamı için Beeswarm plot çizer.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    shap_importance_df = pd.DataFrame(index=X.columns)
    shap_importance_df["total_importance"] = 0

    all_shap_values = []
    all_X_val = []

    pos_weight = (len(y) - sum(y)) / sum(y)

    print(f"--- SHAP Analizi ve CV Süreci Başladı ({n_splits} Fold) ---")

    for i, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = xgb.XGBClassifier(
            n_estimators=1000,
            n_jobs=-1,
            scale_pos_weight=pos_weight,
            tree_method="hist",
            random_state=42,
            eval_metric="aucpr",
            max_depth=3,
            objective="binary:logistic",
            learning_rate=0.05,
            early_stopping_rounds=20,
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)

        abs_shap = np.abs(shap_values).mean(axis=0)
        shap_importance_df[f"fold_{i + 1}"] = abs_shap
        shap_importance_df["total_importance"] += abs_shap

        all_shap_values.append(shap_values)
        all_X_val.append(X_val)
        print(f"Fold {i + 1} tamamlandı.")

    shap_importance_df["mean_importance"] = (
        shap_importance_df["total_importance"] / n_splits
    )
    shap_importance_df = shap_importance_df.sort_values(
        by="mean_importance", ascending=False
    )

    combined_shap_values = np.vstack(all_shap_values)
    combined_X_val = pd.concat(all_X_val)

    print("\n--- Grafik Oluşturuluyor... ---")

    plt.figure(figsize=(16, 24))

    shap.summary_plot(
        combined_shap_values,
        combined_X_val,
        plot_type="dot",
        max_display=len(X.columns),
        show=False,
    )

    plt.title("PD Modeli SHAP Beeswarm Analizi (Tüm Değişkenler)", fontsize=16)
    plt.show()

    return shap_importance_df
