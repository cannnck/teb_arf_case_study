import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


def calculate_recall_at_k(y_true, y_probs, k_percent):
    """Belirli bir yüzde dilimindeki (top k%) recall değerini hesaplar."""
    n_threshold = int(len(y_true) * (k_percent / 100))
    # Olasılıkları büyükten küçüğe sıralayıp top k indisi alıyoruz
    top_indices = np.argsort(y_probs)[::-1][:n_threshold]
    # Bu indisteki gerçek 1'lerin, toplam 1'lere oranı
    recall_at_k = np.sum(y_true.iloc[top_indices]) / np.sum(y_true)
    return recall_at_k


def calculate_precision_at_k(y_true, y_probs, k_percent):
    """
    Belirli bir yüzde dilimindeki (top k%) precision değerini hesaplar.
    Precision@k = (Top k içindeki gerçek pozitifler) / (Top k içindeki toplam örnek sayısı)
    """
    # İnceleyeceğimiz örnek sayısını hesaplıyoruz
    n_threshold = int(len(y_true) * (k_percent / 100))

    # Olasılıkları büyükten küçüğe sıralayıp top n indisi alıyoruz
    top_indices = np.argsort(y_probs)[::-1][:n_threshold]

    # Bu indisteki gerçek 1'lerin (True Positives), n_threshold'a oranı
    precision_at_k = np.sum(y_true.iloc[top_indices]) / n_threshold

    return precision_at_k


def plot_pr_curves(plot_data, target_rate):
    plt.figure(figsize=(8, 6))
    plt.axhline(
        y=target_rate, color="r", linestyle="--", label=f"Baseline ({target_rate:.2%})"
    )

    for model_name, data in plot_data.items():
        precision, recall, _ = precision_recall_curve(data["y_true"], data["y_prob"])
        ap = average_precision_score(data["y_true"], data["y_prob"])
        plt.plot(
            recall, precision, color="blue", lw=2, label=f"HistGB (PR-AUC = {ap:.4f})"
        )

    plt.title("Precision-Recall Curve (Out-of-Fold)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def run_cross_validation(X: pd.DataFrame, y: pd.Series, n_splits: int):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pos_weight = (len(y) - sum(y)) / sum(y)

    all_fold_results = []

    plot_data = {
        model: {"y_true": [], "y_prob": []}
        for model in ["XGBoost", "LightGBM", "CatBoost"]
    }

    for model_name in plot_data.keys():
        print(f"\n{'=' * 20} Eğitiliyor: {model_name} {'=' * 20}")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if model_name == "XGBoost":
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
                best_iter = model.best_iteration

            elif model_name == "LightGBM":
                model = lgb.LGBMClassifier(
                    n_jobs=-1,
                    scale_pos_weight=pos_weight,
                    random_state=42,
                    verbosity=-1,
                    importance_type="gain",
                    metric="average_precision",  # PR-AUC,
                    max_depth=3,
                    objective="binary",
                    learning_rate=0.05,
                )
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                best_iter = model.best_iteration_

            elif model_name == "CatBoost":
                model = CatBoostClassifier(
                    n_estimators=1000,
                    verbose=0,
                    scale_pos_weight=pos_weight,
                    allow_writing_files=False,
                    random_state=42,
                    max_depth=3,
                    eval_metric="PRAUC",
                    loss_function="Logloss",
                    learning_rate=0.05,
                    early_stopping_rounds=10,
                )
                model.fit(X_train, y_train, eval_set=(X_val, y_val))
                best_iter = model.get_best_iteration()

            y_val_pred = model.predict_proba(X_val)[:, 1]
            y_train_pred = model.predict_proba(X_train)[:, 1]

            plot_data[model_name]["y_true"].extend(y_val.tolist())
            plot_data[model_name]["y_prob"].extend(y_val_pred.tolist())

            gini_val = 2 * (roc_auc_score(y_val, y_val_pred)) - 1
            gini_train = 2 * (roc_auc_score(y_train, y_train_pred)) - 1
            pr_auc_val = average_precision_score(y_val, y_val_pred)
            pr_auc_train = average_precision_score(y_train, y_train_pred)
            r10 = calculate_recall_at_k(y_val, y_val_pred, 10)
            pr10 = calculate_precision_at_k(y_val, y_val_pred, 10)

            all_fold_results.append(
                {
                    "Model": model_name,
                    "Fold": fold,
                    "Best_Iter": best_iter,
                    "GINI_TRAIN": gini_train,
                    "GINI_VAL": gini_val,
                    "PR_AUC_TRAIN": pr_auc_train,
                    "PR_AUC_VAL": pr_auc_val,
                    "Recall@10": r10,
                    "Precision@10": pr10,
                    "Diff": average_precision_score(y_train, y_train_pred) - pr_auc_val,
                }
            )
            print(
                f"Fold {fold} | GINI-TRAIN: {gini_train:.4f} | GINI-VAL: {gini_val:.4f} |  PR-AUC-TRAIN: {pr_auc_train:.4f} | PR-AUC-VAL: {pr_auc_val:.4f} | R@10: {r10:.4f} | PR@10: {pr10:.4f}"
            )

    results_df = pd.DataFrame(all_fold_results)

    plot_pr_curves(plot_data, target_rate=y.mean())

    return results_df


def plot_catboost_importance(model, features):
    feature_importance = model.get_feature_importance()
    feature_names = features

    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    fi_df = fi_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis")
    plt.title("CatBoost Feature Importance (Loss Function Change)")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
