from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostClassifier
from pandas.io.formats.style import Styler
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
)


def perform_shap_analysis(
    X: pd.DataFrame, y: pd.Series, best_params: Dict[str, Any]
) -> np.ndarray:
    """
    Catboost modelini eğitir ve SHAP değerlerini hesaplayarak özet grafik çizer.
    """
    print("Final model en iyi parametrelerle eğitiliyor...")
    model = CatBoostClassifier(**best_params, verbose=0, allow_writing_files=False)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # CatBoost binary classification çıktısı list veya array olabilir
    if isinstance(shap_values, list) and len(shap_values) > 1:
        target_shap_values = shap_values[1]
    else:
        target_shap_values = shap_values

    print("\nSHAP Summary Plot oluşturuluyor...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(target_shap_values, X, plot_type="dot", show=False)
    plt.title("SHAP Summary Plot (Impact on Target 1)")
    plt.tight_layout()
    plt.show()

    return target_shap_values


def check_calibration(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, n_bins: int = 10
) -> float:
    """
    Modelin kalibrasyonunu kontrol eder ve Reliability Diagram çizer.
    """
    y_probs = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=n_bins)
    model_score: float = brier_score_loss(y_test, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Mükemmel Kalibrasyon (Ideal)")
    plt.plot(
        prob_pred, prob_true, marker=".", label=f"CatBoost (Brier: {model_score:.4f})"
    )

    plt.xlabel("Tahmin Edilen Ortalama Olasılık")
    plt.ylabel("Gerçekleşme Oranı (Fraction of Positives)")
    plt.title("Reliability Diagram (Calibration Curve)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model_score


def plot_cm_with_optimal_pr_threshold(
    y_true: pd.Series, y_probs: np.ndarray, labels: Optional[List[str]] = None
) -> float:
    """
    Belirli bir Recall hedefine (%70) göre optimal eşiği bulur ve CM çizer.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    # Recall >= 0.70 olan en yüksek eşiği bul
    desired_recall = 0.70
    idx = np.where(recalls >= desired_recall)[0]

    if len(idx) > 0:
        valid_thresholds = thresholds[: len(recalls) - 1]
        opt_threshold = valid_thresholds[min(idx[-1], len(valid_thresholds) - 1)]
    else:
        opt_threshold = 0.5

    y_pred = (y_probs >= opt_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            perc = (count / cm_sum[i, 0]) * 100 if cm_sum[i, 0] > 0 else 0
            annot[i, j] = f"{count}\n({perc:.1f}%)"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="RdPu",
        xticklabels=labels if labels else True,
        yticklabels=labels if labels else True,
    )

    plt.title(
        f"Optimal Threshold Confusion Matrix\n"
        f"Calculated Threshold: {opt_threshold:.4f} (Recall Target: {desired_recall})"
    )
    plt.xlabel("Tahmin Edilen (Predicted)")
    plt.ylabel("Gerçek Değer (Actual)")
    plt.tight_layout()
    plt.show()

    return float(opt_threshold)


def create_risk_buckets(
    y_true: pd.Series, y_probs: np.ndarray, n_buckets: int = 10
) -> pd.DataFrame:
    """
    Tahmin olasılıklarını dilimlere (bucket) ayırarak risk raporu oluşturur.
    """
    df = pd.DataFrame({"true": y_true, "prob": y_probs})
    df["bucket"] = pd.qcut(df["prob"], n_buckets, labels=False, duplicates="drop")

    report = (
        df.groupby("bucket")
        .agg(
            min_prob=("prob", "min"),
            max_prob=("prob", "max"),
            actual_default_rate=("true", "mean"),
            count=("true", "count"),
        )
        .sort_index(ascending=False)
    )

    return report


def display_styled_risk_report(report: pd.DataFrame) -> Styler:
    """
    Risk raporunu renklendirilmiş bir Pandas Styler nesnesi olarak döndürür.
    """
    styled_report = report.rename(
        columns={
            "min_prob": "Min Olasılık",
            "max_prob": "Max Olasılık",
            "actual_default_rate": "Gerçekleşen Default Oranı",
            "count": "Müşteri Sayısı",
        }
    )

    return (
        styled_report.style.format(
            {
                "Min Olasılık": "{:.2%}",
                "Max Olasılık": "{:.2%}",
                "Gerçekleşen Default Oranı": "{:.2%}",
                "Müşteri Sayısı": "{:,}",
            }
        )
        .bar(subset=["Gerçekleşen Default Oranı"], color="#d65f5f", vmin=0, vmax=1)
        .background_gradient(subset=["Min Olasılık", "Max Olasılık"], cmap="YlOrRd")
        .set_caption("Risk Grupları (Decile) Analiz Raporu")
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("color", "#2c3e50"),
                        ("font-size", "16px"),
                        ("font-weight", "bold"),
                    ],
                }
            ]
        )
    )
