from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display, display_html
from scipy.stats import kurtosis, skew, spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer  # NaN hatası için eklendi
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


def analyze_data_quality(df: pd.DataFrame, target_col: str) -> None:
    """
    DataFrame'deki Null ve Zero oranlarını genel ve hedef sınıf bazlı
    hesaplayıp profesyonel bir tablo olarak sunar.
    """
    df_0 = df[df[target_col] == 0]
    df_1 = df[df[target_col] == 1]

    def get_ratios(data, count_type="null"):
        if count_type == "null":
            return (data.isnull().sum() / len(data)) * 100
        return ((data == 0).sum() / len(data)) * 100

    report_data = {
        "Dtype": df.dtypes,
        "Null_Gen (%)": get_ratios(df, "null"),
        "Null_T0 (%)": get_ratios(df_0, "null"),
        "Null_T1 (%)": get_ratios(df_1, "null"),
        "Zero_Gen (%)": get_ratios(df, "zero"),
        "Zero_T0 (%)": get_ratios(df_0, "zero"),
        "Zero_T1 (%)": get_ratios(df_1, "zero"),
    }

    report = pd.DataFrame(report_data).sort_values(by="Null_Gen (%)", ascending=False)

    def style_quality(val):
        """Kritiklik seviyesine göre renklendirme."""
        if val > 90:
            return (
                "background-color: #ff4b4b; color: white; font-weight: bold;"  # Kritik
            )
        elif val > 70:
            return "background-color: #ffa500; color: black;"  # Yüksek
        elif val > 30:
            return "background-color: #ffff00; color: black;"  # Orta
        elif val > 0:
            return "background-color: #d4edda; color: black;"  # Düşük
        return "background-color: #2ecc71; color: black;"  # Temiz

    columns_to_style = [col for col in report.columns if "%" in col]

    styled_report = (
        report.style.format({col: "{:.2f}%" for col in columns_to_style})
        .map(style_quality, subset=columns_to_style)
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [("background-color", "#34495e"), ("color", "white")],
                }
            ]
        )
        .set_properties(**{"text-align": "center", "border": "1px solid #dee2e6"})
    )

    display(styled_report)


def analyze_categorical_data(df: pd.DataFrame) -> None:
    """
    Kategorik sütunlardaki eşsiz değer sayılarını (cardinality) ve
    eksik veri oranlarını (null ratio) hesaplayarak renkli bir tablo sunar.
    """
    # Sadece kategorik (object, category, bool) sütunları seç
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns

    if len(cat_cols) == 0:
        print("Veri setinde kategorik değişken bulunamadı.")
        return

    analysis_results = []
    total_rows = len(df)

    for col in cat_cols:
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        null_ratio = (null_count / total_rows) * 100
        mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"

        analysis_results.append(
            {
                "Column Name": col,
                "Dtype": str(df[col].dtype),
                "Unique Count": unique_count,
                "Null Count": null_count,
                "Null Ratio (%)": null_ratio,
                "Mode (Most Frequent)": mode_val,
            }
        )

    report_df = pd.DataFrame(analysis_results).sort_values(
        by="Unique Count", ascending=False
    )

    def style_cardinality(val):
        """Kardinalite (Eşsiz Değer) renklendirmesi."""
        if val > 100:
            return "background-color: #ff4b4b; color: white;"  # Çok Yüksek
        if val > 20:
            return "background-color: #ffa500; color: black;"  # Orta-Yüksek
        if val > 1:
            return "background-color: #ffff00; color: black;"  # Normal
        return "background-color: #2ecc71; color: black;"  # Sabit Değer

    def style_nulls(val):
        """Eksik veri (Null) renklendirmesi."""
        if val > 50:
            return "border: 2px solid red; color: #ff4b4b; font-weight: bold;"
        if val > 0:
            return "color: #ffa500;"
        return "color: #2ecc71;"

    styled_report = (
        report_df.style.format({"Null Ratio (%)": "{:.2f}%"})
        .map(style_cardinality, subset=["Unique Count"])
        .map(style_nulls, subset=["Null Ratio (%)"])
        .set_properties(
            **{
                "text-align": "center",
                "border": "1px solid #dee2e6",
                "font-family": "Arial",
            }
        )
    )

    print(f"--- Kategorik Değişken Özeti ({len(cat_cols)} Sütun) ---")
    display(styled_report)


def display_cat_counts(df: pd.DataFrame, columns: list, n_cols: int = 3) -> None:
    """
    Kategorik değişkenlerin value_counts çıktılarını yan yana,
    renkli barlar ile görselleştirir.
    """
    html_str = ""

    # Tablo stilini ve hizalamasını belirleyen CSS öznitelikleri
    table_props = (
        "style='display:inline; margin-right:20px; "
        "vertical-align:top; border: 1px solid #dee2e6;'"
    )

    for i, col in enumerate(columns):
        counts_df = df[col].value_counts().to_frame()
        current_col_name = counts_df.columns[0]  # Dinamik sütun ismi alımı

        styled_df = (
            counts_df.style.bar(color="#d4edda", subset=[current_col_name])
            .set_table_attributes(table_props)
            .set_caption(f"<b style='color: #2c3e50;'>Sütun: {col}</b>")
        )

        html_str += styled_df.to_html()

        if (i + 1) % n_cols == 0:
            html_str += "<br><br>"

    display_html(html_str, raw=True)


def plot_target_distribution(df: pd.DataFrame, target_col: str) -> None:
    """
    Hedef değişkenin dağılımını bar chart olarak çizer,
    sütunların üzerine adet ve yüzde bilgilerini ekler.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))

    ax = sns.countplot(
        data=df, x=target_col, palette="viridis", hue=target_col, legend=False
    )

    total = len(df)

    for p in ax.patches:
        height = p.get_height()
        percentage = f"{100 * height / total:.2f}%"

        ax.annotate(
            f"{int(height)}\n({percentage})",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.title(
        f"Target Distribution: {target_col}", fontsize=15, pad=20, fontweight="bold"
    )
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    sns.despine(left=True, bottom=True)

    plt.show()


def perform_univariate_analysis(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Sayısal ve kategorik değişkenler için hedef değişken odaklı
    tek değişkenli analiz (EDA) yapar.
    """

    sns.set_theme(style="whitegrid")
    num_summary = []

    # Sayısal Değişken Analizi
    print("--- Sayısal Değişken Analizi (Target Split) ---")
    for col in num_cols:
        col_data = df[col].dropna()
        col_skew = skew(col_data)

        num_summary.append(
            {
                "Variable": col,
                "Mean": col_data.mean(),
                "Skewness": col_skew,
                "Kurtosis": kurtosis(col_data),
                "Missing": df[col].isnull().sum(),
            }
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.histplot(
            data=df,
            x=col,
            hue=target_col,
            kde=True,
            element="step",
            palette="viridis",
            ax=axes[0],
        )
        axes[0].set_title(f"{col} Dağılımı (Skew: {col_skew:.2f})")

        sns.boxplot(
            data=df, x=col, y=target_col, palette="magma", orient="h", ax=axes[1]
        )
        axes[1].set_title(f"{col} by {target_col}")

        plt.tight_layout()
        plt.show()

    # Kategorik Değişken Analizi
    print("\n--- Kategorik Değişken Analizi (Percentage Based) ---")
    for col in cat_cols:
        if target_col:
            temp_df = (
                df.groupby(col)[target_col]
                .value_counts(normalize=True)
                .rename("percentage")
                .reset_index()
            )

            plt.figure(figsize=(10, 5))
            sns.barplot(
                data=temp_df, x=col, y="percentage", hue=target_col, palette="viridis"
            )

            # Yüzdelik etiketleri ekleme
            for p in plt.gca().patches:
                plt.gca().annotate(
                    f"{p.get_height():.1%}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    xytext=(0, 5),
                    textcoords="offset points",
                )

            plt.title(f"{col} - Hedef Değişken Yüzdesel Dağılımı")
        else:
            plt.figure(figsize=(8, 4))
            sns.countplot(data=df, x=col, palette="pastel")
            plt.title(f"{col} Frekans Tablosu")

        plt.xticks(rotation=45)
        plt.show()

    return pd.DataFrame(num_summary)


def analyze_associations(
    df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], target_col: str
) -> pd.DataFrame:
    """
    Kapsamlı korelasyon ve bağımlılık analizi yapar.
    VIF ile multicollinearity kontrolü gerçekleştirir.
    """

    def plot_readable_correlation(
        df: pd.DataFrame, num_cols: List[str], threshold: float = 0.5
    ):
        """
        Sadece belirlenen eşiğin üzerindeki anlamlı korelasyonları
        okunabilir bir heatmap ile görselleştirir.
        """
        corr_matrix = df[num_cols].corr(method="spearman")

        # Kendisiyle olan korelasyonları (1.0) ve eşik altını maskeliyoruz
        mask_low = np.abs(corr_matrix) < threshold
        filtered_corr = corr_matrix.mask(mask_low)

        # Üst üçgeni maskele (Redundant bilgiyi kaldır)
        # Matris simetrik olduğu için alt taraf yeterlidir.
        mask_upper = np.triu(np.ones_like(filtered_corr, dtype=bool))

        keep_cols = filtered_corr.columns[filtered_corr.notna().any()]
        final_corr = filtered_corr.loc[keep_cols, keep_cols]
        final_mask = mask_upper[
            np.ix_(
                [corr_matrix.columns.get_loc(c) for c in keep_cols],
                [corr_matrix.columns.get_loc(c) for c in keep_cols],
            )
        ]

        if final_corr.empty:
            print(f"Eşik değerinin ({threshold}) üzerinde korelasyon bulunamadı.")
            return

        plt.figure(
            figsize=(max(10, len(keep_cols) * 0.8), max(8, len(keep_cols) * 0.6))
        )

        sns.heatmap(
            final_corr,
            mask=final_mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )

        plt.title(f"Anlamlı Spearman Korelasyonları (|r| > {threshold})", fontsize=15)
        plt.xticks(rotation=45, ha="right")
        plt.show()

    # Sayısal Korelasyon (Spearman)
    print("--- Numeric-Numeric Association (Spearman) ---")
    if num_cols:
        plot_readable_correlation(df, num_cols)

    # Multicollinearity Check (VIF) -
    print("\n--- Multicollinearity Check (VIF) ---")
    vif_df = pd.DataFrame()
    vif_df["Feature"] = num_cols

    if len(num_cols) > 1:
        vif_data = df[num_cols].copy()

        for col in num_cols:
            vif_data[col] = vif_data[col].fillna(vif_data[col].mean())

        scaler = StandardScaler()
        vif_scaled = scaler.fit_transform(vif_data)

        try:
            vif_df["VIF"] = [
                variance_inflation_factor(vif_scaled, i)
                for i in range(vif_scaled.shape[1])
            ]

            high_vif = vif_df[vif_df["VIF"] > 5].sort_values(by="VIF", ascending=False)
            if not high_vif.empty:
                print(f"(!) Uyarı: Yüksek Multicollinearity tespit edildi:\n{high_vif}")
            else:
                print("Kritik seviyede multicollinearity bulunamadı.")

        except Exception as e:
            print(f"(!) VIF hesaplanırken bir hata oluştu: {e}")
    else:
        print("VIF hesaplamak için en az 2 sayısal değişken gereklidir.")


def calculate_iv_and_woe(
    df: pd.DataFrame, target: str, feature: str, bins: int = 10
) -> float:
    """
    Belirli bir değişken için Information Value (IV) hesaplar.
    Sayısal değişkenleri qcut ile gruplandırır.
    """
    temp_df = df[[feature, target]].copy()

    if temp_df[feature].dtype in [np.float64, np.int64]:
        try:
            temp_df["bin"] = pd.qcut(temp_df[feature], q=bins, duplicates="drop")
        except ValueError:
            temp_df["bin"] = pd.cut(temp_df[feature], bins=bins)
    else:
        temp_df["bin"] = temp_df[feature].fillna("Missing")

    # WoE ve IV hesaplama
    count_df = temp_df.groupby("bin", observed=True)[target].agg(["count", "sum"])
    count_df.columns = ["total", "events"]
    count_df["non_events"] = count_df["total"] - count_df["events"]

    # 0 değerlerini handle etme (log hatası almamak için)
    count_df["events"] = count_df["events"].replace(0, 0.5)
    count_df["non_events"] = count_df["non_events"].replace(0, 0.5)

    dist_events = count_df["events"] / count_df["events"].sum()
    dist_non_events = count_df["non_events"] / count_df["non_events"].sum()

    woe = np.log(dist_events / dist_non_events)
    iv = (dist_events - dist_non_events) * woe

    return iv.sum()


def analyze_feature_importance(
    df: pd.DataFrame, features: List[str], target: str
) -> pd.DataFrame:
    """
    Değişkenlerin target ile korelasyonunu (Spearman) ve
    Information Value (IV) değerlerini hesaplayıp raporlar.
    """
    results = []

    for col in features:
        # Spearman Korelasyonu (Sadece sayısal değişkenler için)
        corr_val = np.nan
        if df[col].dtype in [np.float64, np.int64]:
            temp_df = df[[col, target]].dropna()
            corr_val, _ = spearmanr(temp_df[col], temp_df[target])

        # Information Value
        iv_val = calculate_iv_and_woe(df, target, col)

        # IV Yorumlama
        if iv_val < 0.02:
            desc = "Useless"
        elif iv_val < 0.1:
            desc = "Weak"
        elif iv_val < 0.3:
            desc = "Medium"
        elif iv_val < 0.5:
            desc = "Strong"
        else:
            desc = "Suspicious (Leakage?)"

        results.append(
            {
                "Feature": col,
                "Spearman_Corr": corr_val,
                "IV": iv_val,
                "Predictive_Power": desc,
            }
        )

    final_df = pd.DataFrame(results).sort_values(by="IV", ascending=False)
    return final_df


def analyze_outliers(
    df: pd.DataFrame,
    num_cols: List[str],
    target_col: Optional[str] = None,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """
    Sayısal değişkenler için IQR ve Isolation Forest kullanarak
    aykırı değer analizi yapar.
    """
    outlier_report = []

    # İstatistiksel Analiz (IQR Yöntemi)
    print("--- İstatistiksel Outlier Özeti (IQR) ---")
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)

        outlier_report.append(
            {
                "Variable": col,
                "Lower Bound": lower_bound,
                "Upper Bound": upper_bound,
                "Outlier Count": outlier_count,
                "Percentage (%)": (outlier_count / len(df)) * 100,
            }
        )

    # Modern Yaklaşım: Isolation Forest (Çok Boyutlu)
    # Verideki genel anomali skorunu hesaplar
    iso_forest = IsolationForest(
        contamination=contamination, random_state=42, n_jobs=-1
    )

    # NaN değerleri geçici olarak median ile dolduralım
    temp_data = df[num_cols].fillna(df[num_cols].median())
    outlier_preds = iso_forest.fit_predict(temp_data)

    # -1 aykırı, 1 normal değerdir
    df_result = df.copy()
    df_result["is_anomaly"] = np.where(outlier_preds == -1, 1, 0)

    for col in num_cols:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.boxplot(data=df, x=target_col, y=col, palette="Set2")
        plt.title(f"{col} - Boxplot by Target")

        plt.subplot(1, 2, 2)
        sns.stripplot(data=df, x=target_col, y=col, color="red", alpha=0.3)
        plt.title(f"{col} - Individual Data Points")

        plt.tight_layout()
        plt.show()

    return pd.DataFrame(outlier_report).sort_values(
        by="Percentage (%)", ascending=False
    ), df_result


def get_lof_outliers(df, features, n_neighbors=20, contamination=0.01):
    """
    NaN değerleri temizleyerek Local Outlier Factor analizi yapar.
    """

    X = df[features].copy()

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Ölçeklendirme (LOF için mesafe hesabı kritik)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1
    )

    # Tahminleri al: 1 = normal, -1 = aykırı
    outlier_labels = lof.fit_predict(X_scaled)

    lof_scores = -lof.negative_outlier_factor_

    df_result = df.copy()
    df_result["lof_score"] = lof_scores
    df_result["is_outlier"] = outlier_labels

    outliers_df = df_result[df_result["is_outlier"] == -1].sort_values(
        by="lof_score", ascending=False
    )

    print("--- LOF Analizi Tamamlandı ---")
    print(f"İşlenen Değişken Sayısı: {len(features)}")
    print(f"Tespit Edilen Aykırı Sayısı: {len(outliers_df)}")

    return outliers_df, df_result


def calculate_iv(df, feature, target):
    data = df[[feature, target]].copy()

    if data[feature].dtype != "object" and data[feature].nunique() > 10:
        data[feature] = pd.qcut(data[feature], q=10, duplicates="drop").astype(str)
    else:
        data[feature] = data[feature].astype(str)

    grouped = data.groupby(feature)[target].agg(["count", "sum"])
    grouped.columns = ["Total", "Events"]
    grouped["Non-Events"] = grouped["Total"] - grouped["Events"]

    total_events = grouped["Events"].sum()
    total_non_events = grouped["Non-Events"].sum()

    if total_events == 0 or total_non_events == 0:
        return 0

    grouped["Event_Rate"] = grouped["Events"] / total_events
    grouped["Non_Event_Rate"] = grouped["Non-Events"] / total_non_events

    grouped["WOE"] = np.log(
        (grouped["Non_Event_Rate"] + 0.0001) / (grouped["Event_Rate"] + 0.0001)
    )
    grouped["IV"] = (grouped["Non_Event_Rate"] - grouped["Event_Rate"]) * grouped["WOE"]

    return grouped["IV"].sum()


def plot_stripplot_comparison(df_orig, df_clean, num_cols, target_col):
    """
    Orijinal ve temizlenmiş veriyi stripplot kullanarak yan yana kıyaslar.
    """
    sns.set_style("whitegrid")

    for col in num_cols:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)

        # --- 1. Panel: Orijinal Veri ---
        sns.stripplot(
            data=df_orig,
            x=target_col,
            y=col,
            ax=axes[0],
            palette="Blues",
            alpha=0.4,
            jitter=0.3,
            size=3,
        )
        axes[0].set_title(
            f"Orijinal: {col}\n(Aykırı Değerler Mevcut)", fontsize=13, fontweight="bold"
        )

        # --- 2. Panel: Temizlenmiş Veri ---
        sns.stripplot(
            data=df_clean,
            x=target_col,
            y=col,
            ax=axes[1],
            palette="Reds",
            alpha=0.4,
            jitter=0.3,
            size=3,
        )
        axes[1].set_title(
            f"Temizlenmiş: {col}\n(LOF Sonrası)", fontsize=13, fontweight="bold"
        )

        for ax in axes:
            ax.set_xlabel("Target (0: İyi, 1: Kötü)")
            ax.set_ylabel(col)

        plt.tight_layout()
        plt.show()
