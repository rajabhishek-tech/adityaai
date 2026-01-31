import pandas as pd
import numpy as np

def generate_recommendations(df):
    recommendations = []

    n_rows = len(df)

    for col in df.columns:
        col_rec = []
        series = df[col]

        # ---------- BASIC METRICS ----------
        missing_pct = series.isnull().mean()
        unique_ratio = series.nunique() / n_rows

        # ---------- IDENTIFIER-LIKE ----------
        if unique_ratio > 0.9:
            col_rec.append("Likely an identifier column")
            col_rec.append("Exclude from modeling")
            col_rec.append("Use only for joins or references")

        # ---------- NUMERIC ----------
        elif pd.api.types.is_numeric_dtype(series):

            col_rec.append("Numeric feature detected")

            # Missing values
            if missing_pct > 0:
                col_rec.append(
                    f"Missing values detected ({missing_pct:.1%}) — consider median imputation"
                )

            # Skewness
            if series.dropna().skew() > 1:
                col_rec.append("Right-skewed distribution — consider log / Box-Cox transform")
            elif series.dropna().skew() < -1:
                col_rec.append("Left-skewed distribution — consider power transform")

            # Outliers (IQR method)
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if iqr > 0:
                outlier_ratio = (
                    ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).mean()
                )
                if outlier_ratio > 0.05:
                    col_rec.append("Significant outliers detected — consider capping or removal")

            # Stability
            if series.mean() != 0:
                cv = series.std() / abs(series.mean())
                if cv > 1:
                    col_rec.append("High variability — feature may be unstable")

        # ---------- DATETIME ----------
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_rec.append("Datetime feature detected")
            col_rec.append("Extract year, month, day, weekday")
            col_rec.append("Consider trend and seasonality analysis")

        # ---------- CATEGORICAL ----------
        else:
            col_rec.append("Categorical feature detected")

            if missing_pct > 0:
                col_rec.append(
                    f"Missing values detected ({missing_pct:.1%}) — fill with 'Unknown'"
                )

            if unique_ratio < 0.05:
                col_rec.append("Low cardinality — suitable for one-hot encoding")
            elif unique_ratio < 0.2:
                col_rec.append("Moderate cardinality — consider target or frequency encoding")
            else:
                col_rec.append("High cardinality — consider grouping rare categories")

        recommendations.append({
            "Column": col,
            "Recommendations": " | ".join(col_rec)
        })

    return pd.DataFrame(recommendations)
