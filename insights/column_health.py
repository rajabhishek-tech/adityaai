import pandas as pd
import numpy as np

def analyze_column_health(df):
    results = []

    for col in df.columns:
        series = df[col]
        issues = []
        recommendations = []
        impact = []

        missing_pct = series.isnull().mean()
        unique_ratio = series.nunique() / len(df)

        # Identifier-like
        if unique_ratio > 0.9:
            issues.append("Likely identifier column")
            recommendations.append("Exclude from modeling")
            impact.append("Reduces noise in models")

        # Numeric checks
        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_ratio = numeric_series.notna().mean()

        if numeric_ratio > 0.8:
            skew = numeric_series.dropna().skew()

            if abs(skew) > 1:
                issues.append("Highly skewed distribution")
                recommendations.append("Apply log or power transformation")
                impact.append("Improves model stability")

            # Outliers (IQR)
            q1 = numeric_series.quantile(0.25)
            q3 = numeric_series.quantile(0.75)
            iqr = q3 - q1
            outlier_ratio = (
                ((numeric_series < q1 - 1.5 * iqr) |
                 (numeric_series > q3 + 1.5 * iqr)).mean()
            )

            if outlier_ratio > 0.05:
                issues.append("Significant outliers present")
                recommendations.append("Cap or remove outliers")
                impact.append("Prevents model bias")

        # Missing values
        if missing_pct > 0.3:
            issues.append("High missing values")
            recommendations.append("Consider imputation or column removal")
            impact.append("Improves data quality")

        # Health score
        if len(issues) == 0:
            health = "🟢 Healthy"
        elif len(issues) <= 2:
            health = "🟡 Needs Attention"
        else:
            health = "🔴 Risky"

        results.append({
            "Column": col,
            "Health": health,
            "Issues": " | ".join(issues) if issues else "None",
            "Recommendations": " | ".join(recommendations) if recommendations else "None",
            "Impact": " | ".join(impact) if impact else "None"
        })

    return pd.DataFrame(results)
