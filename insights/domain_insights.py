import pandas as pd
import numpy as np

def generate_business_insights(df, domain):
    insights = []

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        series = df[col].dropna()

        if series.empty:
            continue

        mean = series.mean()
        std = series.std()
        cv = std / mean if mean != 0 else 0

        trend = "stable"
        if len(series) > 1:
            if series.iloc[-1] > series.iloc[0]:
                trend = "increasing"
            elif series.iloc[-1] < series.iloc[0]:
                trend = "decreasing"

        # ---------- DOMAIN INTERPRETATION ----------
        if domain == "Retail":
            if cv > 1:
                msg = f"{col} shows high volatility, possibly due to promotions or seasonal demand."
            else:
                msg = f"{col} is relatively stable, indicating predictable sales behavior."

        elif domain == "Finance":
            if cv > 1:
                msg = f"{col} is highly volatile, indicating financial risk exposure."
            else:
                msg = f"{col} appears stable, suggesting controlled financial performance."

        elif domain == "Marketing":
            if trend == "increasing":
                msg = f"{col} is improving, indicating better campaign performance."
            elif trend == "decreasing":
                msg = f"{col} is declining, which may require campaign optimization."
            else:
                msg = f"{col} remains stable with no major performance change."

        else:
            msg = f"{col} shows a {trend} trend with moderate variability."

        insights.append({
            "Metric": col,
            "Insight": msg
        })

    return pd.DataFrame(insights)
