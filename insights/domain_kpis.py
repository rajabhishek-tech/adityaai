import pandas as pd
import numpy as np

def compute_domain_kpis(df, domain):
    kpis = {}

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        return kpis

    main_col = numeric_cols[0]
    series = df[main_col].dropna()

    if series.empty:
        return kpis

    mean = series.mean()
    std = series.std()
    cv = std / mean if mean != 0 else 0

    if domain == "Retail":
        kpis["Revenue (Avg)"] = round(mean, 2)
        kpis["Sales Volatility"] = round(cv, 2)
        kpis["Stability"] = "Low" if cv > 1 else "High"

    elif domain == "Finance":
        kpis["Risk Index"] = round(cv, 2)
        kpis["Stability Score"] = round(1 / (cv + 0.01), 2)

    elif domain == "Marketing":
        kpis["Conversion Proxy"] = round(mean, 2)
        kpis["Growth Signal"] = "Positive" if series.iloc[-1] > series.iloc[0] else "Negative"

    else:
        kpis["Mean"] = round(mean, 2)
        kpis["Variability"] = round(std, 2)

    return kpis
