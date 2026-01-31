import pandas as pd
import numpy as np


def recommend_ml_strategy(df: pd.DataFrame):
    """
    Analyze dataset and recommend:
    - ML task type
    - Target column
    - Feature columns
    - Train-test split
    - Suitable models
    - Expected performance range
    """

    recommendations = {}

    # -------------------------------
    # 1. Basic dataset stats
    # -------------------------------
    n_rows, n_cols = df.shape
    recommendations["rows"] = n_rows
    recommendations["columns"] = n_cols

    # -------------------------------
    # 2. Detect possible target column
    # -------------------------------
    target_candidates = []

    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)

        # avoid ID-like columns
        if 0.01 < unique_ratio < 0.9:
            target_candidates.append(col)

    target = target_candidates[0] if target_candidates else df.columns[-1]
    recommendations["suggested_target"] = target

    # -------------------------------
    # 3. Detect ML task type (ROBUST)
    # -------------------------------
    unique_ratio = df[target].nunique() / len(df)

    # 🔑 HIGH CARDINALITY TARGET → REGRESSION
    if unique_ratio > 0.5:
        recommendations["task_type"] = "Regression"
        metric = "R² / RMSE"

        override_reason = (
            "Target has very high cardinality "
            f"({df[target].nunique()} unique values). "
            "Treating as regression."
        )

    else:
        if pd.api.types.is_numeric_dtype(df[target]):
            recommendations["task_type"] = "Regression"
            metric = "R² / RMSE"
            override_reason = None
        else:
            n_classes = df[target].nunique()
            if n_classes == 2:
                recommendations["task_type"] = "Binary Classification"
            else:
                recommendations["task_type"] = "Multi-class Classification"
            metric = "Accuracy / F1-score"
            override_reason = None

    recommendations["evaluation_metric"] = metric

    # -------------------------------
    # 4. Recommend feature columns
    # -------------------------------
    feature_cols = []

    for col in df.columns:
        if col == target:
            continue

        if df[col].nunique() <= 1:
            continue

        if df[col].isna().mean() > 0.4:
            continue

        feature_cols.append(col)

    recommendations["recommended_features"] = feature_cols
    recommendations["num_features"] = len(feature_cols)

    # -------------------------------
    # 5. Train-test split recommendation
    # -------------------------------
    if n_rows < 1000:
        split = "70% train / 30% test"
    elif n_rows < 10000:
        split = "80% train / 20% test"
    else:
        split = "85% train / 15% test"

    recommendations["train_test_split"] = split

    # -------------------------------
    # 6. Model recommendations
    # -------------------------------
    if recommendations["task_type"] == "Regression":
        models = [
            "Linear Regression (baseline)",
            "Random Forest Regressor",
            "Gradient Boosting Regressor"
        ]
        expected_perf = "R² ≈ 0.65 – 0.85 (after cleaning & tuning)"
    else:
        models = [
            "Logistic Regression (baseline)",
            "Random Forest Classifier",
            "Gradient Boosting Classifier"
        ]
        expected_perf = "Accuracy ≈ 70% – 90% (after cleaning & tuning)"

    recommendations["recommended_models"] = models
    recommendations["expected_performance"] = expected_perf

    # -------------------------------
    # 7. Readiness warnings
    # -------------------------------
    warnings = []

    if df.isna().mean().mean() > 0.2:
        warnings.append("High missing values detected — imputation required.")

    if recommendations["num_features"] < 3:
        warnings.append("Very few usable features — model performance may be limited.")

    if unique_ratio > 0.5:
        warnings.append(
            "Target has very high cardinality; classification may be misleading. "
            "Regression selected automatically."
        )

    recommendations["warnings"] = warnings

    return recommendations
