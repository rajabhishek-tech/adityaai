import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from collections import Counter


def prepare_data(df, target, task_type):
    df = df.dropna(subset=[target]).copy()

    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # -------- SAFE CATEGORICAL HANDLING --------
    cat_cols = []
    for col in X.select_dtypes(include=["object", "category"]).columns:
        unique_ratio = X[col].nunique() / len(X)

        if unique_ratio > 0.05 or X[col].nunique() > 50:
            continue  # drop ID-like columns

        X[col] = X[col].astype(str)
        cat_cols.append(col)

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    stratify_col = None
    if task_type != "Regression":
        class_counts = Counter(y)
        if min(class_counts.values()) >= 2:
            stratify_col = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_col
    )

    return preprocessor, X_train, X_test, y_train, y_test
