import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_squared_error,
    r2_score
)


# -------------------------------
# CONFUSION MATRIX (Classification)
# -------------------------------
def plot_confusion_matrix(model, X_test, y_test):
    """
    Returns a matplotlib figure for confusion matrix
    """
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    ax.set_title("Confusion Matrix")
    return fig


# -------------------------------
# REGRESSION RESIDUAL PLOT
# -------------------------------
def plot_regression_residuals(model, X_test, y_test):
    """
    Returns residual dataframe and matplotlib figure
    """
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--")

    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Regression Residual Plot")

    return fig


# -------------------------------
# QUICK REGRESSION METRICS
# -------------------------------
def regression_metrics(y_test, y_pred):
    return {
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "R2 Score": round(r2_score(y_test, y_pred), 4)
    }


def plot_clusters_2d(X_2d, labels, title="Cluster Visualization"):
    fig, ax = plt.subplots(figsize=(6, 5))

    scatter = ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=30
    )

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    return fig