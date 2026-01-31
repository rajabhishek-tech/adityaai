from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import numpy as np


def evaluate_models(models, X_test, y_test, task_type):
    results = []

    for name, obj in models.items():
        pipe = obj["pipeline"]
        preds = pipe.predict(X_test)

        if task_type == "Regression":
            score = r2_score(y_test, preds)
            metric = "R²"
        else:
            score = accuracy_score(y_test, preds)
            metric = "Accuracy"

        results.append({
            "Model": name,
            "Metric": metric,
            "Score": round(score, 4),
            "Train Time (s)": obj["train_time"]
        })

    return results
