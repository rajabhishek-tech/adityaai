from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
import time


def train_models(preprocessor, X_train, y_train, task_type):
    models = {}

    if task_type == "Regression":
        candidates = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
        }
    else:
        candidates = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42)
        }

    trained = {}

    for name, model in candidates.items():
        start = time.time()

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        train_time = round(time.time() - start, 2)

        trained[name] = {
            "pipeline": pipe,
            "train_time": train_time
        }

    return trained
