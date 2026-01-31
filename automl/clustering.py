import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def run_kmeans(X, max_k=10):
    results = []

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)

        score = silhouette_score(X, labels)
        results.append({
            "Model": "KMeans",
            "Clusters": k,
            "Silhouette Score": round(score, 4),
            "Estimator": model,
            "Labels": labels
        })

    return results


def run_dbscan(X, eps_values=[0.3, 0.5, 0.7], min_samples=5):
    results = []

    for eps in eps_values:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)

        # ignore cases with only noise
        unique_labels = set(labels)
        if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
            continue

        score = silhouette_score(X, labels)
        results.append({
            "Model": "DBSCAN",
            "Clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "Silhouette Score": round(score, 4),
            "Estimator": model,
            "Labels": labels,
            "EPS": eps
        })

    return results


def reduce_dimensions(X):
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X)
