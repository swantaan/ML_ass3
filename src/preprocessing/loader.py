from typing import Dict, Tuple
import numpy as np
from sklearn import datasets

from src.preprocessing.scaler import StandardScaler, MinMaxScaler


def load_classification_dataset(name: str, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    name = name.lower()
    if name in ("dataset1", "wine"):
        data = datasets.load_wine()
        X = data.data
        y = data.target
        info = {
            "name": "Wine Recognition",
            "type": "classification",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)),
        }
    elif name in ("dataset2", "moons"):
        X, y = datasets.make_moons(n_samples=500, noise=0.2, random_state=seed)
        info = {
            "name": "Two Moons",
            "type": "classification",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": 2,
        }
    elif name in ("dataset3", "breast_cancer", "cancer"):
        data = datasets.load_breast_cancer()
        X = data.data
        y = data.target
        info = {
            "name": "Breast Cancer Wisconsin",
            "type": "classification",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": 2,
        }
    elif name == "iris":
        data = datasets.load_iris()
        X = data.data
        y = data.target
        info = {
            "name": "Iris Flower",
            "type": "classification",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)),
        }
    else:
        raise ValueError(f"Unknown classification dataset: {name}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, info


def load_regression_dataset(name: str, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    name = name.lower()
    rng = np.random.RandomState(seed)

    if name in ("dataset1", "damped_sine", "sinc"):
        x = np.linspace(-3.0, 3.0, 500).reshape(-1, 1)
        y = np.sinc(x).ravel() + rng.normal(0.0, 0.05, size=500)
        info = {
            "name": "Damped Sinc Function",
            "type": "regression",
            "n_samples": len(x),
            "n_features": 1,
            "target_dim": 1,
        }
        X = x
    elif name in ("dataset2", "friedman1"):
        X, y = datasets.make_friedman1(n_samples=600, n_features=10, noise=0.1, random_state=seed)
        info = {
            "name": "Friedman 1 Non-Linear",
            "type": "regression",
            "n_samples": X.shape[0],
            "n_features": 10,
            "target_dim": 1,
        }
    elif name in ("dataset3", "diabetes"):
        data = datasets.load_diabetes()
        X = data.data
        y = data.target
        info = {
            "name": "Diabetes Progression",
            "type": "regression",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "target_dim": 1,
        }
    else:
        raise ValueError(f"Unknown regression dataset: {name}")

    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(X)

    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    return X, y, info
