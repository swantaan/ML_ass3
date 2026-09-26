import json
import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn import datasets

from src.preprocessing.scaler import StandardScaler, MinMaxScaler


def load_classification_dataset(name: str, seed: int = 42, base_dir: str = "data/classification") -> Tuple[np.ndarray, np.ndarray, Dict]:
    name = name.lower()
    mapping = {
        "dataset1": "dataset1",
        "wine": "dataset1",
        "dataset2": "dataset2",
        "moons": "dataset2",
        "dataset3": "dataset3",
        "breast_cancer": "dataset3",
        "cancer": "dataset3",
    }
    dir_name = mapping.get(name, name)
    csv_path = os.path.join(base_dir, dir_name, "data.csv")
    meta_path = os.path.join(base_dir, dir_name, "metadata.json")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c != "target"]
        X = df[feature_cols].values.astype(float)
        y = df["target"].values

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                info = json.load(f)
        else:
            info = {
                "name": dir_name,
                "type": "classification",
                "n_samples": len(df),
                "n_features": len(feature_cols),
                "n_classes": len(np.unique(y)),
            }
    else:
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
        else:
            raise ValueError(f"Unknown classification dataset: {name}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, info


def load_regression_dataset(name: str, seed: int = 42, base_dir: str = "data/regression") -> Tuple[np.ndarray, np.ndarray, Dict]:
    name = name.lower()
    mapping = {
        "dataset1": "dataset1",
        "damped_sine": "dataset1",
        "sinc": "dataset1",
        "dataset2": "dataset2",
        "friedman1": "dataset2",
        "dataset3": "dataset3",
        "diabetes": "dataset3",
    }
    dir_name = mapping.get(name, name)
    csv_path = os.path.join(base_dir, dir_name, "data.csv")
    meta_path = os.path.join(base_dir, dir_name, "metadata.json")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c != "target"]
        X = df[feature_cols].values.astype(float)
        y = df["target"].values.astype(float)

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                info = json.load(f)
        else:
            info = {
                "name": dir_name,
                "type": "regression",
                "n_samples": len(df),
                "n_features": len(feature_cols),
                "target_dim": 1,
            }
    else:
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
