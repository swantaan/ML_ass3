from typing import Optional
import numpy as np


class StandardScaler:
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        x = np.asarray(x, dtype=float)
        if self.with_mean:
            self.mean_ = np.mean(x, axis=0)
        else:
            self.mean_ = np.zeros(x.shape[1], dtype=float)

        if self.with_std:
            self.std_ = np.std(x, axis=0)
            self.std_[self.std_ < 1e-12] = 1.0
        else:
            self.std_ = np.ones(x.shape[1], dtype=float)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x * self.std_ + self.mean_


class MinMaxScaler:
    def __init__(self, feature_range: tuple = (0.0, 1.0)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, x: np.ndarray) -> "MinMaxScaler":
        x = np.asarray(x, dtype=float)
        self.data_min_ = np.min(x, axis=0)
        self.data_max_ = np.max(x, axis=0)
        diff = self.data_max_ - self.data_min_
        diff[diff < 1e-12] = 1.0
        self.diff_ = diff
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        res = (x - self.data_min_) / self.diff_
        low, high = self.feature_range
        return res * (high - low) + low

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        low, high = self.feature_range
        unscaled = (x - low) / (high - low)
        return unscaled * self.diff_ + self.data_min_
