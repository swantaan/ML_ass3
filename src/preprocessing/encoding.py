from typing import Optional, Union
import numpy as np


class OneHotEncoder:
    def __init__(self, num_classes: Optional[int] = None):
        self.num_classes = num_classes
        self.classes_ = None

    def fit(self, y: np.ndarray) -> "OneHotEncoder":
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.num_classes is None:
            self.num_classes = len(self.classes_)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        n = len(y)
        one_hot = np.zeros((n, self.num_classes), dtype=float)
        for i, val in enumerate(y):
            idx = np.where(self.classes_ == val)[0]
            if len(idx) > 0:
                one_hot[i, idx[0]] = 1.0
        return one_hot

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y_one_hot: np.ndarray) -> np.ndarray:
        indices = np.argmax(y_one_hot, axis=-1)
        return self.classes_[indices]


class LabelEncoder:
    def __init__(self):
        self.classes_ = None

    def fit(self, y: np.ndarray) -> "LabelEncoder":
        self.classes_ = np.unique(y)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        mapping = {c: i for i, c in enumerate(self.classes_)}
        return np.array([mapping[v] for v in y], dtype=int)

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y_encoded: np.ndarray) -> np.ndarray:
        return self.classes_[np.asarray(y_encoded, dtype=int)]
