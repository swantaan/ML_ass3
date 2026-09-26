from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass

    @abstractmethod
    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass


class MSELoss(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_true = np.asarray(y_true).reshape(y_pred.shape)
        return float(0.5 * np.mean(np.sum((y_pred - y_true) ** 2, axis=-1)))

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_true = np.asarray(y_true).reshape(y_pred.shape)
        n = y_pred.shape[0]
        return (y_pred - y_true) / max(1, n)


class CrossEntropyLoss(Loss):
    def __init__(self, eps: float = 1e-15):
        self.eps = eps

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_true = np.asarray(y_true)
        if y_true.ndim == 1 and y_pred.ndim == 2 and y_pred.shape[1] > 1:
            n_classes = y_pred.shape[1]
            y_one_hot = np.zeros_like(y_pred)
            y_one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
            y_true = y_one_hot
        else:
            y_true = y_true.reshape(y_pred.shape)

        p = np.clip(y_pred, self.eps, 1.0 - self.eps)
        if p.shape[-1] == 1:
            loss = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        else:
            loss = -np.sum(y_true * np.log(p), axis=-1)
        return float(np.mean(loss))

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_true = np.asarray(y_true)
        if y_true.ndim == 1 and y_pred.ndim == 2 and y_pred.shape[1] > 1:
            n_classes = y_pred.shape[1]
            y_one_hot = np.zeros_like(y_pred)
            y_one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
            y_true = y_one_hot
        else:
            y_true = y_true.reshape(y_pred.shape)

        n = y_pred.shape[0]
        p = np.clip(y_pred, self.eps, 1.0 - self.eps)
        if p.shape[-1] == 1:
            return ((p - y_true) / (p * (1.0 - p))) / max(1, n)
        return ((p - y_true) / p) / max(1, n)


def get_loss(name: str) -> Loss:
    name = name.lower()
    if name in ("mse", "mean_squared_error"):
        return MSELoss()
    elif name in ("cross_entropy", "ce", "log_loss"):
        return CrossEntropyLoss()
    raise ValueError(f"Unknown loss: {name}")
