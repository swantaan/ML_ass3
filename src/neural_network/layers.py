from typing import Tuple
import numpy as np


class DenseLayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_type: str = "xavier",
        rng: np.random.RandomState = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

        if init_type == "xavier":
            limit = np.sqrt(6.0 / (in_features + out_features))
            self.W = self.rng.uniform(-limit, limit, (in_features, out_features))
        elif init_type == "he":
            std = np.sqrt(2.0 / in_features)
            self.W = self.rng.normal(0.0, std, (in_features, out_features))
        else:
            self.W = self.rng.uniform(-0.1, 0.1, (in_features, out_features))

        self.b = np.zeros(out_features, dtype=float)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_cache = x
        return np.dot(x, self.W) + self.b

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        self.dW = np.dot(self.x_cache.T, d_out)
        self.db = np.sum(d_out, axis=0)
        return np.dot(d_out, self.W.T)

    def num_parameters(self) -> int:
        return self.W.size + self.b.size

    def get_parameters(self) -> np.ndarray:
        return np.concatenate([self.W.ravel(), self.b.ravel()])

    def set_parameters(self, flat_params: np.ndarray) -> None:
        w_size = self.W.size
        self.W = flat_params[:w_size].reshape(self.W.shape)
        self.b = flat_params[w_size : w_size + self.b.size].reshape(self.b.shape)

    def get_gradients(self) -> np.ndarray:
        return np.concatenate([self.dW.ravel(), self.db.ravel()])
