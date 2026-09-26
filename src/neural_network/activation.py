from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        pass


class Sigmoid(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def derivative(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        if a is None:
            a = self.forward(z)
        return a * (1.0 - a)


class Tanh(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def derivative(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        if a is None:
            a = self.forward(z)
        return 1.0 - a ** 2


class ReLU(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    def derivative(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        return (z > 0.0).astype(float)


class Identity(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return z

    def derivative(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        return np.ones_like(z)


class Softmax(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        shift_z = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(shift_z)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def derivative(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        if a is None:
            a = self.forward(z)
        return a * (1.0 - a)


def get_activation(name: str) -> Activation:
    name = name.lower()
    if name in ("sigmoid", "logistic"):
        return Sigmoid()
    elif name == "tanh":
        return Tanh()
    elif name == "relu":
        return ReLU()
    elif name in ("identity", "linear", "none"):
        return Identity()
    elif name == "softmax":
        return Softmax()
    raise ValueError(f"Unknown activation: {name}")
