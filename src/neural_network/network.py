from typing import Any, List, Optional, Tuple, Union
import numpy as np

from src.neural_network.activation import Activation, get_activation, Softmax
from src.neural_network.loss import Loss, get_loss, MSELoss, CrossEntropyLoss
from src.neural_network.layers import DenseLayer


class NeuralNetwork:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_activation: Union[str, Activation] = "sigmoid",
        output_activation: Union[str, Activation] = "identity",
        loss_fn: Union[str, Loss] = "mse",
        weight_decay: float = 0.0,
        init_type: str = "xavier",
        seed: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weight_decay = weight_decay

        self.rng = np.random.RandomState(seed)

        if isinstance(hidden_activation, str):
            self.hidden_act = get_activation(hidden_activation)
        else:
            self.hidden_act = hidden_activation

        if isinstance(output_activation, str):
            self.output_act = get_activation(output_activation)
        else:
            self.output_act = output_activation

        if isinstance(loss_fn, str):
            self.loss_fn = get_loss(loss_fn)
        else:
            self.loss_fn = loss_fn

        self.layer1 = DenseLayer(input_dim, hidden_dim, init_type=init_type, rng=self.rng)
        self.layer2 = DenseLayer(hidden_dim, output_dim, init_type=init_type, rng=self.rng)

        self.layers = [self.layer1, self.layer2]

        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        self.z1 = self.layer1.forward(x)
        self.a1 = self.hidden_act.forward(self.z1)
        self.z2 = self.layer2.forward(self.a1)
        self.a2 = self.output_act.forward(self.z2)
        return self.a2

    def compute_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.forward(x)
        base_loss = self.loss_fn.forward(y_pred, y)
        if self.weight_decay > 0.0:
            reg_loss = 0.5 * self.weight_decay * (
                np.sum(self.layer1.W ** 2) + np.sum(self.layer2.W ** 2)
            )
            return float(base_loss + reg_loss)
        return float(base_loss)

    def compute_loss_and_grad(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)

        y_pred = self.forward(x)
        base_loss = self.loss_fn.forward(y_pred, y)

        n_samples = x.shape[0]

        is_softmax_ce = isinstance(self.output_act, Softmax) and isinstance(self.loss_fn, CrossEntropyLoss)
        if is_softmax_ce:
            if y.ndim == 1 and self.output_dim > 1:
                y_one_hot = np.zeros_like(y_pred)
                y_one_hot[np.arange(n_samples), y.astype(int)] = 1.0
                target = y_one_hot
            else:
                target = y.reshape(y_pred.shape)
            dz2 = (y_pred - target) / max(1, n_samples)
        else:
            d_loss = self.loss_fn.gradient(y_pred, y)
            d_act2 = self.output_act.derivative(self.z2, self.a2)
            dz2 = d_loss * d_act2

        da1 = self.layer2.backward(dz2)

        dz1 = da1 * self.hidden_act.derivative(self.z1, self.a1)
        self.layer1.backward(dz1)

        if self.weight_decay > 0.0:
            self.layer1.dW += self.weight_decay * self.layer1.W
            self.layer2.dW += self.weight_decay * self.layer2.W
            reg_loss = 0.5 * self.weight_decay * (
                np.sum(self.layer1.W ** 2) + np.sum(self.layer2.W ** 2)
            )
            total_loss = float(base_loss + reg_loss)
        else:
            total_loss = float(base_loss)

        grad_flat = np.concatenate([self.layer1.get_gradients(), self.layer2.get_gradients()])
        return total_loss, grad_flat

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.forward(x)
        if self.output_dim == 1:
            if isinstance(self.loss_fn, CrossEntropyLoss):
                return (probs >= 0.5).astype(int).ravel()
            return probs.ravel()
        return np.argmax(probs, axis=-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def get_parameters(self) -> np.ndarray:
        return np.concatenate([layer.get_parameters() for layer in self.layers])

    def set_parameters(self, flat_params: np.ndarray) -> None:
        idx = 0
        for layer in self.layers:
            num = layer.num_parameters()
            layer.set_parameters(flat_params[idx : idx + num])
            idx += num

    def num_parameters(self) -> int:
        return sum(layer.num_parameters() for layer in self.layers)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimizer: Any,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        callback: Optional[Any] = None,
    ) -> Any:
        return optimizer.train_network(self, X_train, y_train, X_val, y_val, callback=callback)
