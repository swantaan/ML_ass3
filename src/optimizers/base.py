from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import time


@dataclass
class OptimizerResult:
    best_weights: np.ndarray
    best_loss: float
    final_weights: np.ndarray
    final_loss: float
    loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)
    iteration_history: List[int] = field(default_factory=list)
    func_evals: int = 0
    grad_evals: int = 0
    iterations: int = 0
    converged: bool = False
    wall_time: float = 0.0
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseOptimizer(ABC):
    def __init__(self, name: str, max_iters: int = 1000, tol: float = 1e-6, verbose: bool = False):
        self.name = name
        self.max_iters = max_iters
        self.tol = tol
        self.verbose = verbose

    @abstractmethod
    def minimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        w0: np.ndarray,
        val_cost_fn: Optional[Callable[[np.ndarray], float]] = None,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
    ) -> OptimizerResult:
        pass

    def train_network(
        self,
        network: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
    ) -> OptimizerResult:
        w0 = network.get_parameters()

        def cost_fn(w: np.ndarray) -> float:
            network.set_parameters(w)
            return float(network.compute_loss(X_train, y_train))

        def grad_fn(w: np.ndarray) -> np.ndarray:
            network.set_parameters(w)
            _, grad = network.compute_loss_and_grad(X_train, y_train)
            return grad

        val_cost_fn = None
        if X_val is not None and y_val is not None:
            def val_cost_fn(w: np.ndarray) -> float:
                network.set_parameters(w)
                return float(network.compute_loss(X_val, y_val))

        result = self.minimize(
            cost_fn=cost_fn,
            grad_fn=grad_fn,
            w0=w0,
            val_cost_fn=val_cost_fn,
            callback=callback,
        )

        network.set_parameters(result.best_weights)
        return result
