from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import time

from src.optimizers.base import BaseOptimizer, OptimizerResult


class SGDOptimizer(BaseOptimizer):
    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.9,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        lr_decay: float = 0.0,
        batch_size: Optional[int] = None,
        max_iters: int = 1000,
        tol: float = 1e-6,
        patience: int = 0,
        verbose: bool = False,
    ):
        super().__init__(name="SGD", max_iters=max_iters, tol=tol, verbose=verbose)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.patience = patience

    def minimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        w0: np.ndarray,
        val_cost_fn: Optional[Callable[[np.ndarray], float]] = None,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
    ) -> OptimizerResult:
        start_time = time.perf_counter()
        w = np.copy(w0).astype(float)
        velocity = np.zeros_like(w)

        loss_history: List[float] = []
        val_loss_history: List[float] = []
        iter_history: List[int] = []

        func_evals = 0
        grad_evals = 0

        best_weights = np.copy(w)
        current_loss = cost_fn(w)
        func_evals += 1
        best_loss = current_loss

        loss_history.append(current_loss)
        iter_history.append(0)
        if val_cost_fn is not None:
            val_loss = val_cost_fn(w)
            val_loss_history.append(val_loss)

        converged = False
        message = "Reached maximum iterations."
        no_improve_count = 0

        for k in range(1, self.max_iters + 1):
            current_lr = self.lr / (1.0 + self.lr_decay * (k - 1))

            if self.nesterov and self.momentum > 0.0:
                w_lookahead = w - self.momentum * velocity
                grad = grad_fn(w_lookahead)
                grad_evals += 1
                if self.weight_decay > 0.0:
                    grad = grad + self.weight_decay * w_lookahead
                velocity = self.momentum * velocity + current_lr * grad
                w = w - velocity
            elif self.momentum > 0.0:
                grad = grad_fn(w)
                grad_evals += 1
                if self.weight_decay > 0.0:
                    grad = grad + self.weight_decay * w
                velocity = self.momentum * velocity + current_lr * grad
                w = w - velocity
            else:
                grad = grad_fn(w)
                grad_evals += 1
                if self.weight_decay > 0.0:
                    grad = grad + self.weight_decay * w
                w = w - current_lr * grad

            grad_norm = float(np.linalg.norm(grad))
            current_loss = cost_fn(w)
            func_evals += 1

            loss_history.append(current_loss)
            iter_history.append(k)

            if val_cost_fn is not None:
                val_loss = val_cost_fn(w)
                val_loss_history.append(val_loss)

            if callback is not None:
                callback(k, w, current_loss)

            if current_loss < best_loss - 1e-12:
                best_loss = current_loss
                best_weights = np.copy(w)
                no_improve_count = 0
            else:
                no_improve_count += 1

            if grad_norm < self.tol:
                converged = True
                message = f"Gradient norm ({grad_norm:.3e}) below tolerance ({self.tol:.3e})."
                break

            if self.patience > 0 and no_improve_count >= self.patience:
                converged = True
                message = f"Early stopping triggered: no improvement for {self.patience} iterations."
                break

            if self.verbose and (k % max(1, self.max_iters // 10) == 0 or k == 1):
                print(f"[{self.name}] Iter {k:5d}/{self.max_iters} | Loss: {current_loss:.6e} | GradNorm: {grad_norm:.6e}")

        total_time = time.perf_counter() - start_time
        final_loss = loss_history[-1]

        return OptimizerResult(
            best_weights=best_weights,
            best_loss=best_loss,
            final_weights=w,
            final_loss=final_loss,
            loss_history=loss_history,
            val_loss_history=val_loss_history,
            iteration_history=iter_history,
            func_evals=func_evals,
            grad_evals=grad_evals,
            iterations=len(iter_history) - 1,
            converged=converged,
            wall_time=total_time,
            message=message,
            metadata={"lr": self.lr, "momentum": self.momentum, "nesterov": self.nesterov, "weight_decay": self.weight_decay},
        )

    def train_network(
        self,
        network: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
    ) -> OptimizerResult:
        if self.batch_size is None or self.batch_size >= len(X_train):
            return super().train_network(network, X_train, y_train, X_val, y_val, callback)

        start_time = time.perf_counter()
        w = np.copy(network.get_parameters()).astype(float)
        velocity = np.zeros_like(w)

        n_samples = len(X_train)
        loss_history: List[float] = []
        val_loss_history: List[float] = []
        iter_history: List[int] = []

        func_evals = 0
        grad_evals = 0

        initial_loss = float(network.compute_loss(X_train, y_train))
        func_evals += 1
        loss_history.append(initial_loss)
        iter_history.append(0)

        best_weights = np.copy(w)
        best_loss = initial_loss

        if X_val is not None and y_val is not None:
            val_loss = float(network.compute_loss(X_val, y_val))
            val_loss_history.append(val_loss)

        converged = False
        message = "Reached maximum epochs."
        no_improve_count = 0

        indices = np.arange(n_samples)

        for epoch in range(1, self.max_iters + 1):
            current_lr = self.lr / (1.0 + self.lr_decay * (epoch - 1))
            np.random.shuffle(indices)

            for start_idx in range(0, n_samples, self.batch_size):
                batch_idx = indices[start_idx : start_idx + self.batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                if self.nesterov and self.momentum > 0.0:
                    w_lookahead = w - self.momentum * velocity
                    network.set_parameters(w_lookahead)
                    _, grad = network.compute_loss_and_grad(X_batch, y_batch)
                    grad_evals += 1
                    if self.weight_decay > 0.0:
                        grad = grad + self.weight_decay * w_lookahead
                    velocity = self.momentum * velocity + current_lr * grad
                    w = w - velocity
                elif self.momentum > 0.0:
                    network.set_parameters(w)
                    _, grad = network.compute_loss_and_grad(X_batch, y_batch)
                    grad_evals += 1
                    if self.weight_decay > 0.0:
                        grad = grad + self.weight_decay * w
                    velocity = self.momentum * velocity + current_lr * grad
                    w = w - velocity
                else:
                    network.set_parameters(w)
                    _, grad = network.compute_loss_and_grad(X_batch, y_batch)
                    grad_evals += 1
                    if self.weight_decay > 0.0:
                        grad = grad + self.weight_decay * w
                    w = w - current_lr * grad

            network.set_parameters(w)
            epoch_loss = float(network.compute_loss(X_train, y_train))
            func_evals += 1
            loss_history.append(epoch_loss)
            iter_history.append(epoch)

            if X_val is not None and y_val is not None:
                val_loss = float(network.compute_loss(X_val, y_val))
                val_loss_history.append(val_loss)

            if callback is not None:
                callback(epoch, w, epoch_loss)

            if epoch_loss < best_loss - 1e-12:
                best_loss = epoch_loss
                best_weights = np.copy(w)
                no_improve_count = 0
            else:
                no_improve_count += 1

            if self.patience > 0 and no_improve_count >= self.patience:
                converged = True
                message = f"Early stopping at epoch {epoch}: no improvement for {self.patience} epochs."
                break

            if self.verbose and (epoch % max(1, self.max_iters // 10) == 0 or epoch == 1):
                print(f"[{self.name}] Epoch {epoch:5d}/{self.max_iters} | Loss: {epoch_loss:.6e}")

        network.set_parameters(best_weights)
        total_time = time.perf_counter() - start_time

        return OptimizerResult(
            best_weights=best_weights,
            best_loss=best_loss,
            final_weights=w,
            final_loss=loss_history[-1],
            loss_history=loss_history,
            val_loss_history=val_loss_history,
            iteration_history=iter_history,
            func_evals=func_evals,
            grad_evals=grad_evals,
            iterations=len(iter_history) - 1,
            converged=converged,
            wall_time=total_time,
            message=message,
            metadata={"lr": self.lr, "momentum": self.momentum, "batch_size": self.batch_size, "weight_decay": self.weight_decay},
        )
