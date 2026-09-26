from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import time

from src.optimizers.base import BaseOptimizer, OptimizerResult


class SCGOptimizer(BaseOptimizer):
    def __init__(
        self,
        sigma: float = 1e-4,
        lambda_init: float = 1e-6,
        lambda_min: float = 1e-15,
        lambda_max: float = 1e10,
        max_iters: int = 1000,
        tol: float = 1e-6,
        verbose: bool = False,
    ):
        super().__init__(name="SCG", max_iters=max_iters, tol=tol, verbose=verbose)
        self.sigma = sigma
        self.lambda_init = lambda_init
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

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
        N = len(w)

        func_evals = 0
        grad_evals = 0

        loss_history: List[float] = []
        val_loss_history: List[float] = []
        iter_history: List[int] = []

        current_loss = cost_fn(w)
        func_evals += 1

        grad = grad_fn(w)
        grad_evals += 1

        r = -grad
        p = np.copy(r)
        k = 1
        success = True

        lambda_val = float(self.lambda_init)
        lambda_bar = 0.0

        best_weights = np.copy(w)
        best_loss = current_loss

        loss_history.append(current_loss)
        iter_history.append(0)
        if val_cost_fn is not None:
            val_loss = val_cost_fn(w)
            val_loss_history.append(val_loss)

        converged = False
        message = "Reached maximum iterations."

        grad_norm = float(np.linalg.norm(r))
        if grad_norm < self.tol:
            converged = True
            message = f"Initial gradient norm ({grad_norm:.3e}) below tolerance ({self.tol:.3e})."
            total_time = time.perf_counter() - start_time
            return OptimizerResult(
                best_weights=best_weights,
                best_loss=best_loss,
                final_weights=w,
                final_loss=current_loss,
                loss_history=loss_history,
                val_loss_history=val_loss_history,
                iteration_history=iter_history,
                func_evals=func_evals,
                grad_evals=grad_evals,
                iterations=0,
                converged=converged,
                wall_time=total_time,
                message=message,
                metadata={"lambda_final": lambda_val},
            )

        while k <= self.max_iters:
            p_norm_sq = float(np.dot(p, p))
            p_norm = np.sqrt(p_norm_sq)

            if p_norm < 1e-15 or np.isnan(p_norm):
                p = np.copy(r)
                p_norm_sq = float(np.dot(p, p))
                p_norm = np.sqrt(p_norm_sq)
                if p_norm < 1e-15:
                    converged = True
                    message = "Search direction norm vanished."
                    break

            if success:
                sigma_k = self.sigma / p_norm
                w_perturbed = w + sigma_k * p
                grad_perturbed = grad_fn(w_perturbed)
                grad_evals += 1
                s = (grad_perturbed - grad) / sigma_k
                delta = float(np.dot(p, s))

            delta = delta + (lambda_val - lambda_bar) * p_norm_sq

            if delta <= 0.0:
                lambda_bar = 2.0 * (lambda_val - delta / p_norm_sq)
                delta = -delta + lambda_val * p_norm_sq
                lambda_val = lambda_bar

            mu = float(np.dot(p, r))
            if delta != 0.0:
                alpha = mu / delta
            else:
                alpha = 0.0

            w_candidate = w + alpha * p
            candidate_loss = cost_fn(w_candidate)
            func_evals += 1

            if mu != 0.0:
                Delta = 2.0 * delta * (current_loss - candidate_loss) / (mu * mu)
            else:
                Delta = 0.0

            if Delta >= 0.0:
                w = w_candidate
                current_loss = candidate_loss
                grad_new = grad_fn(w)
                grad_evals += 1
                r_new = -grad_new

                lambda_bar = 0.0
                success = True

                loss_history.append(current_loss)
                iter_history.append(k)

                if val_cost_fn is not None:
                    val_loss = val_cost_fn(w)
                    val_loss_history.append(val_loss)

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_weights = np.copy(w)

                if callback is not None:
                    callback(k, w, current_loss)

                grad_norm = float(np.linalg.norm(r_new))
                if grad_norm < self.tol:
                    converged = True
                    message = f"Gradient norm ({grad_norm:.3e}) below tolerance ({self.tol:.3e})."
                    break

                if k % N == 0:
                    p = np.copy(r_new)
                else:
                    beta = (float(np.dot(r_new, r_new)) - float(np.dot(r_new, r))) / mu
                    p = r_new + beta * p
                    if float(np.dot(p, r_new)) <= 0.0:
                        p = np.copy(r_new)

                r = r_new
                grad = grad_new

                if Delta >= 0.75:
                    lambda_val = max(self.lambda_min, 0.25 * lambda_val)
            else:
                lambda_bar = lambda_val
                success = False

            if Delta < 0.25:
                increase = (delta * (1.0 - Delta)) / p_norm_sq
                lambda_val = min(self.lambda_max, lambda_val + increase)

            if self.verbose and (k % max(1, self.max_iters // 10) == 0 or k == 1):
                print(f"[{self.name}] Iter {k:5d}/{self.max_iters} | Loss: {current_loss:.6e} | GradNorm: {grad_norm:.6e} | lambda: {lambda_val:.2e} | Delta: {Delta:.3f}")

            if grad_norm < self.tol:
                converged = True
                message = f"Gradient norm ({grad_norm:.3e}) below tolerance ({self.tol:.3e})."
                break

            k += 1

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
            metadata={"lambda_final": lambda_val, "sigma": self.sigma},
        )
