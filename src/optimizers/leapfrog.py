from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import time

from src.optimizers.base import BaseOptimizer, OptimizerResult


class LeapFrogOptimizer(BaseOptimizer):
    def __init__(
        self,
        dt_init: float = 0.5,
        delta: float = 1.0,
        m: int = 3,
        delta1: float = 0.001,
        version: str = "LFOP1",
        im: int = 10,
        in_limit: int = 2,
        max_iters: int = 2000,
        tol: float = 1e-5,
        verbose: bool = False,
    ):
        super().__init__(name="LeapFrog", max_iters=max_iters, tol=tol, verbose=verbose)
        self.dt_init = dt_init
        self.delta = delta
        self.m = m
        self.delta1 = delta1
        self.version = version.upper()
        self.im = im
        self.in_limit = in_limit

    def minimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        w0: np.ndarray,
        val_cost_fn: Optional[Callable[[np.ndarray], float]] = None,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
    ) -> OptimizerResult:
        start_time = time.perf_counter()
        x = np.copy(w0).astype(float)
        dt = float(self.dt_init)
        delta_max = float(self.delta)
        eps_tol = float(self.tol)

        func_evals = 0
        grad_evals = 0

        loss_history: List[float] = []
        val_loss_history: List[float] = []
        iter_history: List[int] = []

        current_loss = cost_fn(x)
        func_evals += 1
        best_loss = current_loss
        best_weights = np.copy(x)

        loss_history.append(current_loss)
        iter_history.append(0)
        if val_cost_fn is not None:
            val_loss = val_cost_fn(x)
            val_loss_history.append(val_loss)

        gf = grad_fn(x)
        grad_evals += 1
        v = -gf * (dt / 2.0)
        grad_norm = float(np.linalg.norm(gf))
        vr = float(np.linalg.norm(v))

        if grad_norm <= eps_tol:
            total_time = time.perf_counter() - start_time
            return OptimizerResult(
                best_weights=best_weights,
                best_loss=best_loss,
                final_weights=x,
                final_loss=current_loss,
                loss_history=loss_history,
                val_loss_history=val_loss_history,
                iteration_history=iter_history,
                func_evals=func_evals,
                grad_evals=grad_evals,
                iterations=0,
                converged=True,
                wall_time=total_time,
                message=f"Initial gradient norm ({grad_norm:.3e}) <= tolerance ({eps_tol:.3e}).",
                metadata={"final_dt": dt, "version": self.version},
            )

        ix = 2
        ind = 0
        is_count = 0
        id_count = 0
        s_count = 0
        n_succ = 1
        kount = 0

        xo = np.copy(x)
        vo = np.copy(v)
        gf_prev = np.copy(gf)

        converged = False
        message = "Reached maximum iterations."

        while kount < self.max_iters:
            dx = vr * dt

            if self.version == "LFOP1B":
                if dx < delta_max:
                    n_succ += 1
                    p_factor = 1.0 + n_succ * self.delta1
                    dt = p_factor * dt
                    vm = vr
                else:
                    dx = delta_max
                    vm = dx / dt
                    if vr > 1e-15:
                        v = v * (vm / vr)

                    if s_count >= self.m:
                        dt = dt / 2.0
                        x = 0.5 * (x + xo)
                        v = 0.25 * (v + vo)
                        vm = float(np.linalg.norm(v))
                        s_count = 0
            else:
                if dx < delta_max:
                    is_count = 0
                else:
                    is_count += 1
                    dx = delta_max

                vm = dx / dt
                if vr > 1e-15:
                    v = v * (vm / vr)

                if is_count >= self.im:
                    id_count += 1
                    if id_count <= self.in_limit:
                        dt = dt / 4.0
                        x = 0.5 * (x + xo)
                        v = 0.25 * (v + vo)
                        vm = float(np.linalg.norm(v))
                    is_count = 0

            xo = np.copy(x)
            vo = np.copy(v)
            x = x + v * dt

            while True:
                gf = grad_fn(x)
                grad_evals += 1
                v = v - gf * dt
                kount += 1

                vr = float(np.linalg.norm(v))
                grad_norm = float(np.linalg.norm(gf))

                current_loss = cost_fn(x)
                func_evals += 1
                loss_history.append(current_loss)
                iter_history.append(kount)

                if val_cost_fn is not None:
                    val_loss = val_cost_fn(x)
                    val_loss_history.append(val_loss)

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_weights = np.copy(x)

                if callback is not None:
                    callback(kount, x, current_loss)

                if self.version == "LFOP1B":
                    grad_dot = float(np.dot(-gf, -gf_prev))
                    if grad_dot > 0.0:
                        s_count = 0
                    else:
                        s_count += 1
                        n_succ = 1
                gf_prev = np.copy(gf)

                if grad_norm <= eps_tol:
                    converged = True
                    message = f"Gradient norm ({grad_norm:.3e}) <= tolerance ({eps_tol:.3e})."
                    break

                if kount >= self.max_iters:
                    break

                if np.isnan(current_loss) or np.isinf(current_loss):
                    message = "Optimization diverged (NaN/Inf loss encountered)."
                    break

                if vr > vm:
                    ind = 0
                    break
                else:
                    x = 0.5 * (xo + x)
                    ind += 1

                    if ind <= ix:
                        v = 0.25 * (v + vo)
                    else:
                        v = np.zeros_like(v)
                        ix = 1

                    vo = np.copy(v)
                    vm = float(np.linalg.norm(v))

            if converged or kount >= self.max_iters:
                break

            if self.verbose and (kount % max(1, self.max_iters // 10) == 0 or kount == 1):
                print(f"[{self.name}] Step {kount:5d}/{self.max_iters} | Loss: {current_loss:.6e} | GradNorm: {grad_norm:.6e} | dt: {dt:.4e}")

        total_time = time.perf_counter() - start_time

        return OptimizerResult(
            best_weights=best_weights,
            best_loss=best_loss,
            final_weights=x,
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
            metadata={"final_dt": dt, "version": self.version},
        )
