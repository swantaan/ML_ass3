import json
import os
from typing import Any, Dict, List, Tuple
import numpy as np

from src.neural_network import NeuralNetwork
from src.optimizers import SGDOptimizer, SCGOptimizer, LeapFrogOptimizer
from src.preprocessing import k_fold_cross_validation


class HyperparameterTuningExperiment:
    def __init__(self, k_folds: int = 3, seed: int = 42):
        self.k_folds = k_folds
        self.seed = seed

    def get_search_grids(self) -> Dict[str, List[Dict[str, Any]]]:
        sgd_grid = []
        for lr in [0.01, 0.05, 0.1]:
            for mom in [0.0, 0.8, 0.9]:
                for bs in [None, 32]:
                    sgd_grid.append({"lr": lr, "momentum": mom, "batch_size": bs})

        scg_grid = []
        for sigma in [1e-5, 1e-4, 1e-3]:
            for lam in [1e-7, 1e-6, 1e-5]:
                scg_grid.append({"sigma": sigma, "lambda_init": lam})

        lf_grid = []
        for dt in [0.05, 0.1, 0.5]:
            for delta in [0.5, 1.0, 5.0]:
                for ver in ["LFOP1", "LFOP1b"]:
                    lf_grid.append({"dt_init": dt, "delta": delta, "version": ver})

        return {
            "SGD": sgd_grid,
            "SCG": scg_grid,
            "LeapFrog": lf_grid,
        }

    def tune_optimizer(
        self,
        opt_name: str,
        param_list: List[Dict[str, Any]],
        X: np.ndarray,
        y: np.ndarray,
        hidden_dim: int,
        problem_type: str,
        n_classes: int = 1,
        max_iters: int = 200,
    ) -> Tuple[Dict[str, Any], float]:
        input_dim = X.shape[1]
        output_dim = n_classes if (problem_type == "classification" and n_classes > 2) else 1
        out_act = "softmax" if (problem_type == "classification" and n_classes > 2) else ("sigmoid" if problem_type == "classification" else "identity")
        loss_fn = "cross_entropy" if problem_type == "classification" else "mse"

        splits = k_fold_cross_validation(
            X, y, k=self.k_folds, stratify=(problem_type == "classification"), seed=self.seed
        )

        best_params = None
        best_score = float("inf")

        for params in param_list:
            fold_losses = []

            for fold_idx, (X_tr, y_tr, X_val, y_val) in enumerate(splits):
                net = NeuralNetwork(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    hidden_activation="sigmoid",
                    output_activation=out_act,
                    loss_fn=loss_fn,
                    seed=self.seed + fold_idx,
                )

                if opt_name == "SGD":
                    opt = SGDOptimizer(max_iters=max_iters, **params)
                elif opt_name == "SCG":
                    opt = SCGOptimizer(max_iters=max_iters, **params)
                elif opt_name == "LeapFrog":
                    opt = LeapFrogOptimizer(max_iters=max_iters, **params)
                else:
                    raise ValueError(f"Unknown optimizer: {opt_name}")

                net.fit(X_tr, y_tr, opt, X_val, y_val)
                loss = net.compute_loss(X_val, y_val)
                fold_losses.append(loss)

            mean_val_loss = float(np.mean(fold_losses))
            if mean_val_loss < best_score:
                best_score = mean_val_loss
                best_params = params

        return best_params, best_score

    def tune_all(
        self,
        datasets: Dict[str, Tuple[np.ndarray, np.ndarray, Dict]],
        hidden_units_map: Dict[str, int],
        output_dir: str = "results/raw",
        max_iters: int = 200,
    ) -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        grids = self.get_search_grids()
        results = {}

        for ds_key, (X, y, info) in datasets.items():
            h_dim = hidden_units_map.get(ds_key, 8)
            prob_type = info["type"]
            n_cls = info.get("n_classes", 1)
            results[ds_key] = {}

            for opt_name, p_list in grids.items():
                best_p, best_s = self.tune_optimizer(
                    opt_name=opt_name,
                    param_list=p_list,
                    X=X,
                    y=y,
                    hidden_dim=h_dim,
                    problem_type=prob_type,
                    n_classes=n_cls,
                    max_iters=max_iters,
                )
                results[ds_key][opt_name] = {
                    "best_params": best_p,
                    "best_val_loss": best_s,
                }

        out_path = os.path.join(output_dir, "tuned_hyperparameters.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        return results
