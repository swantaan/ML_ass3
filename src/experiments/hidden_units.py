import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.neural_network import NeuralNetwork
from src.optimizers import SCGOptimizer
from src.preprocessing import train_val_test_split, k_fold_cross_validation


class HiddenUnitsExperiment:
    def __init__(
        self,
        candidate_units: List[int] = None,
        k_folds: int = 5,
        n_repeats: int = 5,
        seed: int = 42,
    ):
        if candidate_units is None:
            self.candidate_units = [2, 4, 8, 12, 16, 24, 32]
        else:
            self.candidate_units = candidate_units
        self.k_folds = k_folds
        self.n_repeats = n_repeats
        self.seed = seed

    def evaluate_units(
        self,
        X: np.ndarray,
        y: np.ndarray,
        problem_type: str,
        n_classes: int = 1,
        hidden_activation: str = "sigmoid",
    ) -> Dict:
        input_dim = X.shape[1]
        if problem_type == "classification":
            output_dim = n_classes if n_classes > 2 else 1
            out_act = "softmax" if n_classes > 2 else "sigmoid"
            loss_fn = "cross_entropy"
        else:
            output_dim = 1
            out_act = "identity"
            loss_fn = "mse"

        results = {}

        for units in self.candidate_units:
            val_losses = []
            val_metrics = []

            for rep in range(self.n_repeats):
                splits = k_fold_cross_validation(
                    X, y, k=self.k_folds, stratify=(problem_type == "classification"), seed=self.seed + rep * 100
                )

                for X_train, y_train, X_val, y_val in splits:
                    net = NeuralNetwork(
                        input_dim=input_dim,
                        hidden_dim=units,
                        output_dim=output_dim,
                        hidden_activation=hidden_activation,
                        output_activation=out_act,
                        loss_fn=loss_fn,
                        seed=self.seed + len(val_losses),
                    )

                    opt = SCGOptimizer(max_iters=200, tol=1e-5)
                    opt_res = net.fit(X_train, y_train, opt, X_val, y_val)

                    v_loss = net.compute_loss(X_val, y_val)
                    val_losses.append(v_loss)

                    preds = net.predict(X_val)
                    if problem_type == "classification":
                        score = float(np.mean(preds == y_val))
                    else:
                        score = float(np.mean((preds.ravel() - y_val.ravel()) ** 2))
                    val_metrics.append(score)

            mean_loss = float(np.mean(val_losses))
            std_loss = float(np.std(val_losses, ddof=1)) if len(val_losses) > 1 else 0.0
            mean_metric = float(np.mean(val_metrics))
            std_metric = float(np.std(val_metrics, ddof=1)) if len(val_metrics) > 1 else 0.0

            results[units] = {
                "mean_val_loss": mean_loss,
                "std_val_loss": std_loss,
                "mean_metric": mean_metric,
                "std_metric": std_metric,
                "all_losses": [float(x) for x in val_losses],
            }

        best_units = None
        min_loss = float("inf")
        for u in self.candidate_units:
            if results[u]["mean_val_loss"] < min_loss:
                min_loss = results[u]["mean_val_loss"]
                best_units = u

        threshold = min_loss + results[best_units]["std_val_loss"] / np.sqrt(len(results[best_units]["all_losses"]))
        parsimonious_units = best_units
        for u in sorted(self.candidate_units):
            if results[u]["mean_val_loss"] <= threshold:
                parsimonious_units = u
                break

        return {
            "best_units": best_units,
            "parsimonious_units": parsimonious_units,
            "results_per_unit": results,
        }

    def run_all(
        self,
        datasets_dict: Dict[str, Tuple[np.ndarray, np.ndarray, Dict]],
        output_dir: str = "results/raw",
    ) -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        summary = {}

        for key, (X, y, info) in datasets_dict.items():
            prob_type = info["type"]
            n_cls = info.get("n_classes", 1)
            res = self.evaluate_units(X, y, problem_type=prob_type, n_classes=n_cls)
            summary[key] = {
                "dataset_name": info["name"],
                "type": prob_type,
                "best_units": res["best_units"],
                "parsimonious_units": res["parsimonious_units"],
                "details": res["results_per_unit"],
            }

        out_path = os.path.join(output_dir, "hidden_units_summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary
