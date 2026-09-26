import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.neural_network import NeuralNetwork
from src.optimizers import SGDOptimizer, SCGOptimizer, LeapFrogOptimizer
from src.preprocessing import train_val_test_split
from src.evaluation.metrics import classification_metrics, regression_metrics
from src.evaluation.statistics import summarize_runs, friedman_test, wilcoxon_test


class ComparisonExperiment:
    def __init__(
        self,
        n_runs: int = 30,
        max_iters: int = 500,
        base_seed: int = 42,
    ):
        self.n_runs = n_runs
        self.max_iters = max_iters
        self.base_seed = base_seed

    def run_benchmark(
        self,
        datasets: Dict[str, Tuple[np.ndarray, np.ndarray, Dict]],
        hidden_units_map: Dict[str, int],
        hyperparams_map: Dict[str, Dict[str, Any]],
        output_dir: str = "results",
    ) -> Dict:
        raw_dir = os.path.join(output_dir, "raw")
        tables_dir = os.path.join(output_dir, "tables")
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        benchmark_results = {}

        for ds_key, (X, y, info) in datasets.items():
            prob_type = info["type"]
            n_cls = info.get("n_classes", 1)
            h_units = hidden_units_map.get(ds_key, 8)
            opt_params = hyperparams_map.get(ds_key, {})

            input_dim = X.shape[1]
            output_dim = n_cls if (prob_type == "classification" and n_cls > 2) else 1
            out_act = "softmax" if (prob_type == "classification" and n_cls > 2) else ("sigmoid" if prob_type == "classification" else "identity")
            loss_fn = "cross_entropy" if prob_type == "classification" else "mse"

            ds_record = {
                "dataset_name": info["name"],
                "type": prob_type,
                "hidden_units": h_units,
                "algorithms": {},
            }

            algos = ["SGD", "SCG", "LeapFrog"]

            for alg in algos:
                ds_record["algorithms"][alg] = {
                    "test_metric": [],
                    "func_evals": [],
                    "grad_evals": [],
                    "iterations": [],
                    "wall_time": [],
                    "converged": [],
                    "loss_curves": [],
                }

            for run_idx in range(self.n_runs):
                seed = self.base_seed + run_idx * 17
                X_tr, y_tr, X_val, y_val, X_te, y_te = train_val_test_split(
                    X, y, val_size=0.15, test_size=0.15, stratify=(prob_type == "classification"), seed=seed
                )

                for alg in algos:
                    net = NeuralNetwork(
                        input_dim=input_dim,
                        hidden_dim=h_units,
                        output_dim=output_dim,
                        hidden_activation="sigmoid",
                        output_activation=out_act,
                        loss_fn=loss_fn,
                        seed=seed,
                    )

                    params = opt_params.get(alg, {}).copy()
                    if alg == "SGD":
                        opt = SGDOptimizer(max_iters=self.max_iters, **params)
                    elif alg == "SCG":
                        opt = SCGOptimizer(max_iters=self.max_iters, **params)
                    elif alg == "LeapFrog":
                        opt = LeapFrogOptimizer(max_iters=self.max_iters, **params)

                    res = net.fit(X_tr, y_tr, opt, X_val, y_val)

                    preds = net.predict(X_te)
                    if prob_type == "classification":
                        m = classification_metrics(y_te, preds)
                        score = m["accuracy"]
                    else:
                        m = regression_metrics(y_te, preds)
                        score = m["mse"]

                    rec = ds_record["algorithms"][alg]
                    rec["test_metric"].append(score)
                    rec["func_evals"].append(res.func_evals)
                    rec["grad_evals"].append(res.grad_evals)
                    rec["iterations"].append(res.iterations)
                    rec["wall_time"].append(res.wall_time)
                    rec["converged"].append(res.converged)

                    if run_idx < 5:
                        rec["loss_curves"].append(res.loss_history)

            summary = {}
            for alg in algos:
                rec = ds_record["algorithms"][alg]
                summary[alg] = {
                    "test_metric": summarize_runs(rec["test_metric"]),
                    "func_evals": summarize_runs(rec["func_evals"]),
                    "grad_evals": summarize_runs(rec["grad_evals"]),
                    "iterations": summarize_runs(rec["iterations"]),
                    "wall_time": summarize_runs(rec["wall_time"]),
                    "convergence_rate": float(np.mean(rec["converged"])),
                }

            stat_tests = {}
            fried = friedman_test(
                ds_record["algorithms"]["SGD"]["test_metric"],
                ds_record["algorithms"]["SCG"]["test_metric"],
                ds_record["algorithms"]["LeapFrog"]["test_metric"],
            )
            stat_tests["friedman_test_metric"] = fried

            for pair in [("SGD", "SCG"), ("SGD", "LeapFrog"), ("SCG", "LeapFrog")]:
                a1, a2 = pair
                w_test = wilcoxon_test(
                    ds_record["algorithms"][a1]["test_metric"],
                    ds_record["algorithms"][a2]["test_metric"],
                )
                stat_tests[f"wilcoxon_{a1}_vs_{a2}"] = w_test

            ds_record["summary"] = summary
            ds_record["statistical_tests"] = stat_tests
            benchmark_results[ds_key] = ds_record

            self._plot_dataset_results(ds_key, ds_record, figures_dir)

        self._export_tables(benchmark_results, tables_dir)

        out_json = os.path.join(raw_dir, "benchmark_comparison_results.json")
        cleaned_results = {}
        for k, v in benchmark_results.items():
            cleaned_results[k] = {
                "dataset_name": v["dataset_name"],
                "type": v["type"],
                "hidden_units": v["hidden_units"],
                "summary": v["summary"],
                "statistical_tests": v["statistical_tests"],
            }
        with open(out_json, "w") as f:
            json.dump(cleaned_results, f, indent=2)

        return benchmark_results

    def _plot_dataset_results(self, ds_key: str, ds_record: Dict, figures_dir: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        ax1 = axes[0]
        alg_colors = {"SGD": "#1f77b4", "SCG": "#2ca02c", "LeapFrog": "#d62728"}
        for alg, col in alg_colors.items():
            curves = ds_record["algorithms"][alg]["loss_curves"]
            if curves:
                min_len = min(len(c) for c in curves)
                arr = np.array([c[:min_len] for c in curves])
                mean_curve = np.mean(arr, axis=0)
                ax1.plot(mean_curve, label=alg, color=col, linewidth=1.8)

        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"Convergence Curves: {ds_record['dataset_name']}")
        ax1.set_yscale("log")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        ax2 = axes[1]
        metric_name = "Accuracy" if ds_record["type"] == "classification" else "MSE"
        data_to_plot = [ds_record["algorithms"][alg]["test_metric"] for alg in ["SGD", "SCG", "LeapFrog"]]
        bp = ax2.boxplot(data_to_plot, tick_labels=["SGD", "SCG", "LeapFrog"], patch_artist=True)
        colors = ["#aec7e8", "#98df8a", "#ff9896"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax2.set_ylabel(metric_name)
        ax2.set_title(f"Test {metric_name} ({self.n_runs} Runs)")
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"{ds_key}_comparison.pdf"))
        plt.close()

    def _export_tables(self, benchmark_results: Dict, tables_dir: str) -> None:
        lines = []
        lines.append("\\begin{table*}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Experimental Comparison of SGD, SCG, and LeapFrog across Benchmark Problems}")
        lines.append("\\begin{tabular}{llcccc}")
        lines.append("\\hline")
        lines.append("Problem & Algorithm & Test Metric (Mean $\\pm$ SD) & Function Evals & Gradient Evals & CPU Time (s) \\\\")
        lines.append("\\hline")

        for ds_key, data in benchmark_results.items():
            ds_name = data["dataset_name"]
            is_first = True
            for alg in ["SGD", "SCG", "LeapFrog"]:
                s = data["summary"][alg]
                m_mean = s["test_metric"]["mean"]
                m_std = s["test_metric"]["std"]
                fe_mean = s["func_evals"]["mean"]
                ge_mean = s["grad_evals"]["mean"]
                time_mean = s["wall_time"]["mean"]

                name_col = ds_name if is_first else ""
                lines.append(f"{name_col} & {alg} & {m_mean:.4f} $\\pm$ {m_std:.4f} & {fe_mean:.1f} & {ge_mean:.1f} & {time_mean:.3f} \\\\")
                is_first = False
            lines.append("\\hline")

        lines.append("\\end{tabular}")
        lines.append("\\end{table*}")

        tex_path = os.path.join(tables_dir, "comparison_table.tex")
        with open(tex_path, "w") as f:
            f.write("\n".join(lines))
