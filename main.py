import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
from typing import Dict, Tuple
import numpy as np

from src.preprocessing import load_classification_dataset, load_regression_dataset
from src.experiments import (
    HiddenUnitsExperiment,
    HyperparameterTuningExperiment,
    ComparisonExperiment,
)


def load_all_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict]]:
    datasets = {}
    for i in range(1, 4):
        key = f"classification_{i}"
        X, y, info = load_classification_dataset(f"dataset{i}")
        datasets[key] = (X, y, info)

    for i in range(1, 4):
        key = f"regression_{i}"
        X, y, info = load_regression_dataset(f"dataset{i}")
        datasets[key] = (X, y, info)

    return datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "hidden_units", "hyperparameters", "comparison"],
    )
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--max_iters", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    datasets = load_all_datasets()

    hidden_units_path = os.path.join(args.output_dir, "raw", "hidden_units_summary.json")
    hyperparams_path = os.path.join(args.output_dir, "raw", "tuned_hyperparameters.json")

    hidden_units_map = {}
    hyperparams_map = {}

    if args.experiment in ("all", "hidden_units"):
        print("Starting hidden units optimization experiment...")
        hue = HiddenUnitsExperiment(
            candidate_units=[2, 4, 8, 12, 16, 24, 32],
            k_folds=5,
            n_repeats=3,
            seed=args.seed,
        )
        summary = hue.run_all(datasets, output_dir=os.path.join(args.output_dir, "raw"))
        for k, v in summary.items():
            hidden_units_map[k] = v["parsimonious_units"]
            print(f"[{k}] Parsimonious optimal hidden units: {hidden_units_map[k]}")
    elif os.path.exists(hidden_units_path):
        with open(hidden_units_path, "r") as f:
            data = json.load(f)
            for k, v in data.items():
                hidden_units_map[k] = v.get("parsimonious_units", v.get("best_units", 8))
    else:
        hidden_units_map = {k: 8 for k in datasets}

    if args.experiment in ("all", "hyperparameters"):
        print("Starting hyperparameter tuning experiment...")
        hte = HyperparameterTuningExperiment(k_folds=3, seed=args.seed)
        tuned = hte.tune_all(
            datasets=datasets,
            hidden_units_map=hidden_units_map,
            output_dir=os.path.join(args.output_dir, "raw"),
            max_iters=min(args.max_iters, 200),
        )
        for ds_k, opt_dict in tuned.items():
            hyperparams_map[ds_k] = {}
            for opt_k, p_dict in opt_dict.items():
                hyperparams_map[ds_k][opt_k] = p_dict["best_params"]
                print(f"[{ds_k}][{opt_k}] Tuned parameters: {hyperparams_map[ds_k][opt_k]}")
    elif os.path.exists(hyperparams_path):
        with open(hyperparams_path, "r") as f:
            data = json.load(f)
            for ds_k, opt_dict in data.items():
                hyperparams_map[ds_k] = {}
                for opt_k, p_dict in opt_dict.items():
                    hyperparams_map[ds_k][opt_k] = p_dict.get("best_params", {})
    else:
        default_params = {
            "SGD": {"lr": 0.05, "momentum": 0.9, "batch_size": None},
            "SCG": {"sigma": 1e-4, "lambda_init": 1e-6},
            "LeapFrog": {"dt_init": 0.1, "delta": 1.0, "version": "LFOP1"},
        }
        hyperparams_map = {k: default_params for k in datasets}

    if args.experiment in ("all", "comparison"):
        print(f"Starting comparison experiment with {args.runs} independent runs...")
        ce = ComparisonExperiment(
            n_runs=args.runs,
            max_iters=args.max_iters,
            base_seed=args.seed,
        )
        benchmark_results = ce.run_benchmark(
            datasets=datasets,
            hidden_units_map=hidden_units_map,
            hyperparams_map=hyperparams_map,
            output_dir=args.output_dir,
        )
        print("Comparison benchmark completed successfully.")
        print(f"Results saved to {args.output_dir}/raw, {args.output_dir}/tables, and {args.output_dir}/figures.")


if __name__ == "__main__":
    main()
