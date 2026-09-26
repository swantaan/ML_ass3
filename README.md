# Comparison of Feedforward Neural Network Training Algorithms

This repository implements and evaluates three feedforward neural network training algorithms on classification and function approximation benchmark problems:

1. **Stochastic Gradient Descent (SGD)**: Gradient descent with Polyak and Nesterov momentum, learning rate scheduling, and mini-batching.
2. **Scaled Conjugate Gradient (SCG)**: Second-order optimization method utilizing finite-difference directional Hessian approximations and Levenberg-Marquardt scale parameter adaptation (Møller, 1993).
3. **Leap-Frog Optimizer (LFOP1 / LFOP1(b))**: Dynamic physical particle simulation in conservative force fields with kinetic energy monitoring and adaptive time-step selection (Snyman, 1982, 1983).

---

## Benchmark Datasets

The repository includes three classification and three function approximation problems of varying complexity:

### Classification
- **Dataset 1: Wine Recognition** (`data/classification/dataset1/`): Multi-class tabular dataset (178 samples, 13 features, 3 classes).
- **Dataset 2: Two Moons** (`data/classification/dataset2/`): Synthetic non-linear geometric manifold (500 samples, 2 features, 2 classes).
- **Dataset 3: Breast Cancer Wisconsin** (`data/classification/dataset3/`): High-dimensional clinical diagnostic dataset (569 samples, 30 features, 2 classes).

### Function Approximation (Regression)
- **Dataset 1: Damped Sinc** (`data/regression/dataset1/`): Univariate non-linear oscillatory continuous function approximation (500 samples, 1 feature).
- **Dataset 2: Friedman #1** (`data/regression/dataset2/`): Multivariate non-linear function with trigonometric and quadratic interactions (600 samples, 10 features).
- **Dataset 3: Diabetes Progression** (`data/regression/dataset3/`): Real-world multi-dimensional clinical regression dataset (442 samples, 10 features).

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/swantaan/ML_ass3.git
cd ML_ass3
pip install -r requirements.txt
```

---

## Repository Structure

```text
├── data/
│   ├── classification/
│   │   ├── dataset1/ (Wine Recognition)
│   │   ├── dataset2/ (Two Moons)
│   │   └── dataset3/ (Breast Cancer)
│   └── regression/
│       ├── dataset1/ (Damped Sinc)
│       ├── dataset2/ (Friedman #1)
│       └── dataset3/ (Diabetes)
├── reading/
│   ├── SCG.pdf      (Møller, 1993)
│   ├── LFROGa.pdf   (Snyman, 1982)
│   └── LFROGb.pdf   (Snyman, 1983)
├── report/
│   ├── ass3.tex
│   └── writingRules.pdf
├── results/
│   ├── figures/     (Generated convergence curves and boxplots)
│   ├── tables/      (Generated LaTeX tables)
│   └── raw/         (Raw experimental JSON summaries)
├── src/
│   ├── neural_network/
│   │   ├── activation.py
│   │   ├── loss.py
│   │   ├── layers.py
│   │   └── network.py
│   ├── optimizers/
│   │   ├── base.py
│   │   ├── sgd.py
│   │   ├── scg.py
│   │   └── leapfrog.py
│   ├── preprocessing/
│   │   ├── scaler.py
│   │   ├── encoding.py
│   │   ├── split.py
│   │   └── loader.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── statistics.py
│   └── experiments/
│       ├── hidden_units.py
│       ├── hyperparameters.py
│       └── comparison.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Running Experiments

Execute the experiments through the unified CLI:

### 1. Full Pipeline (Optimal Architecture Search + Tuning + Comparison)
```bash
python3 main.py --experiment all --runs 30 --max_iters 500
```

### 2. Individual Stages

- **Hidden Units Optimization**:
  ```bash
  python3 main.py --experiment hidden_units
  ```

- **Hyperparameter Tuning**:
  ```bash
  python3 main.py --experiment hyperparameters --max_iters 200
  ```

- **30-Run Statistical Comparison**:
  ```bash
  python3 main.py --experiment comparison --runs 30 --max_iters 500
  ```

---

## References

- M. F. Møller, "A scaled conjugate gradient algorithm for fast supervised learning," *Neural Networks*, vol. 6, no. 4, pp. 525–533, 1993.
- J. A. Snyman, "A new and dynamic method for unconstrained minimization," *Applied Mathematical Modelling*, vol. 6, no. 6, pp. 449–462, 1982.
- J. A. Snyman, "An improved version of the original leap-frog dynamic method for unconstrained minimization: LFOP1(b)," *Applied Mathematical Modelling*, vol. 7, no. 3, pp. 216–218, 1983.
