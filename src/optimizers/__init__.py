"""
Optimizer implementations for Neural Network Training (Option 2):
1. SGDOptimizer (Stochastic Gradient Descent with momentum and mini-batching)
2. SCGOptimizer (Scaled Conjugate Gradient, Moller 1993)
3. LeapFrogOptimizer (Leap-Frog dynamic optimization LFOP1 / LFOP1(b), Snyman 1982/1983)
"""

from src.optimizers.base import BaseOptimizer, OptimizerResult
from src.optimizers.sgd import SGDOptimizer
from src.optimizers.scg import SCGOptimizer
from src.optimizers.leapfrog import LeapFrogOptimizer

__all__ = [
    "BaseOptimizer",
    "OptimizerResult",
    "SGDOptimizer",
    "SCGOptimizer",
    "LeapFrogOptimizer",
]
