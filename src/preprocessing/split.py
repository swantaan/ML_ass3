from typing import Generator, List, Optional, Tuple
import numpy as np


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    n = len(X)
    indices = np.arange(n)

    if stratify and y.ndim == 1:
        classes, y_indices = np.unique(y, return_inverse=True)
        train_idx_list, val_idx_list, test_idx_list = [], [], []

        for cls_idx in range(len(classes)):
            cls_samples = indices[y_indices == cls_idx]
            rng.shuffle(cls_samples)
            n_cls = len(cls_samples)
            n_test = max(1, int(round(test_size * n_cls)))
            n_val = max(1, int(round(val_size * n_cls)))

            test_idx_list.append(cls_samples[:n_test])
            val_idx_list.append(cls_samples[n_test : n_test + n_val])
            train_idx_list.append(cls_samples[n_test + n_val :])

        test_idx = np.concatenate(test_idx_list)
        val_idx = np.concatenate(val_idx_list)
        train_idx = np.concatenate(train_idx_list)

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)
    else:
        rng.shuffle(indices)
        n_test = int(round(test_size * n))
        n_val = int(round(val_size * n))

        test_idx = indices[:n_test]
        val_idx = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val :]

    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        X[test_idx],
        y[test_idx],
    )


def k_fold_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    stratify: bool = False,
    seed: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(seed)
    n = len(X)
    indices = np.arange(n)

    folds = [[] for _ in range(k)]

    if stratify and y.ndim == 1:
        classes, y_indices = np.unique(y, return_inverse=True)
        for cls_idx in range(len(classes)):
            cls_samples = indices[y_indices == cls_idx]
            rng.shuffle(cls_samples)
            for i, idx in enumerate(cls_samples):
                folds[i % k].append(idx)
    else:
        rng.shuffle(indices)
        for i, idx in enumerate(indices):
            folds[i % k].append(idx)

    result = []
    for i in range(k):
        val_idx = np.array(folds[i])
        train_folds = [folds[j] for j in range(k) if j != i]
        train_idx = np.concatenate(train_folds)
        result.append((X[train_idx], y[train_idx], X[val_idx], y[val_idx]))

    return result
