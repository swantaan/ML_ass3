from typing import Any, Dict, List, Tuple
import numpy as np
from scipy import stats


def summarize_runs(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def wilcoxon_test(sample1: List[float], sample2: List[float]) -> Dict[str, float]:
    s1 = np.asarray(sample1, dtype=float)
    s2 = np.asarray(sample2, dtype=float)
    diff = s1 - s2
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
    try:
        stat, p = stats.wilcoxon(s1, s2)
        return {"statistic": float(stat), "p_value": float(p), "significant": bool(p < 0.05)}
    except Exception:
        stat, p = stats.mannwhitneyu(s1, s2)
        return {"statistic": float(stat), "p_value": float(p), "significant": bool(p < 0.05)}


def mann_whitney_test(sample1: List[float], sample2: List[float]) -> Dict[str, float]:
    s1 = np.asarray(sample1, dtype=float)
    s2 = np.asarray(sample2, dtype=float)
    stat, p = stats.mannwhitneyu(s1, s2)
    return {"statistic": float(stat), "p_value": float(p), "significant": bool(p < 0.05)}


def friedman_test(*samples: List[float]) -> Dict[str, float]:
    data = [np.asarray(s, dtype=float) for s in samples]
    stat, p = stats.friedmanchisquare(*data)
    return {"statistic": float(stat), "p_value": float(p), "significant": bool(p < 0.05)}
