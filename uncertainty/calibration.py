from typing import Tuple
import numpy as np


def predictive_stats(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean and variance over samples (MC dropout/ensemble in real implementation)."""
    return samples.mean(axis=0), samples.var(axis=0)


def nll_gaussian(y_true: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    var = np.clip(var, 1e-6, None)
    return float(0.5 * np.mean(np.log(2 * np.pi * var) + (y_true - mu) ** 2 / var))


def ece(prob_true: np.ndarray, prob_pred: np.ndarray, bins: int = 10) -> float:
    """Expected Calibration Error (classification analog). Placeholder for regression calibration."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    e = 0.0
    for i in range(bins):
        mask = (prob_pred >= edges[i]) & (prob_pred < edges[i + 1])
        if mask.any():
            e += np.abs(prob_true[mask].mean() - prob_pred[mask].mean()) * (mask.sum() / prob_pred.size)
    return float(e)
