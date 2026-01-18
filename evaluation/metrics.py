from typing import Tuple
import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> float:
    """Approximate CRPS for Gaussian predictive distribution (per element)."""
    from scipy.stats import norm
    z = (y_true - mu) / np.clip(sig, 1e-6, None)
    return float(np.mean(sig * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def drift_kl(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    return float(np.sum(p * np.log((p + 1e-12) / (q + 1e-12))))
