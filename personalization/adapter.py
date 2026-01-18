from typing import Dict
import numpy as np


class UserAdapter:
    """
    Per-user latent adaptation with simple Bayesian-like updating and regularization.
    """

    def __init__(self, state_dim: int, reg: float = 0.1):
        self.m = np.zeros(state_dim)
        self.P = np.eye(state_dim)
        self.reg = reg

    def update(self, latent: np.ndarray) -> np.ndarray:
        y = latent
        K = self.P @ np.linalg.pinv(self.P + self.reg * np.eye(self.P.shape[0]))
        self.m = self.m + K @ (y - self.m)
        self.P = (np.eye(self.P.shape[0]) - K) @ self.P
        return self.m.copy()
