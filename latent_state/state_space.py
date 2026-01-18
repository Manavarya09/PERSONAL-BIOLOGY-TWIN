from typing import Dict, Any
import numpy as np


class LatentTwin:
    """
    Continuous latent state representing autonomic balance, recovery, circadian alignment,
    stress resilience, and cognitive readiness.

    Skeleton dynamics: linear state-space with Kalman-like updates.
    """

    def __init__(self, state_dim: int = 8, seed: int = 123):
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(state_dim)
        # Transition and process noise
        self.A = np.eye(state_dim) * 0.98
        self.Q = np.eye(state_dim) * 0.02
        # Observation mapping and noise
        self.H = np.eye(state_dim)
        self.R = np.eye(state_dim) * 0.05

    def initialize(self, embedding: np.ndarray) -> None:
        self.state = np.tanh(embedding.squeeze()[: self.state_dim])

    def update(self, embedding: np.ndarray) -> np.ndarray:
        """Predict + update with simple linear-Gaussian assumptions."""
        # Predict
        pred_state = self.A @ self.state
        pred_cov = self.Q
        # Observation
        z = np.tanh(embedding.squeeze()[: self.state_dim])
        # Kalman gain (diag simplification)
        S = self.H @ pred_cov @ self.H.T + self.R
        K = pred_cov @ self.H.T @ np.linalg.pinv(S)
        # Update
        innovation = z - (self.H @ pred_state)
        self.state = pred_state + K @ innovation
        return self.state.copy()

    def predict(self, horizon: int = 7) -> np.ndarray:
        """Roll forward expected latent trajectory."""
        traj = []
        s = self.state.copy()
        for _ in range(horizon):
            s = self.A @ s
            traj.append(s.copy())
        return np.stack(traj)

    def simulate_counterfactual(self, intervention: Dict[str, float], horizon: int = 7) -> np.ndarray:
        """Apply simple additive shifts to select latent axes to simulate interventions."""
        traj = []
        s = self.state.copy()
        for _ in range(horizon):
            s = self.A @ s
            for k, v in intervention.items():
                idx = self._axis_index(k)
                if idx is not None:
                    s[idx] += v
            traj.append(s.copy())
        return np.stack(traj)

    def _axis_index(self, name: str):
        mapping = {
            "autonomic": 0,
            "recovery": 1,
            "circadian": 2,
            "stress": 3,
            "cognitive": 4,
        }
        return mapping.get(name)
