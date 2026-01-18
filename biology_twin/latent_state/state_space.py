from typing import Dict, Any, Callable
import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np


class ODEFunc(nn.Module):
    """Neural ODE function for latent dynamics."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 50),
            nn.Tanh(),
            nn.Linear(50, dim)
        )

    def forward(self, t, y):
        return self.net(y)


class LatentTwin:
    """
    Production-ready latent digital twin with Neural ODE dynamics.
    """

    def __init__(self, state_dim: int = 8, device: str = "cpu"):
        self.state_dim = state_dim
        self.device = torch.device(device)
        self.ode_func = ODEFunc(state_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.ode_func.parameters(), lr=1e-3)
        self.state = torch.zeros(state_dim, device=self.device)

    def initialize(self, embedding: np.ndarray) -> None:
        self.state = torch.tanh(torch.tensor(embedding.squeeze()[:self.state_dim], dtype=torch.float32, device=self.device))

    def update(self, embedding: np.ndarray) -> np.ndarray:
        """Update state via ODE integration."""
        z = torch.tensor(embedding.squeeze()[:self.state_dim], dtype=torch.float32, device=self.device)
        # Simple update: integrate from current state towards z
        t_span = torch.linspace(0, 1, 10, device=self.device)
        traj = odeint(self.ode_func, self.state, t_span)
        self.state = traj[-1] + 0.1 * (z - traj[-1])  # Blend with observation
        return self.state.detach().cpu().numpy()

    def predict(self, horizon: int = 7) -> np.ndarray:
        """Predict trajectory via ODE."""
        t_span = torch.linspace(0, horizon, horizon * 10, device=self.device)
        traj = odeint(self.ode_func, self.state, t_span)
        # Downsample to daily
        indices = torch.linspace(0, len(t_span)-1, horizon, dtype=torch.long)
        return traj[indices].detach().cpu().numpy()

    def simulate_counterfactual(self, intervention: Dict[str, float], horizon: int = 7) -> np.ndarray:
        """Simulate interventions by perturbing ODE."""
        perturbed_func = PerturbedODE(self.ode_func, intervention, self._axis_index)
        t_span = torch.linspace(0, horizon, horizon * 10, device=self.device)
        traj = odeint(perturbed_func, self.state, t_span)
        indices = torch.linspace(0, len(t_span)-1, horizon, dtype=torch.long)
        return traj[indices].detach().cpu().numpy()

    def _axis_index(self, name: str) -> Optional[int]:
        mapping = {
            "autonomic": 0,
            "recovery": 1,
            "circadian": 2,
            "stress": 3,
            "cognitive": 4,
        }
        return mapping.get(name)


class PerturbedODE(nn.Module):
    def __init__(self, base_func, intervention, axis_map):
        super().__init__()
        self.base_func = base_func
        self.intervention = intervention
        self.axis_map = axis_map

    def forward(self, t, y):
        dy = self.base_func(t, y)
        for k, v in self.intervention.items():
            idx = self.axis_map(k)
            if idx is not None:
                dy[idx] += v
        return dy