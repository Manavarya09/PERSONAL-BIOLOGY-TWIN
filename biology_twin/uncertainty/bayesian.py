import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from typing import Tuple


class BayesianFoundationModel(nn.Module):
    """Bayesian version of foundation model using Pyro."""

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        # Variational parameters
        self.q_mu = nn.Linear(input_dim, embed_dim)
        self.q_log_var = nn.Linear(input_dim, embed_dim)

    def model(self, x: torch.Tensor, y: torch.Tensor = None):
        """Generative model."""
        # Prior
        w_prior = dist.Normal(torch.zeros(self.embed_dim, self.input_dim), torch.ones(self.embed_dim, self.input_dim))
        b_prior = dist.Normal(torch.zeros(self.embed_dim), torch.ones(self.embed_dim))
        
        w = pyro.sample("w", w_prior)
        b = pyro.sample("b", b_prior)
        
        # Likelihood
        mu = torch.matmul(x, w.t()) + b
        sigma = pyro.sample("sigma", dist.HalfNormal(1.0))
        
        with pyro.plate("data", x.size(0)):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)

    def guide(self, x: torch.Tensor, y: torch.Tensor = None):
        """Variational guide."""
        # Approximate posterior
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        w = pyro.sample("w", dist.Normal(mu, torch.exp(0.5 * log_var)))
        b = pyro.sample("b", dist.Normal(torch.zeros(self.embed_dim), torch.ones(self.embed_dim)))
        sigma = pyro.sample("sigma", dist.HalfNormal(1.0))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean and variance."""
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        return mu, torch.exp(log_var)


class BayesianUncertainty:
    """Bayesian uncertainty quantification."""

    def __init__(self, model: BayesianFoundationModel):
        self.model = model
        self.svi = pyro.infer.SVI(
            model=self.model.model,
            guide=self.model.guide,
            optim=pyro.optim.Adam({"lr": 0.01}),
            loss=pyro.infer.Trace_ELBO()
        )

    def train(self, x: torch.Tensor, y: torch.Tensor, num_epochs: int = 100):
        """Train variational posterior."""
        for epoch in range(num_epochs):
            loss = self.svi.step(x, y)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, x: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predictive distribution."""
        predictive = pyro.infer.Predictive(self.model.model, guide=self.model.guide, num_samples=num_samples)
        samples = predictive(x)
        mu = samples["obs"].mean(0)
        var = samples["obs"].var(0)
        return mu, var