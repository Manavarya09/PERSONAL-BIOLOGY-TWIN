import yaml
import numpy as np
import pandas as pd

from biology_twin.foundation_model.base import FoundationModel
from biology_twin.latent_state.state_space import LatentTwin
from biology_twin.personalization.adapter import UserAdapter
from biology_twin.evaluation.metrics import mae


def synthetic_signals(n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    hr = 60 + 10 * np.sin(np.linspace(0, 10, n)) + rng.normal(0, 2, size=n)
    hrv = 50 + 5 * np.cos(np.linspace(0, 5, n)) + rng.normal(0, 3, size=n)
    sleep_quality = 0.7 + 0.1 * np.sin(np.linspace(0, 2, n)) + rng.normal(0, 0.05, size=n)
    return {"hr": hr, "hrv": hrv, "sleep": sleep_quality}


def main():
    with open("config/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    signals = synthetic_signals(1024, seed=cfg["seed"]) 
    fm = FoundationModel(input_dim=3, embedding_dim=cfg["embedding_dim"], device="cpu") 
    emb = fm.encode(signals)

    twin = LatentTwin(state_dim=cfg["state_dim"], device="cpu") 
    twin.initialize(emb)
    updated = twin.update(emb)
    traj = twin.predict(horizon=cfg["horizon_days"]) 

    adapter = UserAdapter(state_dim=cfg["state_dim"]) 
    personalized = adapter.update(updated)

    print("Initial latent:", updated.round(3))
    print("Personalized latent:", personalized.round(3))
    print("Predicted trajectory shape:", traj.shape)

    # Simple eval: compare next-day prediction to a noisy target
    target = updated + np.random.default_rng(cfg["seed"]).normal(0, 0.1, size=updated.shape)
    pred_next = traj[0]
    print("MAE next-day:", mae(target, pred_next))


if __name__ == "__main__":
    main()
