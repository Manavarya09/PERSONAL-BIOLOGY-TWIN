from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import numpy as np
from biology_twin.foundation_model.base import FoundationModel
from biology_twin.latent_state.state_space import LatentTwin
from biology_twin.personalization.adapter import UserAdapter
import yaml

app = FastAPI(title="Personal Biology Twin API", version="1.0.0")

# Load config
with open("config/default.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Initialize models (in production, load from saved checkpoints)
fm = FoundationModel(embedding_dim=cfg["embedding_dim"])
twin = LatentTwin(state_dim=cfg["state_dim"])
adapter = UserAdapter(state_dim=cfg["state_dim"])

class SignalData(BaseModel):
    signals: Dict[str, List[float]]

class Intervention(BaseModel):
    intervention: Dict[str, float]
    horizon: int = 7

@app.post("/encode")
def encode_signals(data: SignalData):
    """Encode physiological signals to latent embedding."""
    signals = {k: np.array(v) for k, v in data.signals.items()}
    embedding = fm.encode(signals)
    return {"embedding": embedding.tolist()}

@app.post("/update_twin")
def update_twin(data: SignalData):
    """Update latent twin state."""
    signals = {k: np.array(v) for k, v in data.signals.items()}
    embedding = fm.encode(signals)
    twin.initialize(embedding)
    updated = twin.update(embedding)
    personalized = adapter.update(updated)
    return {"latent_state": updated.tolist(), "personalized": personalized.tolist()}

@app.post("/predict_trajectory")
def predict_trajectory(horizon: int = 7):
    """Predict future latent trajectory."""
    traj = twin.predict(horizon)
    return {"trajectory": traj.tolist()}

@app.post("/simulate_counterfactual")
def simulate_counterfactual(data: Intervention):
    """Simulate counterfactual intervention."""
    traj = twin.simulate_counterfactual(data.intervention, data.horizon)
    return {"counterfactual_trajectory": traj.tolist()}

@app.get("/health")
def health_check():
    return {"status": "healthy"}