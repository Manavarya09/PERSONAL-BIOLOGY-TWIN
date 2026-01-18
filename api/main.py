from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import numpy as np
from biology_twin.foundation_model.base import FoundationModel
from biology_twin.latent_state.state_space import LatentTwin
from biology_twin.personalization.adapter import UserAdapter
from biology_twin.monitoring.prometheus import Monitoring
from biology_twin.monitoring.elk import ELKLogger
import yaml
import time

app = FastAPI(title="Personal Biology Twin API", version="1.0.0")

# Load config
with open("config/default.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Initialize components
try:
    fm = FoundationModel(embedding_dim=cfg["embedding_dim"], input_dim=3, device="cpu")
    twin = LatentTwin(state_dim=cfg["state_dim"], device="cpu")
    adapter = UserAdapter(state_dim=cfg["state_dim"])
except Exception as e:
    print(f"Model initialization failed: {e}")
    fm = None
    twin = None
    adapter = None
try:
    monitor = Monitoring()
except Exception as e:
    print(f"Monitoring disabled: {e}")
    monitor = None
try:
    logger = ELKLogger()
except Exception as e:
    print(f"ELK logging disabled: {e}")
    logger = None

class SignalData(BaseModel):
    signals: Dict[str, List[float]]

class Intervention(BaseModel):
    intervention: Dict[str, float]
    horizon: int = 7

@app.post("/encode")
def encode_signals(data: SignalData):
    """Encode physiological signals to latent embedding."""
    if fm is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    start_time = time.time()
    try:
        signals = {k: np.array(v) for k, v in data.signals.items()}
        embedding = fm.encode(signals)
        duration = time.time() - start_time
        if monitor:
            monitor.process_request("POST", "/encode")
        if logger:
            logger.log_request("POST", "/encode", duration=duration)
        return {"embedding": embedding.tolist()}
    except Exception as e:
        if logger:
            logger.log_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_twin")
def update_twin(data: SignalData):
    """Update latent twin state."""
    if fm is None or twin is None:
        raise HTTPException(status_code=500, detail="Models not initialized")
    start_time = time.time()
    try:
        signals = {k: np.array(v) for k, v in data.signals.items()}
        embedding = fm.encode(signals)
        twin.initialize(embedding)
        updated = twin.update(embedding)
        personalized = adapter.update(updated)
        duration = time.time() - start_time
        if monitor:
            monitor.process_request("POST", "/update_twin")
        if logger:
            logger.log_request("POST", "/update_twin", duration=duration)
        return {"latent_state": updated.tolist(), "personalized": personalized.tolist()}
    except Exception as e:
        if logger:
            logger.log_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_trajectory")
def predict_trajectory(horizon: int = 7):
    """Predict future latent trajectory."""
    if twin is None:
        raise HTTPException(status_code=500, detail="Twin model not initialized")
    start_time = time.time()
    try:
        traj = twin.predict(horizon)
        duration = time.time() - start_time
        if monitor:
            monitor.process_request("POST", "/predict_trajectory")
        if logger:
            logger.log_request("POST", "/predict_trajectory", duration=duration)
        return {"trajectory": traj.tolist()}
    except Exception as e:
        if logger:
            logger.log_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate_counterfactual")
def simulate_counterfactual(data: Intervention):
    """Simulate counterfactual intervention."""
    start_time = time.time()
    try:
        traj = twin.simulate_counterfactual(data.intervention, data.horizon)
        duration = time.time() - start_time
        if monitor:
            monitor.process_request("POST", "/simulate_counterfactual")
        if logger:
            logger.log_request("POST", "/simulate_counterfactual", duration=duration)
        return {"counterfactual_trajectory": traj.tolist()}
    except Exception as e:
        if logger:
            logger.log_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}