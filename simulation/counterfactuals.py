from typing import Dict
import numpy as np


def apply_intervention(latent: np.ndarray, deltas: Dict[str, float]) -> np.ndarray:
    s = latent.copy()
    mapping = {
        "autonomic": 0,
        "recovery": 1,
        "circadian": 2,
        "stress": 3,
        "cognitive": 4,
    }
    for k, v in deltas.items():
        idx = mapping.get(k)
        if idx is not None:
            s[idx] += v
    return s
