from typing import Dict, Any
import numpy as np


class SCM:
    """
    Structural Causal Model skeleton for lifestyle interventions.

    Variables (illustrative): sleep, training_load, caffeine, stress_level, recovery.
    """

    def __init__(self, seed: int = 11):
        self.rng = np.random.default_rng(seed)

    def forward(self, x: Dict[str, float]) -> Dict[str, float]:
        sleep = x.get("sleep", 7.0)
        load = x.get("training_load", 0.5)
        caf = x.get("caffeine", 1.0)
        stress = 0.5 * (1.0 - sleep / 8.0) + 0.4 * load + 0.2 * caf
        recovery = sleep / 8.0 - 0.3 * load - 0.2 * stress
        return {"stress_level": float(stress), "recovery": float(recovery)}

    def do(self, x: Dict[str, float], intervention: Dict[str, float]) -> Dict[str, float]:
        x_cf = {**x, **intervention}
        return self.forward(x_cf)
