from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path


class DataLoader:
    """Handles loading and preprocessing of physiological datasets."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def load_physionet(self, dataset: str = "apnea-ecg") -> Dict[str, np.ndarray]:
        """Load PhysioNet dataset using wfdb."""
        import wfdb
        dataset_dir = self.data_dir / dataset
        dataset_dir.mkdir(exist_ok=True)
        
        # Download if not exists
        if not list(dataset_dir.glob("*.hea")):
            print(f"Downloading {dataset}...")
            wfdb.dl_database(dataset, str(dataset_dir))
        
        # Load sample record
        records = wfdb.get_record_list(dataset, str(dataset_dir))
        if records:
            record = wfdb.rdrecord(records[0], pn_dir=dataset)
            signals = {sig: record.p_signal[:, i] for i, sig in enumerate(record.sig_name)}
            return self.preprocess_irregular_ts(signals)
        else:
            return self._synthetic_physio_data()

    def load_sleep_edf(self) -> Dict[str, np.ndarray]:
        """Load Sleep-EDF data."""
        # Stub
        return {"eeg": np.random.randn(1000, 5), "labels": np.random.randint(0, 5, 1000)}

    def preprocess_irregular_ts(self, signals: Dict[str, np.ndarray], target_freq: float = 1.0) -> Dict[str, np.ndarray]:
        """Resample irregular time-series to regular grid."""
        # Simple interpolation
        processed = {}
        for k, v in signals.items():
            if len(v.shape) == 1:
                # Assume 1D time series
                processed[k] = np.interp(np.arange(0, len(v), target_freq), np.arange(len(v)), v)
            else:
                processed[k] = v  # Higher dim
        return processed

    def detect_artifacts(self, signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Detect motion artifacts, gaps."""
        cleaned = {}
        for k, v in signals.items():
            # Simple threshold-based artifact removal
            mask = np.abs(v - np.mean(v)) < 3 * np.std(v)
            cleaned[k] = v[mask]
        return cleaned

    def _synthetic_physio_data(self) -> Dict[str, np.ndarray]:
        """Generate synthetic physiological data for demo."""
        np.random.seed(42)
        n = 1000
        return {
            "hr": 60 + 10 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 2, n),
            "hrv": 50 + 5 * np.cos(np.linspace(0, 2*np.pi, n)) + np.random.normal(0, 3, n),
            "resp": 12 + 2 * np.sin(np.linspace(0, 6*np.pi, n)) + np.random.normal(0, 1, n),
        }