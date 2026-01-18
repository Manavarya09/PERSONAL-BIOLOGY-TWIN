import pytest
import numpy as np
from biology_twin.foundation_model.base import FoundationModel
from biology_twin.latent_state.state_space import LatentTwin
from biology_twin.evaluation.metrics import mae


def test_foundation_model_encode():
    fm = FoundationModel(input_dim=2, embedding_dim=64)  # Match signal dims
    signals = {"hr": np.random.randn(100), "hrv": np.random.randn(100)}
    emb = fm.encode(signals)
    assert emb.shape == (1, 64)


def test_latent_twin_update():
    twin = LatentTwin(state_dim=8)
    embedding = np.random.randn(1, 64)
    twin.initialize(embedding)
    updated = twin.update(embedding)
    assert updated.shape == (8,)


def test_mae():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 2.1, 2.9])
    assert mae(y_true, y_pred) > 0