import yaml
import torch
from biology_twin.foundation_model.base import FoundationModel
from biology_twin.data.data_loader import DataLoader


def train_foundation_model(config_path: str = "config/default.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fm = FoundationModel(input_dim=cfg.get("input_dim", 10), embedding_dim=cfg["embedding_dim"], device=device)
    loader = DataLoader()

    # Load training data
    train_data = loader.load_physionet()
    preprocessed = loader.preprocess_irregular_ts(train_data)
    cleaned = loader.detect_artifacts(preprocessed)

    # Convert to dict of arrays for pretrain
    signals = {k: v.reshape(1, -1, 1) for k, v in cleaned.items()}  # Batch=1, seq, features

    print("Starting pretraining...")
    fm.pretrain(signals, epochs=cfg.get("pretrain_epochs", 5))

    # Save model
    torch.save(fm.model.state_dict(), "models/foundation_model.pth")
    print("Model saved.")


if __name__ == "__main__":
    train_foundation_model()