from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np


class PhysiologicalTransformer(nn.Module):
    """Transformer encoder for multi-signal time-series."""

    def __init__(self, input_dim: int, embed_dim: int = 64, num_heads: int = 8, num_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = nn.Embedding(1000, embed_dim)  # Max seq len
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.input_proj(x) + self.pos_encoder(pos)
        x = self.transformer(x)
        return self.output_proj(x.mean(dim=1))  # Pool to embedding


class FoundationModel:
    """
    Production-ready physiological foundation model with transformer backbone.
    Supports masked modeling for SSL pretraining.
    """

    def __init__(self, input_dim: int = 10, embedding_dim: int = 64, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = PhysiologicalTransformer(input_dim, embedding_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.mask_ratio = 0.15

    def pretrain(self, signals: Dict[str, np.ndarray], epochs: int = 10) -> None:
        """Self-supervised pretraining with masked signal modeling."""
        # Convert to tensor: assume signals are dict of (batch, seq, features)
        # For demo, stack signals
        data = torch.tensor(np.stack(list(signals.values()), axis=-1), dtype=torch.float32).to(self.device)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            masked_data, mask = self._mask_data(data)
            pred = self.model(masked_data)
            loss = self._reconstruction_loss(pred, data.mean(dim=1), mask)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    def encode(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Encode signals to latent embeddings."""
        data = torch.tensor(np.stack(list(signals.values()), axis=-1), dtype=torch.float32).to(self.device)
        data = data.unsqueeze(0)  # Add batch dim
        with torch.no_grad():
            emb = self.model(data)
        return emb.cpu().numpy()

    def _mask_data(self, x: torch.Tensor) -> tuple:
        mask = torch.rand(x.shape[:-1], device=self.device) < self.mask_ratio
        masked = x.clone()
        masked[mask] = 0  # Simple mask
        return masked, mask

    def _reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return nn.MSELoss()(pred[mask.any(dim=1)], target[mask.any(dim=1)])