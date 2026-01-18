import flwr as fl
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from biology_twin.foundation_model.base import PhysiologicalTransformer


class FederatedClient(fl.client.NumPyClient):
    """Flower client for federated learning."""

    def __init__(self, model: PhysiologicalTransformer, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(5):  # Local epochs
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(batch["x"])
                loss = self.criterion(output, batch["y"])
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                output = self.model(batch["x"])
                loss += self.criterion(output, batch["y"]).item()
        loss /= len(self.val_loader)
        return loss, len(self.val_loader), {"loss": loss}


def start_federated_client(model: PhysiologicalTransformer, train_loader, val_loader, server_address: str = "127.0.0.1:8080"):
    """Start federated learning client."""
    client = FederatedClient(model, train_loader, val_loader)
    fl.client.start_numpy_client(server_address=server_address, client=client)


class FederatedServer:
    """Federated learning server."""

    def __init__(self, model: PhysiologicalTransformer, num_clients: int = 3):
        self.model = model
        self.num_clients = num_clients

    def start_server(self, server_address: str = "127.0.0.1:8080"):
        """Start Flower server."""
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=self.num_clients,
            min_evaluate_clients=self.num_clients,
            min_available_clients=self.num_clients,
        )
        
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=3),
            strategy=strategy,
        )