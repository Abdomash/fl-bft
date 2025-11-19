import torch
import flwr as fl
from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        model: torch.nn.Module,
        trainloader: torch.utils.data.DataLoader,
        device: str = "cpu",  # Changed to string
        is_malicious: bool = False,
    ):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        # Handle device per-client to avoid Ray issues
        if device == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
            print(f"Client {cid}: CUDA not available, using CPU")
        else:
            self.device = torch.device(device)
        self.is_malicious = is_malicious

    def get_parameters(self, config):
        """Return model parameters as a list of NumPy arrays"""
        return [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays"""
        params_dict = zip(
            self.model.state_dict().keys(), parameters
        )
        state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in params_dict
        })
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters, config
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model and return updated parameters"""
        self.set_parameters(parameters)

        epochs = config.get("local_epochs", 1)
        lr = config.get("learning_rate", 0.01)

        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9
        )
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs):
            for batch in self.trainloader:
                images, labels = batch
                images, labels = (
                    images.to(self.device),
                    labels.to(self.device)
                )

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # If malicious, flip the signs of the gradient
        if self.is_malicious:
            parameters = [
                -val.cpu().numpy()
                for _, val in self.model.state_dict().items()
            ]
            return (
                parameters,
                len(self.trainloader.dataset),
                {"loss": avg_loss, "malicious": True}
            )

        return (
            self.get_parameters(config),
            len(self.trainloader.dataset),
            {"loss": avg_loss, "malicious": False}
        )

    def evaluate(
        self, parameters, config
    ) -> Tuple[float, int, Dict]:
        """Evaluate model on local data"""
        self.set_parameters(parameters)

        self.model.to(self.device)
        self.model.eval()

        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.trainloader:
                images, labels = batch
                images, labels = (
                    images.to(self.device),
                    labels.to(self.device)
                )

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.trainloader)

        return avg_loss, total, {"accuracy": accuracy}
