from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from flwr.client import NumPyClient
from torch.utils.data import DataLoader


class PyTorchClient(NumPyClient, ABC):
    """Abstract PyTorch client for federated learning."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cpu",
        is_byzantine: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.is_byzantine = is_byzantine

    @abstractmethod
    def train(self, epochs: int, learning_rate: float) -> Tuple[int, float]:
        """Train the model and return number of samples and loss."""
        pass

    @abstractmethod
    def test(self) -> Tuple[int, float, float]:
        """Test the model and return samples, loss, and accuracy."""
        pass

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return model parameters as list of NumPy arrays."""
        if self.is_byzantine:
            # Byzantine behavior: return random parameters
            params = []
            for param in self.model.parameters():
                random_param = np.random.randn(*param.shape).astype(
                    np.float32
                )
                params.append(random_param)
            return params
        else:
            return [
                param.cpu().detach().numpy()
                for param in self.model.parameters()
            ]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from list of NumPy arrays."""
        params_dict = zip(self.model.parameters(), parameters)
        for param, new_param in params_dict:
            param.data = torch.from_numpy(new_param).to(self.device)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Fit the model using provided parameters."""
        self.set_parameters(parameters)
        num_samples, loss = self.train(
            epochs=config.get("local_epochs", 1),
            learning_rate=config.get("learning_rate", 0.01),
        )
        return self.get_parameters(config={}), num_samples, {"loss": loss}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        """Evaluate the model using provided parameters."""
        self.set_parameters(parameters)
        num_samples, loss, accuracy = self.test()
        return loss, num_samples, {"accuracy": accuracy}


class SimpleClient(PyTorchClient):
    """Concrete implementation of PyTorchClient with SGD optimizer."""

    def train(self, epochs: int, learning_rate: float) -> Tuple[int, float]:
        """Train the model."""
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=learning_rate, momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        num_samples = 0

        for _ in range(epochs):
            for batch in self.train_loader:
                images, labels = batch[0].to(self.device), batch[1].to(
                    self.device
                )

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(labels)
                num_samples += len(labels)

        return num_samples, total_loss / num_samples

    def test(self) -> Tuple[int, float, float]:
        """Test the model."""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        num_samples = 0

        with torch.no_grad():
            for batch in self.test_loader:
                images, labels = batch[0].to(self.device), batch[1].to(
                    self.device
                )

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * len(labels)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                num_samples += len(labels)

        accuracy = correct / num_samples
        return num_samples, total_loss / num_samples, accuracy


class AdamClient(PyTorchClient):
    """Client implementation using Adam optimizer."""

    def train(self, epochs: int, learning_rate: float) -> Tuple[int, float]:
        """Train the model with Adam optimizer."""
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        num_samples = 0

        for _ in range(epochs):
            for batch in self.train_loader:
                images, labels = batch[0].to(self.device), batch[1].to(
                    self.device
                )

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(labels)
                num_samples += len(labels)

        return num_samples, total_loss / num_samples

    def test(self) -> Tuple[int, float, float]:
        """Test the model."""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        num_samples = 0

        with torch.no_grad():
            for batch in self.test_loader:
                images, labels = batch[0].to(self.device), batch[1].to(
                    self.device
                )

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * len(labels)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                num_samples += len(labels)

        accuracy = correct / num_samples
        return num_samples, total_loss / num_samples, accuracy


# Client factory
CLIENT_REGISTRY: Dict[str, Type[PyTorchClient]] = {
    "simple": SimpleClient,
    "sgd": SimpleClient,
    "adam": AdamClient,
}


def get_client_class(client_type: str = "simple") -> Type[PyTorchClient]:
    """
    Factory function to get client class.

    Args:
        client_type: Type of client ('simple', 'sgd', 'adam')

    Returns:
        Client class

    Example:
        ClientClass = get_client_class("adam")
        client = ClientClass(model, train_loader, test_loader)
    """
    if client_type not in CLIENT_REGISTRY:
        raise ValueError(
            f"Unknown client type: {client_type}. "
            f"Available types: {list(CLIENT_REGISTRY.keys())}"
        )

    return CLIENT_REGISTRY[client_type]


def register_client(name: str, client_class: Type[PyTorchClient]):
    """Register a custom client type."""
    CLIENT_REGISTRY[name] = client_class


def list_client_types() -> list:
    """Return list of available client types."""
    return list(CLIENT_REGISTRY.keys())
