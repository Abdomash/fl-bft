from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays, Scalar


class PyTorchClientApp(NumPyClient, ABC):
    """Abstract base class for PyTorch FL clients."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device

    @abstractmethod
    def train(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the model and return updated parameters."""
        pass

    @abstractmethod
    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model and return loss and metrics."""
        pass

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return current model parameters as NumPy arrays."""
        return [
            val.cpu().numpy() for val in self.model.state_dict().values()
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)


def create_client_app(client_class):
    """Factory to create a ClientApp from a PyTorchClientApp subclass."""

    def client_fn(context: Context):
        return client_class(context)

    return ClientApp(client_fn=client_fn)
