from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ExperimentConfig:
    """Configuration for a federated learning experiment."""

    # Experiment metadata
    name: str
    description: str = ""

    # Dataset configuration
    dataset_name: str = "uoft-cs/cifar10"
    partition_type: str = "iid"
    dirichlet_alpha: float = 0.5

    # Client configuration
    num_clients: int = 10
    num_byzantine_clients: int = 0

    # Model configuration
    model_name: str = "SimpleCNN"
    model_params: Dict[str, Any] = field(default_factory=dict)

    # Strategy configuration
    strategy_name: str = "FedAvg"
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    # Training configuration
    num_rounds: int = 10
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    clients_per_round: int = 10

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cpu"

    # Output
    save_dir: str = "results"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "dataset_name": self.dataset_name,
            "partition_type": self.partition_type,
            "dirichlet_alpha": self.dirichlet_alpha,
            "num_clients": self.num_clients,
            "num_byzantine_clients": self.num_byzantine_clients,
            "model_name": self.model_name,
            "model_params": self.model_params,
            "strategy_name": self.strategy_name,
            "strategy_params": self.strategy_params,
            "num_rounds": self.num_rounds,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "clients_per_round": self.clients_per_round,
            "seed": self.seed,
            "device": self.device,
        }
