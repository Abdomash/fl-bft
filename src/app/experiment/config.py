# experiment/config.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for a single FL experiment."""

    name: str
    dataset: str
    dataset_config: Dict[str, Any]
    partition_config: Dict[str, Any]

    num_clients: int
    client_models: List[str]

    strategy_name: str
    strategy_config: Dict[str, Any]

    num_rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float

    results_dir: str = "./results"
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "dataset": self.dataset,
            "dataset_config": self.dataset_config,
            "partition_config": self.partition_config,
            "num_clients": self.num_clients,
            "client_models": self.client_models,
            "strategy_name": self.strategy_name,
            "strategy_config": self.strategy_config,
            "num_rounds": self.num_rounds,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
        }
