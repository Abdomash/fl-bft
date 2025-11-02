# experiment/runner.py
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedProx, FedOpt
from flwr.simulation import run_simulation

from clients.pytorch_client import create_client_app
from clients.vision_client import VisionClient
from datasets.vision import CIFAR10Dataset, MNISTDataset
from strategies.fedavg_with_logs import FedAvgWithLogging

from .config import ExperimentConfig


class ExperimentRunner:
    """Runner for FL experiments."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._set_seed()
        self._setup_results_dir()

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.config.seed)

    def _setup_results_dir(self):
        """Create results directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(self.config.results_dir, f"{
                                    self.config.name}_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)

        # Save config
        with open(os.path.join(self.exp_dir, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def _load_dataset(self):
        """Load and partition dataset."""
        if self.config.dataset == "cifar10":
            dataset = CIFAR10Dataset(**self.config.dataset_config)
        elif self.config.dataset == "mnist":
            dataset = MNISTDataset(**self.config.dataset_config)
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")

        train_data, test_data = dataset.load()
        partitions = dataset.partition(
            train_data, self.config.num_clients, self.config.partition_config
        )

        return partitions, test_data

    def _create_strategy(self, metrics_log: List[Dict]):
        """Create FL strategy."""

        def log_callback(phase: str, round_num: int, metrics: Dict):
            metrics_log.append({
                "phase": phase,
                "round": round_num,
                **metrics,
            })

        base_config = {
            "fraction_fit": self.config.strategy_config.get("fraction_fit", 1.0),
            "fraction_evaluate": self.config.strategy_config.get(
                "fraction_evaluate", 1.0
            ),
            "min_fit_clients": self.config.strategy_config.get(
                "min_fit_clients", self.config.num_clients
            ),
            "min_evaluate_clients": self.config.strategy_config.get(
                "min_evaluate_clients", self.config.num_clients
            ),
            "min_available_clients": self.config.num_clients,
        }

        if self.config.strategy_name == "FedAvg":
            return FedAvgWithLogging(**base_config, log_callback=log_callback)
        elif self.config.strategy_name == "FedProx":
            proximal_mu = self.config.strategy_config.get("proximal_mu", 0.01)
            return FedProx(**base_config, proximal_mu=proximal_mu)
        elif self.config.strategy_name == "FedOpt":
            return FedOpt(**base_config, **self.config.strategy_config)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy_name}")

    def run(self) -> Dict[str, Any]:
        """Run the experiment."""
        print(f"\nRunning experiment: {self.config.name}")
        print(f"Results will be saved to: {self.exp_dir}")

        # Load and partition dataset
        print("Loading dataset...")
        train_partitions, test_data = self._load_dataset()

        # Determine model specifications
        num_classes = 10
        in_channels = 3 if self.config.dataset == "cifar10" else 1

        # Setup client configuration
        node_config = {
            "model_name": self.config.client_models[0],
            "num_classes": num_classes,
            "in_channels": in_channels,
            "train_datasets": train_partitions,
            "test_dataset": test_data,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
        }

        # Create client app
        client_app = create_client_app(VisionClient)

        # Create strategy with metrics logging
        metrics_log = []
        strategy = self._create_strategy(metrics_log)

        server_config = ServerConfig(num_rounds=self.config.num_rounds)

        # Run simulation
        print(f"Starting simulation with {self.config.num_clients} clients...")
        history = run_simulation(
            server_app=ServerApp(
                config=server_config,
                strategy=strategy,
            ),
            client_app=client_app,
            num_supernodes=self.config.num_clients,
        )

        # Save results
        results = {
            "config": self.config.to_dict(),
            "metrics": metrics_log,
            "history": {
                "losses_distributed": [
                    (round_num, loss)
                    for round_num, loss in enumerate(
                        history.losses_distributed, start=1
                    )
                ],
                "losses_centralized": [
                    (round_num, loss)
                    for round_num, loss in enumerate(
                        history.losses_centralized, start=1
                    )
                ],
                "metrics_distributed": history.metrics_distributed,
                "metrics_centralized": history.metrics_centralized,
            },
        }

        with open(os.path.join(self.exp_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"Experiment completed. Results saved to {self.exp_dir}")
        return results
