import torch
import flwr as fl
from .models import get_model
from .datasets import create_dataloaders
from .client import FlowerClient
from .strategies import CustomStrategy, RobustFedAvg
from .server import get_evaluate_fn
from .utils import ExperimentLogger
from .plotting import (
    plot_single_experiment,
    plot_comparison,
    plot_malicious_impact
)
import numpy as np
from typing import List
import argparse


class ExperimentConfig:
    def __init__(
        self,
        dataset: str = "cifar10",
        num_clients: int = 10,
        malicious_ratio: float = 0.0,
        strategy: str = "fedavg",
        num_rounds: int = 50,
        local_epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        iid: bool = True,
        filter_malicious: bool = False,
        trim_ratio: float = 0.1,
        device: str = "cpu",  # Added device config
    ):
        self.dataset = dataset
        self.num_clients = num_clients
        self.malicious_ratio = malicious_ratio
        self.strategy = strategy
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iid = iid
        self.filter_malicious = filter_malicious
        self.trim_ratio = trim_ratio
        self.device = device


def run_experiment(config: ExperimentConfig):
    """Run a single federated learning experiment"""

    # Setup device for server-side evaluation
    # In simulation, clients will use CPU by default to avoid Ray issues
    device = torch.device(
        config.device if torch.cuda.is_available()
        and config.device == "cuda" else "cpu"
    )
    print(f"Server evaluation device: {device}")

    # For simulation, force CPU for clients to avoid Ray/CUDA conflicts
    client_device = "cpu"
    print(f"Client training device: {client_device}")

    # Create experiment name
    malicious_clients = int(config.num_clients * config.malicious_ratio)
    experiment_name = (
        f"{config.dataset}_{config.strategy}_"
        f"clients{config.num_clients}_"
        f"malicious{malicious_clients}"
    )
    if config.filter_malicious:
        experiment_name += "_filtered"

    print(f"\n{'='*60}")
    print(f"Starting experiment: {experiment_name}")
    print(f"{'='*60}\n")

    # Create dataloaders
    print("Preparing data...")
    client_loaders, test_loader = create_dataloaders(
        config.dataset,
        config.num_clients,
        config.batch_size,
        config.iid
    )

    # Determine malicious clients
    malicious_clients_set = set()
    if config.malicious_ratio > 0:
        num_malicious = int(
            config.num_clients * config.malicious_ratio
        )
        malicious_clients_set = set(
            np.random.choice(
                config.num_clients, num_malicious, replace=False
            )
        )
        print(f"Malicious clients: {sorted(malicious_clients_set)}")

    # Initialize model
    model = get_model(config.dataset)

    # Client function
    def client_fn(cid: str) -> FlowerClient:
        cid_int = int(cid)
        is_malicious = cid_int in malicious_clients_set

        # Create a new model instance for each client
        client_model = get_model(config.dataset)

        return FlowerClient(
            cid=cid_int,
            model=client_model,
            trainloader=client_loaders[cid_int],
            device=client_device,  # Use client_device
            is_malicious=is_malicious,
        )

    # Setup strategy
    if config.strategy == "fedavg":
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=config.num_clients,
            min_evaluate_clients=0,
            min_available_clients=config.num_clients,
            evaluate_fn=get_evaluate_fn(model, test_loader, device),
            on_fit_config_fn=lambda _: {
                "local_epochs": config.local_epochs,
                "learning_rate": config.learning_rate,
            },
        )
    elif config.strategy == "custom":
        strategy = CustomStrategy(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=config.num_clients,
            min_evaluate_clients=0,
            min_available_clients=config.num_clients,
            evaluate_fn=get_evaluate_fn(model, test_loader, device),
            on_fit_config_fn=lambda _: {
                "local_epochs": config.local_epochs,
                "learning_rate": config.learning_rate,
            },
            filter_malicious=config.filter_malicious,
        )
    elif config.strategy == "robust":
        strategy = RobustFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=config.num_clients,
            min_evaluate_clients=0,
            min_available_clients=config.num_clients,
            evaluate_fn=get_evaluate_fn(model, test_loader, device),
            on_fit_config_fn=lambda _: {
                "local_epochs": config.local_epochs,
                "learning_rate": config.learning_rate,
            },
            trim_ratio=config.trim_ratio,
        )
    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")

    # Setup logger
    logger = ExperimentLogger(experiment_name)

    # Custom history callback to log metrics
    class HistoryLogger:
        def __init__(self):
            self.rounds = []

        def log_round(self, server_round, loss, accuracy):
            self.rounds.append({
                "round": server_round,
                "loss": loss,
                "accuracy": accuracy,
            })

    history_logger = HistoryLogger()

    # Wrap evaluate_fn to capture metrics
    original_eval_fn = strategy.evaluate_fn

    def wrapped_eval_fn(server_round, parameters, config):
        result = original_eval_fn(server_round, parameters, config)
        if result:
            loss, metrics = result
            accuracy = metrics.get("accuracy", 0)
            history_logger.log_round(server_round, loss, accuracy)
            logger.log_round(
                server_round,
                {"loss": loss, "accuracy": accuracy}
            )
        return result

    strategy.evaluate_fn = wrapped_eval_fn

    # Run simulation
    print("\nStarting training...\n")

    # Configure Ray resources to avoid GPU allocation
    ray_init_args = {
        "include_dashboard": False,
        "num_cpus": None,  # Use all available CPUs
    }

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )

    # Save logs
    log_file = logger.save()

    # Plot results
    print("\nGenerating plots...")
    plot_single_experiment(log_file)

    print(f"\nExperiment completed: {experiment_name}\n")

    return log_file, history_logger.rounds


def run_multiple_experiments(configs: List[ExperimentConfig]):
    """Run multiple experiments and create comparison plots"""

    log_files = []
    labels = []

    for config in configs:
        log_file, _ = run_experiment(config)
        log_files.append(log_file)

        malicious_clients = int(
            config.num_clients * config.malicious_ratio
        )
        label = (
            f"{config.strategy} - "
            f"{config.num_clients} clients - "
            f"{malicious_clients} malicious"
        )
        labels.append(label)

    # Create comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(log_files, labels)

    # If experiments vary by malicious ratio, create impact plot
    malicious_ratios = set(c.malicious_ratio for c in configs)
    if len(malicious_ratios) > 1:
        plot_malicious_impact(log_files, labels)

    print("\nAll experiments completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Run Federated Learning Experiments"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "fashion_mnist"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "comparison", "malicious_study"],
        help="Experiment mode"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for server evaluation (clients use CPU in simulation)"
    )

    args = parser.parse_args()

    if args.mode == "single":
        # Run a single experiment
        config = ExperimentConfig(
            dataset=args.dataset,
            num_clients=10,
            malicious_ratio=0.2,
            strategy="fedavg",
            num_rounds=30,
            local_epochs=1,
            device=args.device,
        )
        run_experiment(config)

    elif args.mode == "comparison":
        # Compare different strategies
        configs = [
            ExperimentConfig(
                dataset=args.dataset,
                num_clients=50,
                malicious_ratio=0.2,
                strategy="fedavg",
                num_rounds=30,
                device=args.device,
            ),
            ExperimentConfig(
                dataset=args.dataset,
                num_clients=50,
                malicious_ratio=0.2,
                strategy="custom",
                filter_malicious=True,
                num_rounds=30,
                device=args.device,
            ),
            ExperimentConfig(
                dataset=args.dataset,
                num_clients=50,
                malicious_ratio=0.2,
                strategy="robust",
                trim_ratio=0.2,
                num_rounds=30,
                device=args.device,
            ),
        ]
        run_multiple_experiments(configs)

    elif args.mode == "malicious_study":
        # Study impact of different malicious ratios
        configs = [
            ExperimentConfig(
                dataset=args.dataset,
                num_clients=50,
                malicious_ratio=0.0,
                strategy="fedavg",
                num_rounds=30,
                device=args.device,
            ),
            ExperimentConfig(
                dataset=args.dataset,
                num_clients=50,
                malicious_ratio=0.1,
                strategy="fedavg",
                num_rounds=30,
                device=args.device,
            ),
            ExperimentConfig(
                dataset=args.dataset,
                num_clients=50,
                malicious_ratio=0.2,
                strategy="fedavg",
                num_rounds=30,
                device=args.device,
            ),
            ExperimentConfig(
                dataset=args.dataset,
                num_clients=50,
                malicious_ratio=0.3,
                strategy="fedavg",
                num_rounds=30,
                device=args.device,
            ),
        ]
        run_multiple_experiments(configs)


if __name__ == "__main__":
    main()
