# run_experiments.py
import torch
import flwr as fl
from .models import get_model
from .datasets import create_dataloaders
from .client import FlowerClient
from .strategies import CustomStrategy, RobustFedAvg
from .server import get_evaluate_fn
from .utils import (
    ExperimentLogger,
    find_existing_experiment,
    get_experiment_summary
)
from .plotting import plot_experiments_auto
import numpy as np
from typing import List
import argparse
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    dataset: str = "cifar10"
    num_clients: int = 10
    malicious_ratio: float = 0.0
    strategy: str = "fedavg"
    num_rounds: int = 50
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    iid: bool = True
    filter_malicious: bool = False
    trim_ratio: float = 0.1
    device: str = "cpu"

    def to_dict(self):
        """Convert to dictionary for hashing/comparison"""
        return asdict(self)


def run_experiment(
    config: ExperimentConfig,
    force_rerun: bool = False,
    verbose: bool = True
):
    """
    Run a single federated learning experiment

    Args:
        config: Experiment configuration
        force_rerun: If True, run even if cached result exists
        verbose: Print detailed progress

    Returns:
        Tuple of (log_file_path, metrics)
    """

    config_dict = config.to_dict()

    # Check for existing experiment
    if not force_rerun:
        existing_log = find_existing_experiment(config_dict)
        if existing_log:
            print(f"\n{'='*60}")
            print(f"Found existing experiment: {existing_log}")
            summary = get_experiment_summary(existing_log)
            print(f"Final Accuracy: {summary.get('final_accuracy', 0):.4f}")
            print(f"Reusing cached results...")
            print(f"{'='*60}\n")
            return existing_log, summary

    # Setup device for server-side evaluation
    device = torch.device(
        config.device if torch.cuda.is_available()
        and config.device == "cuda" else "cpu"
    )
    if verbose:
        print(f"Server evaluation device: {device}")

    # For simulation, force CPU for clients
    client_device = "cpu"
    if verbose:
        print(f"Client training device: {client_device}")

    # Create experiment name
    malicious_clients = int(config.num_clients * config.malicious_ratio)
    experiment_name = (
        f"{config.dataset}_{config.strategy}_"
        f"c{config.num_clients}_"
        f"m{malicious_clients}"
    )
    if config.filter_malicious:
        experiment_name += "_filtered"

    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting experiment: {experiment_name}")
        print(f"Configuration:")
        for key, value in config_dict.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")

    # Create dataloaders
    if verbose:
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
        if verbose:
            print(f"Malicious clients: {sorted(malicious_clients_set)}")

    # Initialize model
    model = get_model(config.dataset)

    # Client function
    def client_fn(cid: str) -> FlowerClient:
        cid_int = int(cid)
        is_malicious = cid_int in malicious_clients_set

        client_model = get_model(config.dataset)

        return FlowerClient(
            cid=cid_int,
            model=client_model,
            trainloader=client_loaders[cid_int],
            device=client_device,
            is_malicious=is_malicious,
        )

    # Setup strategy
    strategy_kwargs = {
        'fraction_fit': 1.0,
        'fraction_evaluate': 0.0,
        'min_fit_clients': config.num_clients,
        'min_evaluate_clients': 0,
        'min_available_clients': config.num_clients,
        'evaluate_fn': get_evaluate_fn(model, test_loader, device),
        'on_fit_config_fn': lambda _: {
            'local_epochs': config.local_epochs,
            'learning_rate': config.learning_rate,
        },
    }

    if config.strategy == "fedavg":
        strategy = fl.server.strategy.FedAvg(**strategy_kwargs)
    elif config.strategy == "custom":
        strategy = CustomStrategy(
            **strategy_kwargs,
            filter_malicious=config.filter_malicious
        )
    elif config.strategy == "robust":
        strategy = RobustFedAvg(
            **strategy_kwargs,
            trim_ratio=config.trim_ratio
        )
    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")

    # Setup logger
    logger = ExperimentLogger(experiment_name)
    logger.set_config(config_dict)

    # History tracker
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

    def wrapped_eval_fn(server_round, parameters, config_dict):
        result = original_eval_fn(server_round, parameters, config_dict)
        if result:
            loss, metrics = result
            accuracy = metrics.get("accuracy", 0)
            history_logger.log_round(server_round, loss, accuracy)
            logger.log_round(
                server_round,
                {"loss": loss, "accuracy": accuracy}
            )
            if verbose:
                print(f"Round {server_round}: Loss={
                      loss:.4f}, Acc={accuracy:.4f}")
        return result

    strategy.evaluate_fn = wrapped_eval_fn

    # Run simulation
    if verbose:
        print("\nStarting training...\n")

    ray_init_args = {
        "include_dashboard": False,
        "num_cpus": None,
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

    if verbose:
        print(f"\nExperiment completed: {experiment_name}\n")

    return log_file, history_logger.rounds


def run_experiment_suite(
    configs: List[ExperimentConfig],
    force_rerun: bool = False,
    verbose: bool = True,
    plot_results: bool = True
):
    """
    Run multiple experiments and create visualizations

    Args:
        configs: List of experiment configurations
        force_rerun: If True, rerun all experiments even if cached
        verbose: Print detailed progress
        plot_results: Generate plots after all experiments

    Returns:
        List of (log_file, metrics) tuples
    """

    print(f"\n{'#'*60}")
    print(f"# Running Experiment Suite: {len(configs)} experiments")
    print(f"# Force rerun: {force_rerun}")
    print(f"{'#'*60}\n")

    log_files = []
    all_metrics = []
    config_dicts = []

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Running experiment...")

        try:
            log_file, metrics = run_experiment(
                config,
                force_rerun=force_rerun,
                verbose=verbose
            )
            log_files.append(log_file)
            all_metrics.append(metrics)
            config_dicts.append(config.to_dict())
        except Exception as e:
            print(f"ERROR: Experiment failed: {e}")
            continue

    print(f"\n{'#'*60}")
    print(f"# All experiments completed!")
    print(f"# Successful: {len(log_files)}/{len(configs)}")
    print(f"{'#'*60}\n")

    # Generate plots
    if plot_results and log_files:
        print("\nGenerating visualizations...")
        plot_experiments_auto(log_files, config_dicts)

    return list(zip(log_files, all_metrics))


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
        choices=[
            "single", "malicious_study", "client_scaling",
            "strategy_comparison", "custom"
        ],
        help="Experiment mode"
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force rerun even if cached results exist"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for server evaluation"
    )

    args = parser.parse_args()

    if args.mode == "single":
        # Single experiment
        config = ExperimentConfig(
            dataset=args.dataset,
            num_clients=10,
            malicious_ratio=0.2,
            strategy="fedavg",
            num_rounds=30,
            device=args.device,
        )
        run_experiment(config, force_rerun=args.force_rerun)

    elif args.mode == "malicious_study":
        # Study impact of malicious clients
        configs = [
            ExperimentConfig(
                dataset=args.dataset,
                num_clients=50,
                malicious_ratio=ratio,
                strategy="fedavg",
                num_rounds=30,
                device=args.device,
            )
            for ratio in [0.0, 0.1, 0.2, 0.3, 0.4]
        ]
        run_experiment_suite(
            configs,
            force_rerun=args.force_rerun,
            plot_results=not args.no_plot
        )

    elif args.mode == "client_scaling":
        # Study scaling with number of clients
        configs = [
            ExperimentConfig(
                dataset=args.dataset,
                num_clients=n,
                malicious_ratio=0.1,
                strategy="fedavg",
                num_rounds=30,
                device=args.device,
            )
            for n in [10, 25, 50, 75, 100]
        ]
        run_experiment_suite(
            configs,
            force_rerun=args.force_rerun,
            plot_results=not args.no_plot
        )

    elif args.mode == "strategy_comparison":
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
        run_experiment_suite(
            configs,
            force_rerun=args.force_rerun,
            plot_results=not args.no_plot
        )

    elif args.mode == "custom":
        print("Custom mode: Edit the script to define your experiments")


if __name__ == "__main__":
    main()
