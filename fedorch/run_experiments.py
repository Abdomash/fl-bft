# run_experiments.py
import torch
import flwr as fl
from .models import get_model
from .datasets import create_dataloaders
from .client import FlowerClient
from .strategies import BulyanStrategy, RobustFedAvg, VAEByzantineStrategy
from .server import get_evaluate_fn
from .utils import (
    ExperimentLogger,
    find_existing_experiment,
    get_experiment_summary
)
from .plotting import plot_experiments_auto, detect_varying_parameter
import numpy as np
from typing import List
from dataclasses import dataclass, asdict
import os
from datetime import datetime


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


def create_suite_folder(name, base_dir: str = "./plots") -> str:
    """Create a unique subfolder for this experiment suite run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_folder = os.path.join(base_dir, f"{name}_{timestamp}")
    os.makedirs(suite_folder, exist_ok=True)
    return suite_folder


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
    elif config.strategy == "fedadam":
        strategy = fl.server.strategy.FedAdam(**strategy_kwargs)
    elif config.strategy == "fedmedian":
        strategy = fl.server.strategy.FedMedian(**strategy_kwargs)
    elif config.strategy == "krum":
        strategy = fl.server.strategy.Krum(
            num_malicious_clients=int(
                config.num_clients * config.malicious_ratio
            ),
            **strategy_kwargs
        )
    elif config.strategy == "bulyan":
        strategy = BulyanStrategy(
            num_malicious_clients=int(
                config.num_clients * config.malicious_ratio
            ),
            **strategy_kwargs)
    elif config.strategy == "robust":
        strategy = RobustFedAvg(
            **strategy_kwargs,
            trim_ratio=config.trim_ratio
        )
    elif config.strategy == "vae":
        strategy = VAEByzantineStrategy(
            **strategy_kwargs,
            warmup_rounds=0,
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

    print(f"Results will be saved in the './plots/' directory.\n")

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

    # Create unique subfolder for this suite
    suite_name = detect_varying_parameter(config_dicts)
    suite_folder = create_suite_folder(suite_name)
    print(f"Results are saved in: {suite_folder}\n")

    print(f"\n{'#'*60}")
    print(f"# All experiments completed!")
    print(f"# Successful: {len(log_files)}/{len(configs)}")
    print(f"{'#'*60}\n")

    # Generate plots in suite-specific folder
    if plot_results and log_files:
        print(f"\nGenerating visualizations in {suite_folder}...")
        plot_experiments_auto(log_files, config_dicts, suite_folder)

    return list(zip(log_files, all_metrics))
