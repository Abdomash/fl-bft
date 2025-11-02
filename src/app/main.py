from experiment.config import ExperimentConfig
from experiment.runner import ExperimentRunner


def create_experiments():
    """Define all experiments to run."""
    experiments = []

    # Experiment 1: CIFAR-10 with SimpleCNN, IID, FedAvg
    experiments.append(
        ExperimentConfig(
            name="cifar10_simplecnn_iid_fedavg",
            dataset="cifar10",
            dataset_config={},
            partition_config={"method": "iid"},
            num_clients=10,
            client_models=["simple_cnn"],
            strategy_name="FedAvg",
            strategy_config={},
            num_rounds=20,
            local_epochs=2,
            batch_size=32,
            learning_rate=0.01,
            seed=42,
        )
    )

    # Experiment 2: CIFAR-10 with SimpleCNN, Non-IID, FedAvg
    experiments.append(
        ExperimentConfig(
            name="cifar10_simplecnn_noniid_fedavg",
            dataset="cifar10",
            dataset_config={},
            partition_config={"method": "dirichlet", "alpha": 0.5},
            num_clients=10,
            client_models=["simple_cnn"],
            strategy_name="FedAvg",
            strategy_config={},
            num_rounds=20,
            local_epochs=2,
            batch_size=32,
            learning_rate=0.01,
            seed=42,
        )
    )

    # Experiment 3: CIFAR-10 with SimpleResNet, Non-IID, FedProx
    experiments.append(
        ExperimentConfig(
            name="cifar10_resnet_noniid_fedprox",
            dataset="cifar10",
            dataset_config={},
            partition_config={"method": "dirichlet", "alpha": 0.5},
            num_clients=10,
            client_models=["simple_resnet"],
            strategy_name="FedProx",
            strategy_config={"proximal_mu": 0.01},
            num_rounds=20,
            local_epochs=2,
            batch_size=32,
            learning_rate=0.01,
            seed=42,
        )
    )

    # Experiment 4: MNIST with SimpleCNN, IID, FedAvg
    experiments.append(
        ExperimentConfig(
            name="mnist_simplecnn_iid_fedavg",
            dataset="mnist",
            dataset_config={},
            partition_config={"method": "iid"},
            num_clients=10,
            client_models=["simple_cnn"],
            strategy_name="FedAvg",
            strategy_config={},
            num_rounds=15,
            local_epochs=1,
            batch_size=64,
            learning_rate=0.01,
            seed=42,
        )
    )

    # Experiment 5: MNIST with SimpleCNN, Non-IID (high alpha), FedAvg
    experiments.append(
        ExperimentConfig(
            name="mnist_simplecnn_noniid_alpha1_fedavg",
            dataset="mnist",
            dataset_config={},
            partition_config={"method": "dirichlet", "alpha": 1.0},
            num_clients=10,
            client_models=["simple_cnn"],
            strategy_name="FedAvg",
            strategy_config={},
            num_rounds=15,
            local_epochs=1,
            batch_size=64,
            learning_rate=0.01,
            seed=42,
        )
    )

    # Experiment 6: MNIST with SimpleCNN, Non-IID (low alpha), FedAvg
    experiments.append(
        ExperimentConfig(
            name="mnist_simplecnn_noniid_alpha01_fedavg",
            dataset="mnist",
            dataset_config={},
            partition_config={"method": "dirichlet", "alpha": 0.1},
            num_clients=10,
            client_models=["simple_cnn"],
            strategy_name="FedAvg",
            strategy_config={},
            num_rounds=15,
            local_epochs=1,
            batch_size=64,
            learning_rate=0.01,
            seed=42,
        )
    )

    return experiments


def main():
    """Run all experiments."""
    experiments = create_experiments()

    print(f"Total experiments to run: {len(experiments)}")

    all_results = []
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{len(experiments)}")
        print(f"{'='*60}")

        try:
            runner = ExperimentRunner(exp_config)
            results = runner.run()
            all_results.append(results)
        except Exception as e:
            print(f"Error in experiment {exp_config.name}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
