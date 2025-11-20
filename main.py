from fedorch.run_experiments import ExperimentConfig, run_experiment_suite


def test_cifar_malicious_clients():
    """Test experiments with CIFAR-10 and varying malicious client ratios."""
    configs = []

    for ratio in range(0, 6):
        config = ExperimentConfig(
            dataset="cifar10",
            num_clients=50,
            num_rounds=200,
            malicious_ratio=ratio * 0.1
        )
        configs.append(config)

    run_experiment_suite(
        configs,
        force_rerun=False,
        verbose=True,
        plot_results=True
    )


def test_cifar_scaling_clients():
    """Test experiments with CIFAR-10 and varying number of clients."""
    configs = []

    for num_clients in [10, 20, 50, 100]:
        config = ExperimentConfig(
            dataset="cifar10",
            num_clients=num_clients,
            num_rounds=200,
            malicious_ratio=0.0
        )
        configs.append(config)

    run_experiment_suite(
        configs,
        force_rerun=False,
        verbose=True,
        plot_results=True
    )


def test_cifar_strategy_malicious_tolerance():
    """Test with CIFAR-10 and varying strategies against malicious clients."""

    strategies = ["fedavg", "robust", "krum", "fedmedian", "bulyan", "vae"]
    malicious_ratio = [0.0, 0.1, 0.3, 0.5]

    configs = []
    for ratio in malicious_ratio:
        for strategy in strategies:
            config = ExperimentConfig(
                dataset="cifar10",
                num_clients=50,
                num_rounds=500,
                malicious_ratio=ratio,
                strategy=strategy
            )
            configs.append(config)
        run_experiment_suite(
            configs,
            force_rerun=False,
            verbose=True,
            plot_results=True
        )


'''
def test_fashion_mnist_malicious_clients():
    """Test experiments with Fashion MNIST and varying malicious client ratios."""
    configs = []

    for ratio in range(0, 6):
        config = ExperimentConfig(
            dataset="fashion_mnist",
            num_clients=50,
            num_rounds=200,
            malicious_ratio=ratio * 0.1
        )
        configs.append(config)

    run_experiment_suite(
        configs,
        force_rerun=False,
        verbose=True,
        plot_results=True
    )
'''


def main():
    """Main function to run tests."""

    test_cifar_malicious_clients()
    test_cifar_scaling_clients()
    test_cifar_strategy_malicious_tolerance()


if __name__ == "__main__":
    main()
