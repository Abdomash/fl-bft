from fedorch.run_experiments import ExperimentConfig, run_experiment_suite


def main():
    configs = [
        ExperimentConfig(dataset="cifar10", num_clients=10,
                         num_rounds=20, malicious_ratio=0.0),
        ExperimentConfig(dataset="cifar10", num_clients=10,
                         num_rounds=20, malicious_ratio=0.3),
        ExperimentConfig(dataset="cifar10", num_clients=10,
                         num_rounds=20, malicious_ratio=0.5),
        # ExperimentConfig(dataset="cifar10", num_clients=10,
        #                  num_rounds=200, malicious_ratio=0.1),
        # ExperimentConfig(dataset="cifar10", num_clients=10,
        #                  num_rounds=200, malicious_ratio=0.2),
        # ExperimentConfig(dataset="cifar10", num_clients=10,
        #                  num_rounds=200, malicious_ratio=0.3),
        # ExperimentConfig(dataset="cifar10", num_clients=10,
        #                  num_rounds=200, malicious_ratio=0.4),
        # ExperimentConfig(dataset="cifar10", num_clients=10,
        #                  num_rounds=200, malicious_ratio=0.5),
    ]

    run_experiment_suite(
        configs,
        force_rerun=False,  # Use cache
        verbose=True,
        plot_results=True
    )


if __name__ == "__main__":
    main()
