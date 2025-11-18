from fedorch.runner import ExperimentRunner
from fedorch.config import ExperimentConfig


def main():
    # Example usage with factories
    config = ExperimentConfig(
        name="test_experiment",
        description="Test with model and strategy factories",
        model_name="SimpleCNN",
        model_params={"num_classes": 10, "num_channels": 3},
        dataset_name="uoft-cs/cifar10",
        partition_type="dirichlet",
        dirichlet_alpha=0.5,
        num_clients=10,
        num_byzantine_clients=2,
        num_rounds=5,
        strategy_name="median",
        strategy_params={},
    )

    runner = ExperimentRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
