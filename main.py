from fedorch.run_experiments import ExperimentConfig, run_multiple_experiments


def main():
    configs = [
        ExperimentConfig(dataset="cifar10", num_clients=10,
                         num_rounds=20, malicious_ratio=0.0),
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
    run_multiple_experiments(configs)

    # configs = [
    #     ExperimentConfig(dataset="cifar10", num_clients=50,
    #                      num_rounds=200, malicious_ratio=0.0),
    #     ExperimentConfig(dataset="cifar10", num_clients=50,
    #                      num_rounds=200, malicious_ratio=0.1),
    #     ExperimentConfig(dataset="cifar10", num_clients=50,
    #                      num_rounds=200, malicious_ratio=0.2),
    #     ExperimentConfig(dataset="cifar10", num_clients=50,
    #                      num_rounds=200, malicious_ratio=0.3),
    #     ExperimentConfig(dataset="cifar10", num_clients=50,
    #                      num_rounds=200, malicious_ratio=0.4),
    #     ExperimentConfig(dataset="cifar10", num_clients=50,
    #                      num_rounds=200, malicious_ratio=0.5),
    # ]
    # run_multiple_experiments(configs)
    #
    # configs = [
    #     ExperimentConfig(dataset="cifar10", num_clients=100,
    #                      num_rounds=200, malicious_ratio=0.0),
    #     ExperimentConfig(dataset="cifar10", num_clients=100,
    #                      num_rounds=200, malicious_ratio=0.1),
    #     ExperimentConfig(dataset="cifar10", num_clients=100,
    #                      num_rounds=200, malicious_ratio=0.2),
    #     ExperimentConfig(dataset="cifar10", num_clients=100,
    #                      num_rounds=200, malicious_ratio=0.3),
    #     ExperimentConfig(dataset="cifar10", num_clients=100,
    #                      num_rounds=200, malicious_ratio=0.4),
    #     ExperimentConfig(dataset="cifar10", num_clients=100,
    #                      num_rounds=200, malicious_ratio=0.5),
    # ]
    # run_multiple_experiments(configs)


if __name__ == "__main__":
    main()
