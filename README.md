<img width="1574" height="623" alt="image" src="https://github.com/user-attachments/assets/e791389c-d0c8-449f-a61c-61e3371a1821" />

# Federated Learning with Byzantine Fault Tolerance

This repository provides an extensible framework for running, analyzing, and visualizing Federated Learning (FL) experiments. It was developed to test various FL configurations, strategies, and robustness scenarios under both simulated and real distributed environments.

> [!NOTE]
> I used this repository to evaluate several federated robustness strategies and analyze their performance across different attack and data distribution scenarios. This was done for my CS 240 Concurrency final project in Fall 2025 at KAUST. You can find the project report [here](cs240_final_project.pdf).
---

## Overview

The framework integrates **Flower** (for FL orchestration) and **PyTorch** (for model training) to support local simulations and multi-machine deployments. Its goal is to make it simple to reproduce, compare, and analyze the performance of different aggregation strategies, especially under the presence of malicious clients.

The setup allows testing:
- Varying numbers of clients
- IID vs non-IID data partitions
- Different levels of malicious/Byzantine clients
- Various FL aggregation strategies (FedAvg, Median, Krum, Bulyan, VAE-based)

---

## Repository Structure

```
.
├── logs/                     # Simulation experiment logs
├── plots/                    # Generated plots and analytical summaries
├── real-logs/                # Logs from real distributed runs
├── real-plots/               # Plots for the real distributed runs
├── main.py                   # Run local simulations
└── fedorch/
    │
    ├── client.py             # Client training logic and local update behavior
    ├── server.py             # Evaluation logic for global model
    ├── datasets.py           # Dataset loading and partitioning (CIFAR, MNIST)
    ├── models.py             # CNN models for supported datasets
    ├── strategies.py         # Federated aggregation algorithms (FedAvg, Krum, etc.)
    ├── run_experiments.py    # Launches and manages experiments with configs
    ├── plotting.py           # Generates plots and comparison graphs
    ├── utils.py              # Experiment logging, caching, summaries
    │
    └── deploy/               # Minimal versions for real client-server runs
        ├── client.py         # All-in-one FL client
        └── server.py         # All-in-one FL server

```

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd <repo_name>
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```
   or if you prefer `pip`:
   ```bash
   pip install .
   ```

---

## Usage

### Defining Experiments

You can define a new setup directly using the `ExperimentConfig` dataclass:

```python
from fedorch.run_experiments import ExperimentConfig, run_experiment

config = ExperimentConfig(
    dataset="cifar10",
    num_clients=10,
    num_rounds=100,
    malicious_ratio=0.2,
    strategy="krum"
)

run_experiment(config)
```

or for a suite of experiments:

```python
from fedorch.run_experiments import run_experiment_suite

configs = [
    ExperimentConfig(strategy="fedavg", malicious_ratio=0.0),
    ExperimentConfig(strategy="krum", malicious_ratio=0.2),
]

run_experiment_suite(configs)
```

Results are automatically logged and visualized.


#### Experiment Configuration Options

- dataset: "cifar10" | "mnist" = "cifar10"
- num_clients: int = 10
- malicious_ratio: float = 0.0
- strategy: "fedavg" | "fedmedian" | "krum" | "bulyan" | "vae" | "robust" = "fedavg"
- num_rounds: int = 50
- local_epochs: int = 1
- batch_size: int = 32
- learning_rate: float = 0.01
- iid: bool = True
- filter_malicious: bool = False
- trim_ratio: float = 0.1
- device: "cpu" | "cuda" = "cpu"

---

### Running Local Simulation

To run experiments locally simulating multiple clients:
```bash
python main.py
```

I have several functions pre-defined to run different experiment suites:
- `test_cifar_malicious_clients()`
- `test_cifar_scaling_clients()`
- `test_cifar_small_strategy_malicious_tolerance()`
- `test_cifar_real_strategy_malicious_tolerance()`
- `test_cifar_real_strategy_malicious_tolerance()`
- `test_cifar_data_distribution_client_scaling()`

You can delete, modify, or add to those experiments as needed.


### Real-World Deployment

> [!NOTE]
> The server.py and client.py in the `deploy/` folder are copied-and-paste minimal versions of the main framework files, modified to run in a distributed setting. If you add new strategies or models, remember to update these files accordingly.

To run on separate machines:
- Start the server:
  ```bash
  python fedorch/deploy/server.py --strategy fedavg --malicious-num 1
  ```
- Start each client:
  ```bash
  python fedorch/deploy/client.py --client-id 0 --server <server_ip>:8080
  ```

> [!TIP]
> The client automatically closes after a server finishes an experiment. If you want to run multiple experiments automatically, you can run the client on a bash loop to ensure it reconnects for each new experiment.

Each client connects to the central server using gRPC and participates in real FL rounds.

---


## Logs and Visualization

After each experiment:
- Results are saved in `logs/` (JSON and CSV)
- Plots and summaries are saved in `plots/`

Generated plots include:
- Accuracy and loss curves per round
- Strategy performance comparison
- Effect of malicious ratio or client count
- Summary tables in PNG, PDF, and CSV

---

## Extending the Framework

To add a new aggregation strategy:
1. Implement a new class in `fedorch/strategies.py` inheriting from `flwr.server.strategy.FedAvg`.
2. Override the `aggregate_fit` method.
3. Register the strategy in:
   - `fedorch/run_experiments.py`
   - Optionally, add to `deploy/server.py` for distributed runs.

