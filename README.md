# Federated Learning Robustness Testing Framework

This repository provides a modular and extensible framework for running, analyzing, and visualizing Federated Learning (FL) experiments. It was developed to test various FL configurations, strategies, and robustness scenarios under both simulated and real distributed environments.

> [!NOTE]
> TL;DR: This repo was designed to make it easy to prototype and test Federated Learning behaviors under various controlled conditions. It automates dataset partitioning, client setup, server aggregation, result logging, and report generation. I used it to evaluate several federated robustness strategies and analyze their performance across different attack and data distribution scenarios. This was done for my CS 240 Concurrency final project in Fall 2025 at KAUST.
---

## Overview

The framework integrates **Flower** (for FL orchestration) and **PyTorch** (for model training) to support local simulations and multi-machine deployments. Its goal is to make it simple to reproduce, compare, and analyze the performance of different aggregation strategies, especially under the presence of malicious clients.

The setup allows testing:
- Varying numbers of clients
- IID vs non-IID data partitions
- Different levels of malicious/Byzantine clients
- Various FL aggregation strategies (FedAvg, Median, Krum, Bulyan, VAE-based)

---


## Summary


---

## Repository Structure

```
fedorch/
│
├── client.py           # Client training logic and local update behavior
├── server.py           # Evaluation logic for global model
├── datasets.py         # Dataset loading and partitioning (CIFAR, MNIST)
├── models.py           # CNN models for supported datasets
├── strategies.py       # Federated aggregation algorithms (FedAvg, Krum, etc.)
├── run_experiments.py  # Launches and manages experiments with configs
├── plotting.py         # Generates plots and comparison graphs
├── utils.py            # Experiment logging, caching, summaries
│
├── deploy/             # Minimal versions for real client-server runs
│   ├── client.py       # Standalone Federated client
│   └── server.py       # Standalone Federated server
│
├── logs/               # Simulation experiment logs
├── plots/              # Generated plots and analytical summaries
├── real-logs/          # Logs from real distributed runs
└── main.py             # Example experiment suites and entry point
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
   pip install -r requirements.txt
   ```
   or if `pyproject.toml` is used:
   ```bash
   pip install .
   ```

---

## Usage

The primary entry point for running experiments is `main.py`. It defines several experiment suites to test different scenarios.

Examples:

### 1. Malicious Client Ratio Tests
Run experiments varying the fraction of malicious clients on CIFAR-10:
```bash
python main.py
```
(defaults to `test_cifar_data_distribution_client_scaling()`)

You can edit `main.py` to enable:
```python
test_cifar_malicious_clients()
```

### 2. Strategy Robustness Comparison
Compare multiple aggregation strategies under a fixed malicious ratio:
```python
test_cifar_small_strategy_malicious_tolerance()
```

### 3. Real-World Deployment
To run on separate machines:
- Start the server:
  ```bash
  python fedorch/deploy/server.py --strategy fedavg --malicious-num 1
  ```
- Start each client:
  ```bash
  python fedorch/deploy/client.py --client-id 0 --server <server_ip>:8080
  ```

Each client connects to the central server using gRPC and participates in real FL rounds.

---

## Running Custom Experiments

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

---

## Logs and Visualization

After each experiment:
- Results are saved in `./logs/` (JSON and CSV)
- Plots and summaries are saved in `./plots/`

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

