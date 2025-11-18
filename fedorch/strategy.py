from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, Strategy


class MedianStrategy(FedAvg):
    """Federated learning strategy using coordinate-wise median."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model parameters using coordinate-wise median."""
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        weights_results = [
            (
                parameters_to_ndarrays(fit_res.parameters),
                fit_res.num_examples,
            )
            for _, fit_res in results
        ]

        all_weights = [weights for weights, _ in weights_results]

        # Compute coordinate-wise median
        median_weights = []
        for layer_idx in range(len(all_weights[0])):
            layer_weights = np.array(
                [client_weights[layer_idx] for client_weights in all_weights]
            )
            median_layer = np.median(layer_weights, axis=0)
            median_weights.append(median_layer)

        parameters_aggregated = ndarrays_to_parameters(median_weights)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (res.num_examples, res.metrics) for _, res in results
            ]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


class TrimmedMeanStrategy(FedAvg):
    """Federated learning strategy using trimmed mean aggregation."""

    def __init__(self, trim_ratio: float = 0.1, *args, **kwargs):
        """
        Args:
            trim_ratio: Fraction of extreme values to trim (from each end)
        """
        super().__init__(*args, **kwargs)
        self.trim_ratio = trim_ratio

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model parameters using trimmed mean."""
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        weights_results = [
            (
                parameters_to_ndarrays(fit_res.parameters),
                fit_res.num_examples,
            )
            for _, fit_res in results
        ]

        all_weights = [weights for weights, _ in weights_results]

        trimmed_weights = []
        for layer_idx in range(len(all_weights[0])):
            layer_weights = np.array(
                [client_weights[layer_idx] for client_weights in all_weights]
            )

            num_clients = layer_weights.shape[0]
            num_trim = int(num_clients * self.trim_ratio)

            if num_trim > 0:
                sorted_weights = np.sort(layer_weights, axis=0)
                trimmed = sorted_weights[num_trim:-num_trim]
                trimmed_layer = np.mean(trimmed, axis=0)
            else:
                trimmed_layer = np.mean(layer_weights, axis=0)

            trimmed_weights.append(trimmed_layer)

        parameters_aggregated = ndarrays_to_parameters(trimmed_weights)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (res.num_examples, res.metrics) for _, res in results
            ]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated


class KrumStrategy(FedAvg):
    """Krum aggregation strategy for Byzantine-robust federated learning."""

    def __init__(self, num_byzantine: int = 0, *args, **kwargs):
        """
        Args:
            num_byzantine: Expected number of Byzantine clients
        """
        super().__init__(*args, **kwargs)
        self.num_byzantine = num_byzantine

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate using Krum algorithm."""
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        weights_results = [
            (
                parameters_to_ndarrays(fit_res.parameters),
                fit_res.num_examples,
            )
            for _, fit_res in results
        ]

        all_weights = [weights for weights, _ in weights_results]
        n = len(all_weights)
        f = self.num_byzantine

        # Flatten weights for distance computation
        flattened = [
            np.concatenate([layer.flatten() for layer in weights])
            for weights in all_weights
        ]

        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(flattened[i] - flattened[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute scores (sum of distances to n-f-2 closest clients)
        scores = np.zeros(n)
        for i in range(n):
            sorted_distances = np.sort(distances[i])
            scores[i] = np.sum(sorted_distances[: n - f - 2])

        # Select client with minimum score
        selected_idx = np.argmin(scores)
        selected_weights = all_weights[selected_idx]

        parameters_aggregated = ndarrays_to_parameters(selected_weights)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (res.num_examples, res.metrics) for _, res in results
            ]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated


# Strategy factory
STRATEGY_REGISTRY: Dict[str, Type[Strategy]] = {
    "fedavg": FedAvg,
    "median": MedianStrategy,
    "trimmed_mean": TrimmedMeanStrategy,
    "krum": KrumStrategy,
}


def get_strategy(
    strategy_name: str,
    fit_metrics_agg_fn: Optional[Callable] = None,
    evaluate_metrics_agg_fn: Optional[Callable] = None,
    **kwargs,
) -> Strategy:
    """
    Factory function to create strategies.

    Args:
        strategy_name: Name of the strategy
        fit_metrics_agg_fn: Function to aggregate fit metrics
        evaluate_metrics_agg_fn: Function to aggregate evaluation metrics
        **kwargs: Additional parameters for the strategy

    Returns:
        Instantiated strategy

    Example:
        strategy = get_strategy(
            "median",
            fraction_fit=0.5,
            min_fit_clients=5
        )
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available strategies: {list(STRATEGY_REGISTRY.keys())}"
        )

    strategy_class = STRATEGY_REGISTRY[strategy_name]

    # Add metric aggregation functions if provided
    if fit_metrics_agg_fn is not None:
        kwargs["fit_metrics_aggregation_fn"] = fit_metrics_agg_fn
    if evaluate_metrics_agg_fn is not None:
        kwargs["evaluate_metrics_aggregation_fn"] = evaluate_metrics_agg_fn

    return strategy_class(**kwargs)


def register_strategy(name: str, strategy_class: Type[Strategy]):
    """Register a custom strategy."""
    STRATEGY_REGISTRY[name] = strategy_class


def list_strategies() -> list:
    """Return list of available strategy names."""
    return list(STRATEGY_REGISTRY.keys())
