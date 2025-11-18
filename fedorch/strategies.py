import flwr as fl
from typing import Dict, List, Optional, Tuple, Union
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
import numpy as np


class CustomStrategy(fl.server.strategy.FedAvg):
    """
    Custom strategy that can filter malicious clients
    or apply custom aggregation logic
    """

    def __init__(
        self,
        *args,
        filter_malicious: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filter_malicious = filter_malicious

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results with optional malicious filtering"""

        # Filter malicious clients if enabled
        if self.filter_malicious:
            filtered_results = [
                (client, fit_res)
                for client, fit_res in results
                if not fit_res.metrics.get("malicious", False)
            ]

            if len(filtered_results) < len(results):
                print(
                    f"Round {server_round}: Filtered "
                    f"{len(results) - len(filtered_results)} "
                    f"malicious clients"
                )

            results = filtered_results

        # Call parent aggregate_fit
        return super().aggregate_fit(server_round, results, failures)


class RobustFedAvg(fl.server.strategy.FedAvg):
    """
    Robust FedAvg using trimmed mean or median aggregation
    """

    def __init__(
        self,
        *args,
        trim_ratio: float = 0.1,
        use_median: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.trim_ratio = trim_ratio
        self.use_median = use_median

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate using robust statistics"""

        if not results:
            return None, {}

        # Convert results to weights
        weights_results = [
            (
                parameters_to_ndarrays(fit_res.parameters),
                fit_res.num_examples
            )
            for _, fit_res in results
        ]

        # Stack all parameter arrays
        num_layers = len(weights_results[0][0])
        aggregated_weights = []

        for layer_idx in range(num_layers):
            layer_weights = np.array([
                weights[layer_idx] for weights, _ in weights_results
            ])

            if self.use_median:
                # Use median aggregation
                aggregated = np.median(layer_weights, axis=0)
            else:
                # Use trimmed mean
                num_clients = len(layer_weights)
                trim_count = int(num_clients * self.trim_ratio)

                if trim_count > 0:
                    sorted_idx = np.argsort(
                        np.mean(
                            np.abs(layer_weights.reshape(num_clients, -1)),
                            axis=1
                        )
                    )
                    keep_idx = sorted_idx[trim_count:-trim_count]
                    layer_weights = layer_weights[keep_idx]

                aggregated = np.mean(layer_weights, axis=0)

            aggregated_weights.append(aggregated)

        parameters_aggregated = ndarrays_to_parameters(
            aggregated_weights
        )

        # Aggregate metrics
        metrics_aggregated = {}
        if results:
            losses = [
                fit_res.metrics.get("loss", 0) * fit_res.num_examples
                for _, fit_res in results
            ]
            examples = [fit_res.num_examples for _, fit_res in results]
            metrics_aggregated["loss"] = sum(losses) / sum(examples)

        return parameters_aggregated, metrics_aggregated
