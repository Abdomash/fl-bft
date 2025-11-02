# strategies/custom_strategy.py
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedAvgWithLogging(FedAvg):
    """FedAvg with additional logging capabilities."""

    def __init__(self, *args, log_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_callback = log_callback
        self.round_metrics = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if self.log_callback:
            self.log_callback("fit", server_round, metrics)

        return aggregated_params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        aggregated_loss, metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        if self.log_callback:
            self.log_callback("evaluate", server_round, metrics)

        self.round_metrics.append({
            "round": server_round,
            "loss": aggregated_loss,
            **metrics,
        })

        return aggregated_loss, metrics
