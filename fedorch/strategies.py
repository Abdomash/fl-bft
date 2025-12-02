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
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from sklearn.decomposition import PCA


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


class BulyanStrategy(fl.server.strategy.FedAvg):
    """
    Bulyan aggregation strategy combining Krum selection with 
    coordinate-wise median-centered trimmed mean.
    More robust than Krum alone.
    """

    def __init__(
        self,
        *args,
        num_malicious_clients: int = 0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_malicious_clients = num_malicious_clients

    def _compute_krum_scores(
        self, weights_list: List[np.ndarray], num_closest: int
    ) -> np.ndarray:
        """Compute Krum scores for each client"""
        n = len(weights_list)

        if num_closest >= n:
            num_closest = n - 1

        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(weights_list[i] - weights_list[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute scores (sum of num_closest closest distances)
        scores = np.zeros(n)
        for i in range(n):
            sorted_distances = np.sort(distances[i])
            # Skip index 0 (distance to itself = 0)
            scores[i] = np.sum(sorted_distances[1:num_closest+1])

        return scores

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate using Bulyan (Krum selection + coordinate-wise trimmed mean)"""

        if not results:
            return None, {}

        n = len(results)
        f = self.num_malicious_clients

        # Need at least 4f + 3 clients for Bulyan
        if n < 4 * f + 3:
            print(f"Warning: Bulyan requires at least {4*f+3} clients, "
                  f"got {n}. Falling back to FedAvg.")
            return super().aggregate_fit(server_round, results, failures)

        # Convert to weights
        weights_results = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        # Flatten all weights
        flattened_weights = [
            np.concatenate([w.flatten() for w in weights])
            for weights in weights_results
        ]

        # Step 1: Select theta = n - 2f clients using iterative Krum
        theta = n - 2 * f
        selected_indices = []
        remaining_indices = list(range(n))

        for iteration in range(theta):
            if len(remaining_indices) <= 1:
                break

            # For Krum, select n - f - 2 closest neighbors
            # But n here is the number of *remaining* clients
            n_remaining = len(remaining_indices)
            m = n_remaining - f - 2

            # Ensure m is valid
            if m < 1:
                m = max(1, n_remaining - 1)

            # Compute Krum scores for remaining clients
            remaining_weights = [
                flattened_weights[i] for i in remaining_indices
            ]
            scores = self._compute_krum_scores(remaining_weights, m)

            # Select client with minimum score
            local_best_idx = np.argmin(scores)
            global_best_idx = remaining_indices[local_best_idx]
            selected_indices.append(global_best_idx)
            remaining_indices.remove(global_best_idx)

        # Step 2: Apply coordinate-wise median-centered trimmed mean
        selected_weights = [weights_results[i] for i in selected_indices]

        # Calculate beta = theta - 2f (number of values to keep per coordinate)
        beta = theta - 2 * f

        if beta < 1:
            print(f"Warning: beta={beta} < 1, using all selected clients")
            beta = len(selected_weights)

        # Aggregate each layer
        num_layers = len(selected_weights[0])
        aggregated_weights = []

        for layer_idx in range(num_layers):
            layer_weights = np.array([
                weights[layer_idx] for weights in selected_weights
            ])

            # Apply coordinate-wise trimmed mean around median
            # Shape: (num_selected_clients, *layer_shape)
            original_shape = layer_weights.shape[1:]

            # Flatten spatial dimensions to process all coordinates
            layer_flat = layer_weights.reshape(len(selected_weights), -1)

            # For each coordinate (parameter)
            aggregated_flat = np.zeros(layer_flat.shape[1])

            for coord_idx in range(layer_flat.shape[1]):
                coord_values = layer_flat[:, coord_idx]

                # Compute median
                median_val = np.median(coord_values)

                # Compute distances to median
                distances_to_median = np.abs(coord_values - median_val)

                # Select beta closest values
                if beta < len(coord_values):
                    closest_indices = np.argpartition(
                        distances_to_median, beta - 1
                    )[:beta]
                else:
                    closest_indices = np.arange(len(coord_values))

                # Average the beta closest values
                aggregated_flat[coord_idx] = np.mean(
                    coord_values[closest_indices]
                )

            # Reshape back to original layer shape
            aggregated = aggregated_flat.reshape(original_shape)
            aggregated_weights.append(aggregated)

        print(f"Round {server_round}: Bulyan selected "
              f"{len(selected_indices)}/{n} clients, "
              f"aggregated beta={beta} per coordinate")

        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)

        # Aggregate metrics
        metrics_aggregated = {}
        if results:
            losses = [
                fit_res.metrics.get("loss", 0) * fit_res.num_examples
                for _, fit_res in results
            ]
            examples = [fit_res.num_examples for _, fit_res in results]
            metrics_aggregated["loss"] = sum(losses) / sum(examples)
            metrics_aggregated["num_selected"] = len(selected_indices)
            metrics_aggregated["beta"] = beta

        return parameters_aggregated, metrics_aggregated


class GradientVAE(nn.Module):
    """Variational Autoencoder for gradient anomaly detection"""

    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        hidden_dim = 256

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def reconstruction_loss(self, x, x_recon, mu, logvar):
        """Compute VAE loss (reconstruction + KL divergence)"""
        recon_loss = nn.functional.mse_loss(
            x_recon, x, reduction='sum'
        )
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld


class VAEByzantineStrategy(fl.server.strategy.FedAvg):
    """
    VAE-based Byzantine-robust aggregation strategy.

    Uses a Variational Autoencoder to learn the distribution of normal
    client updates and identify anomalies based on reconstruction error.
    Clients with high reconstruction errors receive lower weights in
    aggregation.
    """

    def __init__(
        self,
        *args,
        latent_dim: int = 32,
        confidence_decay: float = 0.9,
        anomaly_threshold: float = 2.0,
        vae_lr: float = 0.001,
        vae_epochs: int = 10,
        warmup_rounds: int = 5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.confidence_decay = confidence_decay
        self.anomaly_threshold = anomaly_threshold
        self.vae_lr = vae_lr
        self.vae_epochs = vae_epochs
        self.warmup_rounds = warmup_rounds

        self.vae = None
        self.optimizer = None
        self.pca = None
        self.max_input_dim = 1000  # Max dimensions for PCA
        self.client_confidence = defaultdict(lambda: 1.0)
        self.reconstruction_history = defaultdict(list)

    def _reduce_dimensions(
        self, gradients: np.ndarray, fit: bool = False
    ) -> np.ndarray:
        """Reduce gradient dimensionality using PCA"""
        if self.pca is None:
            n_components = min(self.max_input_dim,
                               gradients.shape[1], gradients.shape[0])
            self.pca = PCA(n_components=n_components)
            print(f"Reducing dimensions from {gradients.shape[1]} "
                  f"to {n_components}")
            return self.pca.fit_transform(gradients)
        elif fit:
            return self.pca.fit_transform(gradients)
        else:
            return self.pca.transform(gradients)

    def _flatten_gradients(self, weights: List[np.ndarray]) -> np.ndarray:
        """Flatten model weights into a single vector"""
        return np.concatenate([w.flatten() for w in weights])

    def _initialize_vae(self, input_dim: int):
        """Initialize VAE with appropriate dimensions"""
        if self.vae is None:
            self.vae = GradientVAE(input_dim, self.latent_dim)
            self.optimizer = optim.Adam(self.vae.parameters(), lr=self.vae_lr)
            print(f"Initialized VAE with input_dim={input_dim}, "
                  f"latent_dim={self.latent_dim}")

    def _train_vae(self, gradients: torch.Tensor):
        """Train VAE on current round's gradients"""
        self.vae.train()
        for epoch in range(self.vae_epochs):
            self.optimizer.zero_grad()

            x_recon, mu, logvar = self.vae(gradients)
            loss = self.vae.reconstruction_loss(
                gradients, x_recon, mu, logvar
            )

            loss.backward()
            self.optimizer.step()

    def _compute_reconstruction_errors(
        self, gradients: torch.Tensor
    ) -> np.ndarray:
        """Compute reconstruction error for each client's gradients"""
        self.vae.eval()
        with torch.no_grad():
            x_recon, mu, logvar = self.vae(gradients)
            errors = torch.mean((gradients - x_recon) ** 2, dim=1)
        return errors.numpy()

    def _update_confidence_scores(
        self,
        client_ids: List[str],
        reconstruction_errors: np.ndarray,
        server_round: int
    ):
        """Update confidence scores based on reconstruction errors"""
        # Normalize errors
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors) + 1e-8
        normalized_errors = (reconstruction_errors - mean_error) / std_error

        # Update confidence scores
        for client_id, norm_error in zip(client_ids, normalized_errors):
            # Store reconstruction error history
            self.reconstruction_history[client_id].append(norm_error)

            # Compute new confidence (lower for high errors)
            anomaly_score = max(0, norm_error - self.anomaly_threshold)
            new_confidence = np.exp(-anomaly_score)

            # Exponentially weighted average
            old_confidence = self.client_confidence[client_id]
            self.client_confidence[client_id] = (
                self.confidence_decay * old_confidence +
                (1 - self.confidence_decay) * new_confidence
            )

        # Log statistics
        avg_confidence = np.mean(list(self.client_confidence.values()))
        suspicious_clients = sum(
            1 for c in self.client_confidence.values() if c < 0.5
        )
        print(f"Round {server_round}: Avg confidence={avg_confidence:.3f}, "
              f"Suspicious clients={suspicious_clients}/{len(client_ids)}")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate with VAE-based weighting"""

        if not results:
            return None, {}

        # Extract weights and client IDs
        weights_results = []
        client_ids = []
        for client_proxy, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            weights_results.append((weights, fit_res.num_examples))
            client_ids.append(client_proxy.cid)

        # Flatten gradients
        flattened_grads = np.array([
            self._flatten_gradients(w) for w, _ in weights_results
        ])

        # Reduce dimensions if necessary
        flattened_grads = self._reduce_dimensions(flattened_grads)

        # Initialize VAE on first round
        if self.vae is None:
            self._initialize_vae(flattened_grads.shape[1])

        # Convert to torch tensors
        grad_tensor = torch.FloatTensor(flattened_grads)

        # Train VAE (skip during warmup to build initial distribution)
        if server_round > self.warmup_rounds:
            self._train_vae(grad_tensor)

            # Compute reconstruction errors
            recon_errors = self._compute_reconstruction_errors(grad_tensor)

            # Update confidence scores
            self._update_confidence_scores(
                client_ids, recon_errors, server_round
            )
        else:
            print(f"Round {server_round}: Warmup phase, "
                  f"skipping anomaly detection")

        # Apply confidence-based weighting
        weighted_results = []
        for (weights, num_examples), client_id in zip(
            weights_results, client_ids
        ):
            confidence = self.client_confidence[client_id]
            # Weight by both confidence and number of examples
            effective_weight = num_examples * confidence
            weighted_results.append((weights, effective_weight))

        # Aggregate with confidence-weighted averaging
        total_weight = sum(w for _, w in weighted_results)
        if total_weight == 0:
            # Fallback to uniform weighting
            total_weight = len(weighted_results)
            weighted_results = [
                (w, 1.0) for w, _ in weighted_results
            ]

        # Perform weighted aggregation
        num_layers = len(weighted_results[0][0])
        aggregated_weights = []

        for layer_idx in range(num_layers):
            layer_arrays = [
                weights[layer_idx] * (weight / total_weight)
                for weights, weight in weighted_results
            ]
            aggregated = np.sum(layer_arrays, axis=0)
            aggregated_weights.append(aggregated)

        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)

        # Aggregate metrics
        metrics_aggregated = {}
        if results:
            # Use original weights for metric aggregation
            losses = [
                fit_res.metrics.get("loss", 0) * fit_res.num_examples
                for _, fit_res in results
            ]
            examples = [fit_res.num_examples for _, fit_res in results]
            metrics_aggregated["loss"] = sum(losses) / sum(examples)

            # Add confidence statistics
            confidences = [
                self.client_confidence[cid] for cid in client_ids
            ]
            metrics_aggregated["avg_confidence"] = float(np.mean(confidences))
            metrics_aggregated["min_confidence"] = float(np.min(confidences))

        return parameters_aggregated, metrics_aggregated


class KrumStrategy(fl.server.strategy.FedAvg):
    """A Custom implementation of the Multi-Krum aggregation strategy."""

    def __init__(
        self,
        *args,
        num_malicious_clients: int = 0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_malicious_clients = num_malicious_clients

    def _compute_krum_scores(
        self, weights_list: List[np.ndarray], num_closest: int
    ) -> np.ndarray:
        """Compute Krum scores for each client"""
        n = len(weights_list)

        if num_closest >= n:
            num_closest = n - 1

        # Compute pairwise squared distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                # Use squared Euclidean distance
                dist = np.sum((weights_list[i] - weights_list[j]) ** 2)
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute scores (sum of num_closest closest distances)
        scores = np.zeros(n)
        for i in range(n):
            sorted_distances = np.sort(distances[i])
            # Skip index 0 (distance to itself = 0)
            scores[i] = np.sum(sorted_distances[1:num_closest+1])

        return scores

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate using Multi-Krum"""

        if not results:
            return None, {}

        if len(results) < 3:
            print(f"Warning: Krum requires at least 3 clients, got {
                  len(results)}")
            aggregated_weights = parameters_to_ndarrays(
                results[0][1].parameters)
            return ndarrays_to_parameters(aggregated_weights), {}

        weights_results = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        flattened_weights = [
            np.concatenate([w.flatten() for w in weights])
            for weights in weights_results
        ]

        n = len(flattened_weights)
        m = max(1, min(n - self.num_malicious_clients - 2, n - 1))

        scores = self._compute_krum_scores(flattened_weights, m)

        # Select top k clients with lowest scores (most trustworthy)
        k = n - self.num_malicious_clients  # Or use a fixed k
        top_k_indices = np.argsort(scores)[:k]

        print(
            f"Round {server_round}: Multi-Krum selected clients {top_k_indices.tolist()}")

        # Average the selected clients' weights
        aggregated_weights = [
            np.mean([weights_results[i][j] for i in top_k_indices], axis=0)
            for j in range(len(weights_results[0]))
        ]

        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)

        # Aggregate metrics
        metrics_aggregated = {}
        if results:
            losses = [
                fit_res.metrics.get("loss", 0) * fit_res.num_examples
                for _, fit_res in results
            ]
            examples = [fit_res.num_examples for _, fit_res in results]
            if sum(examples) > 0:
                metrics_aggregated["loss"] = sum(losses) / sum(examples)

        return parameters_aggregated, metrics_aggregated
