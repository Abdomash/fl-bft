# clients/vision_client.py
from typing import Dict, Tuple

import torch
import torch.nn as nn
from flwr.common import Context, NDArrays, Scalar
from torch.utils.data import DataLoader

from .pytorch_client import PyTorchClientApp


class VisionClient(PyTorchClientApp):
    """Vision client for image classification tasks."""

    def __init__(self, context: Context):
        # Get client-specific configuration
        node_config = context.node_config
        partition_id = context.node_id

        # Model setup
        model_name = str(node_config.get("model_name", "unknown"))
        num_classes = int(node_config.get("num_classes", 10))
        in_channels = int(node_config.get("in_channels", 3))

        model = self._create_model(model_name, num_classes, in_channels)
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        super().__init__(model, device)

        # Training setup
        self.train_dataset = node_config.get("train_datasets")[partition_id]
        self.test_dataset = node_config.get("test_dataset")
        self.batch_size = node_config.get("batch_size", 32)
        self.lr = node_config.get("learning_rate", 0.01)

        self.criterion = nn.CrossEntropyLoss()

    def _create_model(
        self, model_name: str, num_classes: int, in_channels: int
    ) -> nn.Module:
        from models.cnn import SimpleCNN, SimpleResNet

        if model_name == "simple_cnn":
            return SimpleCNN(num_classes, in_channels)
        elif model_name == "simple_resnet":
            return SimpleResNet(num_classes, in_channels)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def train(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)

        epochs = int(config.get("local_epochs", 1))
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        total_loss = 0.0
        num_samples = 0

        for _ in range(epochs):
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                num_samples += images.size(0)

        avg_loss = total_loss / num_samples
        return self.get_parameters({}), num_samples, {"train_loss": avg_loss}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)

        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

        self.model.eval()
        total_loss = 0.0
        correct = 0
        num_samples = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                num_samples += images.size(0)

        accuracy = correct / num_samples
        avg_loss = total_loss / num_samples

        return avg_loss, num_samples, {"accuracy": accuracy}
