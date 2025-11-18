from typing import Dict, Type

import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple CNN for image classification."""

    def __init__(self, num_classes: int = 10, num_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for ResNet."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """ResNet-18 architecture."""

    def __init__(self, num_classes: int = 10, num_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_channels, 64, 3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int
    ):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list = None,
        num_classes: int = 10,
        dropout: float = 0.5,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


class LeNet5(nn.Module):
    """LeNet-5 architecture."""

    def __init__(self, num_classes: int = 10, num_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.pool = nn.AvgPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGG11(nn.Module):
    """VGG-11 architecture."""

    def __init__(self, num_classes: int = 10, num_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(num_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Model factory
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "SimpleCNN": SimpleCNN,
    "ResNet18": ResNet18,
    "MLP": MLP,
    "LeNet5": LeNet5,
    "VGG11": VGG11,
}


def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_name: Name of the model to create
        **kwargs: Additional parameters to pass to the model constructor

    Returns:
        Instantiated model

    Example:
        model = get_model("SimpleCNN", num_classes=10, num_channels=3)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_name]
    return model_class(**kwargs)


def register_model(name: str, model_class: Type[nn.Module]):
    """
    Register a custom model to the factory.

    Args:
        name: Name to register the model under
        model_class: Model class to register

    Example:
        register_model("CustomModel", CustomModel)
    """
    MODEL_REGISTRY[name] = model_class


def list_models() -> list:
    """Return list of available model names."""
    return list(MODEL_REGISTRY.keys())
