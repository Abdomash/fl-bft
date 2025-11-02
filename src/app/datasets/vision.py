# datasets/vision.py
from typing import Any, Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from .base import FederatedDataset, iid_partition, non_iid_dirichlet


class CIFAR10Dataset(FederatedDataset):
    """CIFAR-10 federated dataset."""

    def __init__(self, data_root: str = "./data"):
        self.data_root = data_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            ),
        ])

    def load(self) -> Tuple[Dataset, Dataset]:
        train = datasets.CIFAR10(
            self.data_root, train=True, download=True, transform=self.transform
        )
        test = datasets.CIFAR10(
            self.data_root, train=False, download=True, transform=self.transform
        )
        return train, test

    def partition(
        self, train_data: Dataset, num_clients: int, config: Dict[str, Any]
    ) -> List[Dataset]:
        method = config.get("method", "iid")
        labels = np.array([train_data[i][1] for i in range(len(train_data))])

        if method == "iid":
            partitions = iid_partition(train_data, num_clients)
        elif method == "dirichlet":
            alpha = config.get("alpha", 0.5)
            partitions = non_iid_dirichlet(labels, num_clients, alpha)
        else:
            raise ValueError(f"Unknown partitioning method: {method}")

        return [Subset(train_data, indices) for indices in partitions]


class MNISTDataset(FederatedDataset):
    """MNIST federated dataset."""

    def __init__(self, data_root: str = "./data"):
        self.data_root = data_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def load(self) -> Tuple[Dataset, Dataset]:
        train = datasets.MNIST(
            self.data_root, train=True, download=True, transform=self.transform
        )
        test = datasets.MNIST(
            self.data_root, train=False, download=True, transform=self.transform
        )
        return train, test

    def partition(
        self, train_data: Dataset, num_clients: int, config: Dict[str, Any]
    ) -> List[Dataset]:
        method = config.get("method", "iid")
        labels = np.array([train_data[i][1] for i in range(len(train_data))])

        if method == "iid":
            partitions = iid_partition(train_data, num_clients)
        elif method == "dirichlet":
            alpha = config.get("alpha", 0.5)
            partitions = non_iid_dirichlet(labels, num_clients, alpha)
        else:
            raise ValueError(f"Unknown partitioning method: {method}")

        return [Subset(train_data, indices) for indices in partitions]
