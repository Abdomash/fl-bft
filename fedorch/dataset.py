from typing import Callable, Dict, Tuple, Type

import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    IidPartitioner,
    Partitioner,
)
from torch.utils.data import DataLoader
from torchvision import transforms


# Transform registry
def get_cifar10_transforms() -> Tuple[Callable, Callable]:
    """Get transforms for CIFAR-10."""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    return train_transform, test_transform


def get_mnist_transforms() -> Tuple[Callable, Callable]:
    """Get transforms for MNIST."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return transform, transform


def get_fashion_mnist_transforms() -> Tuple[Callable, Callable]:
    """Get transforms for Fashion-MNIST."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    return transform, transform


def get_cifar100_transforms() -> Tuple[Callable, Callable]:
    """Get transforms for CIFAR-100."""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),
        ]
    )
    return train_transform, test_transform


def get_default_transforms() -> Tuple[Callable, Callable]:
    """Get default transforms."""
    transform = transforms.ToTensor()
    return transform, transform


TRANSFORM_REGISTRY: Dict[str, Callable] = {
    "uoft-cs/cifar10": get_cifar10_transforms,
    "cifar10": get_cifar10_transforms,
    "mnist": get_mnist_transforms,
    "fashion_mnist": get_fashion_mnist_transforms,
    "uoft-cs/cifar100": get_cifar100_transforms,
    "cifar100": get_cifar100_transforms,
}


def get_transforms(dataset_name: str) -> Tuple[Callable, Callable]:
    """
    Factory function to get transforms for a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Tuple of (train_transform, test_transform)
    """
    # Normalize dataset name
    dataset_key = dataset_name.lower()

    if dataset_key in TRANSFORM_REGISTRY:
        return TRANSFORM_REGISTRY[dataset_key]()

    # Check if dataset name contains known datasets
    for key in TRANSFORM_REGISTRY.keys():
        if key in dataset_key:
            return TRANSFORM_REGISTRY[key]()

    return get_default_transforms()


# Partitioner factory
def create_iid_partitioner(num_partitions: int, **kwargs) -> Partitioner:
    """Create IID partitioner."""
    return IidPartitioner(num_partitions=num_partitions)


def create_dirichlet_partitioner(
    num_partitions: int, alpha: float = 0.5, **kwargs
) -> Partitioner:
    """Create Dirichlet partitioner."""
    return DirichletPartitioner(
        num_partitions=num_partitions, partition_by="label", alpha=alpha
    )


PARTITIONER_REGISTRY: Dict[str, Callable] = {
    "iid": create_iid_partitioner,
    "dirichlet": create_dirichlet_partitioner,
    "non-iid": create_dirichlet_partitioner,
}


def get_partitioner(
    partition_type: str, num_partitions: int, **kwargs
) -> Partitioner:
    """
    Factory function to create partitioners.

    Args:
        partition_type: Type of partitioning ('iid', 'dirichlet', etc.)
        num_partitions: Number of client partitions
        **kwargs: Additional parameters (e.g., alpha for Dirichlet)

    Returns:
        Partitioner instance

    Example:
        partitioner = get_partitioner("dirichlet", 10, alpha=0.5)
    """
    if partition_type not in PARTITIONER_REGISTRY:
        raise ValueError(
            f"Unknown partition type: {partition_type}. "
            f"Available types: {list(PARTITIONER_REGISTRY.keys())}"
        )

    creator_fn = PARTITIONER_REGISTRY[partition_type]
    return creator_fn(num_partitions=num_partitions, **kwargs)


def register_partitioner(name: str, creator_fn: Callable):
    """Register a custom partitioner creator."""
    PARTITIONER_REGISTRY[name] = creator_fn


def register_transform(dataset_name: str, transform_fn: Callable):
    """Register custom transforms for a dataset."""
    TRANSFORM_REGISTRY[dataset_name] = transform_fn


class DatasetManager:
    """Manage federated datasets using flwr-datasets."""

    def __init__(
        self,
        dataset_name: str,
        num_partitions: int,
        partition_type: str = "iid",
        batch_size: int = 32,
        **partition_kwargs,
    ):
        """
        Args:
            dataset_name: Name of dataset (e.g., 'uoft-cs/cifar10')
            num_partitions: Number of client partitions
            partition_type: Type of partitioning ('iid', 'dirichlet', etc.)
            batch_size: Batch size for DataLoaders
            **partition_kwargs: Additional parameters for partitioner
        """
        self.dataset_name = dataset_name
        self.num_partitions = num_partitions
        self.batch_size = batch_size

        # Create partitioner using factory
        partitioner = get_partitioner(
            partition_type, num_partitions, **partition_kwargs
        )

        # Load federated dataset
        self.fds = FederatedDataset(
            dataset=dataset_name, partitioners={"train": partitioner}
        )

        # Get transforms using factory
        self.train_transform, self.test_transform = get_transforms(
            dataset_name
        )

    def get_client_loaders(
        self, partition_id: int
    ) -> Tuple[DataLoader, DataLoader]:
        """Get train and test loaders for a specific client."""
        partition = self.fds.load_partition(partition_id, "train")

        # Apply transforms
        partition = partition.map(
            lambda x: {
                "image": self.train_transform(x["img"]),
                "label": x["label"],
            }
        )

        partition.set_format(type="torch", columns=["image", "label"])

        # Split into train and validation
        train_test_split = partition.train_test_split(test_size=0.2, seed=42)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        return train_loader, test_loader

    def get_centralized_test_loader(self) -> DataLoader:
        """Get centralized test set."""
        centralized_dataset = self.fds.load_split("test")

        centralized_dataset = centralized_dataset.map(
            lambda x: {
                "image": self.test_transform(x["img"]),
                "label": x["label"],
            }
        )

        centralized_dataset.set_format(
            type="torch", columns=["image", "label"]
        )

        return DataLoader(
            centralized_dataset, batch_size=self.batch_size, shuffle=False
        )


def list_partitioners() -> list:
    """Return list of available partitioner types."""
    return list(PARTITIONER_REGISTRY.keys())


def list_dataset_transforms() -> list:
    """Return list of datasets with registered transforms."""
    return list(TRANSFORM_REGISTRY.keys())
