from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


def get_cifar10(data_path="./data"):
    """Download and return CIFAR-10 dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = datasets.CIFAR10(
        data_path, train=True, download=True, transform=transform
    )
    testset = datasets.CIFAR10(
        data_path, train=False, download=True, transform=transform
    )

    return trainset, testset


def get_fashion_mnist(data_path="./data"):
    """Download and return Fashion-MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    trainset = datasets.FashionMNIST(
        data_path, train=True, download=True, transform=transform
    )
    testset = datasets.FashionMNIST(
        data_path, train=False, download=True, transform=transform
    )

    return trainset, testset


def partition_data(
    dataset, num_clients, iid=True, alpha=0.5
):
    """
    Partition dataset among clients

    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        iid: If True, IID partition; if False, non-IID partition
        alpha: Dirichlet alpha for non-IID (lower = more skewed)
    """
    num_samples = len(dataset)

    if iid:
        # IID partition
        samples_per_client = num_samples // num_clients
        indices = list(range(num_samples))
        np.random.shuffle(indices)

        client_indices = [
            indices[i * samples_per_client:(i + 1) * samples_per_client]
            for i in range(num_clients)
        ]
    else:
        # Non-IID partition using Dirichlet distribution
        labels = np.array([dataset[i][1] for i in range(num_samples)])
        num_classes = len(np.unique(labels))

        client_indices = [[] for _ in range(num_clients)]

        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)

            proportions = np.random.dirichlet(
                np.repeat(alpha, num_clients)
            )
            proportions = (
                np.cumsum(proportions) * len(idx_k)
            ).astype(int)[:-1]

            idx_k_split = np.split(idx_k, proportions)

            for i, idx in enumerate(idx_k_split):
                client_indices[i].extend(idx.tolist())

    return client_indices


def create_dataloaders(
    dataset_name, num_clients, batch_size=32, iid=True
):
    """Create dataloaders for all clients and test set"""
    if dataset_name == "cifar10":
        trainset, testset = get_cifar10()
    elif dataset_name == "fashion_mnist":
        trainset, testset = get_fashion_mnist()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Partition training data
    client_indices = partition_data(trainset, num_clients, iid=iid)

    # Create client dataloaders
    client_loaders = []
    for indices in client_indices:
        subset = Subset(trainset, indices)
        loader = DataLoader(
            subset, batch_size=batch_size, shuffle=True
        )
        client_loaders.append(loader)

    # Create test dataloader
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return client_loaders, test_loader
