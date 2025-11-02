# datasets/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset


class FederatedDataset(ABC):
    """Base class for federated datasets."""

    @abstractmethod
    def load(self) -> Tuple[Dataset, Dataset]:
        """Load train and test datasets.

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        pass

    @abstractmethod
    def partition(
        self, train_data: Dataset, num_clients: int, config: Dict[str, Any]
    ) -> List[Dataset]:
        """Partition training data for clients.

        Args:
            train_data: Training dataset
            num_clients: Number of clients
            config: Partitioning configuration (e.g., {'method': 'iid',
                    'alpha': 0.5})

        Returns:
            List of partitioned datasets, one per client
        """
        pass


def iid_partition(
    dataset: Dataset, num_clients: int
) -> List[NDArray[np.long]]:
    """IID partitioning of dataset indices."""
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    return np.array_split(indices, num_clients)


def non_iid_dirichlet(
    labels: np.ndarray, num_clients: int, alpha: float
) -> List[NDArray[np.long]]:
    """Non-IID Dirichlet partitioning."""
    num_classes = len(np.unique(labels))
    label_distribution = np.random.dirichlet(
        [alpha] * num_clients, num_classes)

    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]

    for c_idx, c_list in enumerate(class_indices):
        np.random.shuffle(c_list)
        splits = (label_distribution[c_idx] * len(c_list)).astype(int)
        splits[-1] = len(c_list) - splits[:-1].sum()  # Adjust last split

        idx = 0
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(
                c_list[idx: idx + split].tolist()
            )
            idx += split

    return [np.array(indices) for indices in client_indices]
