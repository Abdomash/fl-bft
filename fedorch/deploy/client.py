"""
This is a single-file implementation of a Federated Learning (FL) client using Flower (flwr)
and PyTorch. It combines the various model definitions and data loading functions needed into
a single file.

This is meant as a compliation of the original multi-file code in the repository for easier
copying to new machines and environments for running tests.

Note, while this code is mostly a copy, it has been modified to run actual Federated Learning
clients with an external server using gRPC.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
from collections import OrderedDict
from torch.utils.data import DataLoader, Subset
import numpy as np
from torchvision import datasets, transforms


# =============================================================================
# CONFIGURATION
# =============================================================================

SERVER_ADDRESS = "10.164.0.6:8080"
DATASET = "cifar10"
DATA_PATH = "./data"
BATCH_SIZE = 32
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.01
NUM_CLIENTS = 7


# =============================================================================
# MODEL DEFINITION
# =============================================================================

class CIFARModel(nn.Module):
    """CNN for CIFAR-10"""

    def __init__(self, num_classes=10):
        super(CIFARModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# =============================================================================
# DATA LOADING
# =============================================================================

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


def partition_data_iid(dataset, num_clients, client_id):
    """Get IID partition for a specific client"""
    num_samples = len(dataset)
    samples_per_client = num_samples // num_clients

    np.random.seed(42)  # Consistent partitioning across all clients
    indices = np.random.permutation(num_samples)

    start_idx = client_id * samples_per_client
    end_idx = start_idx + samples_per_client
    client_indices = indices[start_idx:end_idx]

    return Subset(dataset, client_indices)


# =============================================================================
# CLIENT
# =============================================================================

class FLClient(fl.client.NumPyClient):
    def __init__(self, cid, model, trainloader, device):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.device = device
        self.is_malicious = False

        print(f"Client {cid} initialized "
              f"({'MALICIOUS' if self.is_malicious else 'HONEST'})")

    def get_parameters(self, config):
        return [
            val.cpu().numpy() for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        epochs = config.get("local_epochs", LOCAL_EPOCHS)
        lr = config.get("learning_rate", LEARNING_RATE)
        malicious_clients_str = config.get("malicious_clients", "")
        if malicious_clients_str != "":
            mal_clients = list(int(x)
                               for x in malicious_clients_str.split(","))
            self.is_malicious = self.cid in mal_clients
        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9
        )
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Malicious behavior: flip gradient signs
        if self.is_malicious:
            parameters = [
                -val.cpu().numpy()
                for _, val in self.model.state_dict().items()
            ]
            return (
                parameters,
                len(self.trainloader.dataset),
                {"loss": avg_loss, "malicious": True}
            )

        return (
            self.get_parameters(config),
            len(self.trainloader.dataset),
            {"loss": avg_loss, "malicious": False}
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        self.model.to(self.device)
        self.model.eval()

        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.trainloader)

        return avg_loss, total, {"accuracy": accuracy}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FL Client")
    parser.add_argument(
        "--client-id", type=int, required=True,
        help="Client ID (0-6 for 7 clients)"
    )
    parser.add_argument(
        "--server", type=str, default=None,
        help="Server address (default from config)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use"
    )

    args = parser.parse_args()

    # Setup
    device = torch.device(
        args.device if torch.cuda.is_available()
        and args.device == "cuda" else "cpu"
    )
    server_address = args.server or SERVER_ADDRESS

    print(f"\n{'='*60}")
    print(f"Starting Client {args.client_id}")
    print(f"Server: {server_address}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading CIFAR-10 dataset...")
    trainset, _ = get_cifar10(DATA_PATH)

    # Get client's partition
    client_data = partition_data_iid(trainset, NUM_CLIENTS, args.client_id)
    trainloader = DataLoader(
        client_data, batch_size=BATCH_SIZE, shuffle=True
    )

    print(f"Client {args.client_id} has {len(client_data)} samples")

    # Create model
    model = CIFARModel()

    # Create client
    client = FLClient(
        cid=args.client_id,
        model=model,
        trainloader=trainloader,
        device=device
    )

    # Start client
    import time
    print(f"\nConnecting to server at {server_address}...")
    while True:
        try:
            fl.client.start_numpy_client(
                server_address=server_address,
                client=client
            )
        except KeyboardInterrupt:
            exit(1)
        except:
            print("FL Failed... Retrying")
            time.sleep(5)


if __name__ == "__main__":
    main()
