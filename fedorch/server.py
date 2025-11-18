import torch
from typing import Dict, Tuple, Optional
import flwr as fl
from flwr.common import Scalar


def get_evaluate_fn(model, testloader, device):
    """Return an evaluation function for server-side evaluation"""

    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model on centralized test set"""

        # Set model parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {
            k: torch.tensor(v) for k, v in params_dict
        }
        model.load_state_dict(state_dict, strict=True)

        model.to(device)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in testloader:
                images, labels = batch
                images, labels = (
                    images.to(device),
                    labels.to(device)
                )

                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(testloader)

        print(
            f"Round {server_round} - "
            f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
        )

        return avg_loss, {"accuracy": accuracy}

    return evaluate
