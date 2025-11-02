# analysis/plotter.py
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


class ExperimentAnalyzer:
    """Analyze and plot experiment results."""

    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)

    def load_experiment(self, exp_name: str) -> Dict:
        """Load experiment results."""
        # Find the most recent experiment with this name
        exp_dirs = sorted(
            self.results_dir.glob(f"{exp_name}_*"), reverse=True
        )
        if not exp_dirs:
            raise ValueError(f"No results found for experiment: {exp_name}")

        with open(exp_dirs[0] / "results.json", "r") as f:
            return json.load(f)

    def plot_accuracy_comparison(
        self, exp_names: List[str], save_path: str
    ):
        """Plot accuracy comparison across experiments."""
        plt.figure(figsize=(10, 6))

        for exp_name in exp_names:
            results = self.load_experiment(exp_name)
            metrics = results["metrics"]

            rounds = []
            accuracies = []
            for metric in metrics:
                if metric["phase"] == "evaluate" and "accuracy" in metric:
                    rounds.append(metric["round"])
                    accuracies.append(metric["accuracy"])

            plt.plot(rounds, accuracies, marker="o", label=exp_name)

        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.title("Test Accuracy Across Experiments")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(save_path, bbox_inches="tight", format="svg")

    def plot_loss_curves(
        self, exp_names: List[str], save_path
    ):
        """Plot loss curves for experiments."""
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        for exp_name in exp_names:
            results = self.load_experiment(exp_name)

            # Training loss
            train_metrics = [
                m for m in results["metrics"] if m["phase"] == "fit"
            ]
            if train_metrics:
                rounds = [m["round"] for m in train_metrics]
                losses = [m.get("train_loss", 0) for m in train_metrics]
                ax1.plot(rounds, losses, marker="o", label=exp_name)

            # Test loss
            test_metrics = [
                m for m in results["metrics"] if m["phase"] == "evaluate"
            ]
            if test_metrics:
                rounds = [m["round"] for m in test_metrics]
                losses = [m.get("loss", 0) for m in test_metrics]
                ax2.plot(rounds, losses, marker="o", label=exp_name)

        ax1.set_xlabel("Round")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Round")
        ax2.set_ylabel("Loss")
        ax2.set_title("Test Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", format="svg")

    def generate_summary_table(self, exp_names: List[str], save_path: str = None) -> str:
        """Generate summary table of final results."""
        summary = []
        summary.append("| Experiment | Final Accuracy | Final Loss |")
        summary.append("|-----------|----------------|------------|")

        for exp_name in exp_names:
            results = self.load_experiment(exp_name)
            metrics = results["metrics"]

            eval_metrics = [
                m for m in metrics if m["phase"] == "evaluate"
            ]
            if eval_metrics:
                last_metric = eval_metrics[-1]
                acc = last_metric.get("accuracy", 0) * 100
                loss = last_metric.get("loss", 0)
                summary.append(
                    f"| {exp_name} | {acc:.2f}% | {loss:.4f} |"
                )

        full_summary = "\n".join(summary)
        if save_path:
            with open(save_path, "w") as f:
                f.write(full_summary)
        return full_summary


if __name__ == "__main__":
    analyzer = ExperimentAnalyzer()

    # Compare experiments
    mnist_experiments = [
        "mnist_simplecnn_iid_fedavg",
        "mnist_simplecnn_noniid_alpha1_fedavg",
        "mnist_simplecnn_noniid_alpha01_fedavg",
    ]

    cifar_experiments = [
        "cifar10_simplecnn_iid_fedavg",
        "cifar10_simplecnn_noniid_fedavg",
        "cifar10_resnet_noniid_fedprox",
    ]

    # Plot MNIST results
    analyzer.plot_accuracy_comparison(
        mnist_experiments, save_path="mnist_accuracy.png"
    )
    analyzer.plot_loss_curves(
        mnist_experiments, save_path="mnist_loss.png"
    )

    # Plot CIFAR-10 results
    analyzer.plot_accuracy_comparison(
        cifar_experiments, save_path="cifar10_accuracy.png"
    )
    analyzer.plot_loss_curves(
        cifar_experiments, save_path="cifar10_loss.png"
    )

    # Print summary tables
    print("\nMNIST Experiments Summary:")
    print(analyzer.generate_summary_table(mnist_experiments))

    print("\nCIFAR-10 Experiments Summary:")
    print(analyzer.generate_summary_table(cifar_experiments))
