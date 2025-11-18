import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def plot_single_experiment(log_file, save_dir="./plots"):
    """Plot results for a single experiment"""
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(log_file.replace(".json", ".csv"))

    experiment_name = os.path.basename(log_file).replace(".json", "")

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    axes[0].plot(df["round"], df["loss"], marker="o",
                 linewidth=2)
    axes[0].set_xlabel("Round", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training Loss", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    axes[1].plot(
        df["round"], df["accuracy"], marker="o",
        linewidth=2, color="green"
    )
    axes[1].set_xlabel("Round", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Test Accuracy", fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(experiment_name, fontsize=16, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{experiment_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {save_path}")


def plot_comparison(log_files, labels=None, save_dir="./plots"):
    """Compare multiple experiments"""
    os.makedirs(save_dir, exist_ok=True)

    if labels is None:
        labels = [
            os.path.basename(f).replace(".json", "")
            for f in log_files
        ]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Set color palette
    colors = sns.color_palette("husl", len(log_files))

    for i, (log_file, label) in enumerate(zip(log_files, labels)):
        df = pd.read_csv(log_file.replace(".json", ".csv"))

        # Plot loss
        axes[0].plot(
            df["round"], df["loss"], marker="o",
            label=label, linewidth=2, color=colors[i]
        )

        # Plot accuracy
        axes[1].plot(
            df["round"], df["accuracy"], marker="o",
            label=label, linewidth=2, color=colors[i]
        )

    # Configure loss plot
    axes[0].set_xlabel("Round", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training Loss Comparison", fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Configure accuracy plot
    axes[1].set_xlabel("Round", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Test Accuracy Comparison", fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Comparison plot saved to {save_path}")


def plot_malicious_impact(log_files, labels, save_dir="./plots"):
    """
    Plot comparison showing impact of malicious clients
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = sns.color_palette("husl", len(log_files))

    for i, (log_file, label) in enumerate(zip(log_files, labels)):
        df = pd.read_csv(log_file.replace(".json", ".csv"))

        # Loss
        axes[0, 0].plot(
            df["round"], df["loss"], marker="o",
            label=label, linewidth=2, color=colors[i],
            xticks=df["round"]
        )

        # Accuracy
        axes[0, 1].plot(
            df["round"], df["accuracy"], marker="o",
            label=label, linewidth=2, color=colors[i],
            xticks=df["round"]
        )

        # Loss delta (if more than one experiment)
        if i > 0:
            baseline_df = pd.read_csv(
                log_files[0].replace(".json", ".csv")
            )
            loss_delta = df["loss"] - baseline_df["loss"]
            axes[1, 0].plot(
                df["round"], loss_delta, marker="o",
                label=label, linewidth=2, color=colors[i],
                xticks=df["round"]
            )

            # Accuracy delta
            acc_delta = df["accuracy"] - baseline_df["accuracy"]
            axes[1, 1].plot(
                df["round"], acc_delta, marker="o",
                label=label, linewidth=2, color=colors[i],
                xticks=df["round"]
            )

    # Configure plots
    axes[0, 0].set_title("Loss", fontsize=14)
    axes[0, 0].set_ylabel("Loss", fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Accuracy", fontsize=14)
    axes[0, 1].set_ylabel("Accuracy", fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Loss Delta (vs Baseline)", fontsize=14)
    axes[1, 0].set_xlabel("Round", fontsize=12)
    axes[1, 0].set_ylabel("Loss Delta", fontsize=12)
    axes[1, 0].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title(
        "Accuracy Delta (vs Baseline)", fontsize=14
    )
    axes[1, 1].set_xlabel("Round", fontsize=12)
    axes[1, 1].set_ylabel("Accuracy Delta", fontsize=12)
    axes[1, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Impact of Malicious Clients", fontsize=16, y=1.0)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "malicious_impact.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Malicious impact plot saved to {save_path}")
