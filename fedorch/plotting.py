# plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import List, Optional
from .utils import get_experiment_summary


def detect_varying_parameter(configs: List[dict]) -> Optional[str]:
    """
    Automatically detect which parameter is varying across experiments

    Returns: parameter name that varies, or None if multiple/no params vary
    """
    if len(configs) <= 1:
        return None

    # Parameters to check
    param_keys = [
        'num_clients',
        'malicious_ratio',
        'strategy',
        'num_rounds',
        'local_epochs',
        'learning_rate',
        'batch_size',
        'iid',
    ]

    varying_params = []

    for key in param_keys:
        values = [config.get(key) for config in configs]
        unique_values = set(values)
        if len(unique_values) > 1:
            varying_params.append(key)

    # Return primary varying parameter
    if len(varying_params) == 1:
        return varying_params[0]
    elif len(varying_params) > 1:
        # If multiple parameters vary, prioritize certain ones
        priority = ['malicious_ratio', 'num_clients', 'strategy', 'num_rounds']
        for param in priority:
            if param in varying_params:
                return param

    return None


def plot_single_experiment(
    log_file, save_dir="./plots", show_summary=True
):
    """Plot results for a single experiment"""
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(log_file.replace(".json", ".csv"))

    experiment_name = os.path.basename(log_file).replace(".json", "")

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    axes[0].plot(df["round"], df["loss"], marker="o", linewidth=2)
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

    # Add summary text
    if show_summary:
        final_acc = df["accuracy"].iloc[-1]
        best_acc = df["accuracy"].max()
        final_loss = df["loss"].iloc[-1]

        summary_text = (
            f"Final Acc: {final_acc:.3f}\n"
            f"Best Acc: {best_acc:.3f}\n"
            f"Final Loss: {final_loss:.3f}"
        )

        axes[1].text(
            0.02, 0.98, summary_text,
            transform=axes[1].transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10
        )

    plt.suptitle(experiment_name, fontsize=16, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{experiment_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {save_path}")


def plot_comparison(
    log_files: List[str],
    labels: Optional[List[str]] = None,
    save_dir: str = "./plots",
    title: str = "Comparison"
):
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
    axes[0].set_title("Training Loss", fontsize=14)
    axes[0].legend(fontsize=9, loc='best')
    axes[0].grid(True, alpha=0.3)

    # Configure accuracy plot
    axes[1].set_xlabel("Round", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Test Accuracy", fontsize=14)
    axes[1].legend(fontsize=9, loc='best')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Comparison plot saved to {save_path}")
    return save_path


def plot_malicious_ratio_study(
    log_files: List[str],
    configs: List[dict],
    save_dir: str = "./plots"
):
    """Specialized plot for malicious ratio comparison"""
    os.makedirs(save_dir, exist_ok=True)

    # Extract malicious ratios and sort
    data = []
    for log_file, config in zip(log_files, configs):
        df = pd.read_csv(log_file.replace(".json", ".csv"))
        malicious_ratio = config.get('malicious_ratio', 0)
        data.append({
            'ratio': malicious_ratio,
            'df': df,
            'log_file': log_file
        })

    data.sort(key=lambda x: x['ratio'])

    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    colors = sns.color_palette("RdYlGn_r", len(data))

    # 1. Loss over rounds
    ax1 = fig.add_subplot(gs[0, 0])
    for i, item in enumerate(data):
        label = f"{item['ratio']*100:.0f}% malicious"
        ax1.plot(
            item['df']['round'], item['df']['loss'],
            marker='o', label=label, linewidth=2, color=colors[i]
        )
    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss vs Malicious Ratio", fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy over rounds
    ax2 = fig.add_subplot(gs[0, 1])
    for i, item in enumerate(data):
        label = f"{item['ratio']*100:.0f}% malicious"
        ax2.plot(
            item['df']['round'], item['df']['accuracy'],
            marker='o', label=label, linewidth=2, color=colors[i]
        )
    ax2.set_xlabel("Round", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy vs Malicious Ratio", fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Final metrics bar chart
    ax3 = fig.add_subplot(gs[1, 0])
    ratios = [item['ratio'] * 100 for item in data]
    final_losses = [item['df']['loss'].iloc[-1] for item in data]
    ax3.bar(ratios, final_losses, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xlabel("Malicious Ratio (%)", fontsize=12)
    ax3.set_ylabel("Final Loss", fontsize=12)
    ax3.set_title("Final Loss by Malicious Ratio", fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')

    ax4 = fig.add_subplot(gs[1, 1])
    final_accs = [item['df']['accuracy'].iloc[-1] for item in data]
    ax4.bar(ratios, final_accs, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xlabel("Malicious Ratio (%)", fontsize=12)
    ax4.set_ylabel("Final Accuracy", fontsize=12)
    ax4.set_title("Final Accuracy by Malicious Ratio", fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')

    # 4. Degradation metrics
    ax5 = fig.add_subplot(gs[2, :])
    if len(data) > 0:
        baseline_acc = data[0]['df']['accuracy'].iloc[-1]
        baseline_loss = data[0]['df']['loss'].iloc[-1]

        acc_degradation = [
            (baseline_acc - item['df']['accuracy'].iloc[-1]) * 100
            for item in data
        ]
        loss_increase = [
            ((item['df']['loss'].iloc[-1] - baseline_loss) / baseline_loss) * 100
            for item in data
        ]

        x = np.arange(len(ratios))
        width = 0.35

        ax5.bar(x - width/2, acc_degradation, width, label='Accuracy Drop (%)',
                color='red', alpha=0.7, edgecolor='black')
        ax5.bar(x + width/2, loss_increase, width, label='Loss Increase (%)',
                color='blue', alpha=0.7, edgecolor='black')

        ax5.set_xlabel("Malicious Ratio (%)", fontsize=12)
        ax5.set_ylabel("Degradation (%)", fontsize=12)
        ax5.set_title("Performance Degradation (vs 0% Malicious)", fontsize=14)
        ax5.set_xticks(x)
        ax5.set_xticklabels([f"{r:.0f}" for r in ratios])
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.axhline(y=0, color='black', linestyle='--', linewidth=1)

    plt.suptitle("Malicious Client Impact Analysis", fontsize=18, y=0.995)

    save_path = os.path.join(save_dir, "malicious_ratio_study.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Malicious ratio study plot saved to {save_path}")
    return save_path


def plot_client_scaling_study(
    log_files: List[str],
    configs: List[dict],
    save_dir: str = "./plots"
):
    """Specialized plot for client number comparison"""
    os.makedirs(save_dir, exist_ok=True)

    # Extract client numbers and sort
    data = []
    for log_file, config in zip(log_files, configs):
        df = pd.read_csv(log_file.replace(".json", ".csv"))
        num_clients = config.get('num_clients', 0)
        data.append({
            'num_clients': num_clients,
            'df': df,
            'log_file': log_file
        })

    data.sort(key=lambda x: x['num_clients'])

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    colors = sns.color_palette("viridis", len(data))

    # 1. Loss convergence
    ax1 = fig.add_subplot(gs[0, 0])
    for i, item in enumerate(data):
        label = f"{item['num_clients']} clients"
        ax1.plot(
            item['df']['round'], item['df']['loss'],
            marker='o', label=label, linewidth=2, color=colors[i]
        )
    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss Convergence", fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy progression
    ax2 = fig.add_subplot(gs[0, 1])
    for i, item in enumerate(data):
        label = f"{item['num_clients']} clients"
        ax2.plot(
            item['df']['round'], item['df']['accuracy'],
            marker='o', label=label, linewidth=2, color=colors[i]
        )
    ax2.set_xlabel("Round", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy Progression", fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Scaling efficiency
    ax3 = fig.add_subplot(gs[1, 0])
    client_counts = [item['num_clients'] for item in data]
    final_accs = [item['df']['accuracy'].iloc[-1] for item in data]

    ax3.plot(client_counts, final_accs, marker='o', linewidth=2,
             markersize=10, color='green')
    ax3.set_xlabel("Number of Clients", fontsize=12)
    ax3.set_ylabel("Final Accuracy", fontsize=12)
    ax3.set_title("Scaling: Final Accuracy", fontsize=14)
    ax3.grid(True, alpha=0.3)

    # 4. Convergence speed
    ax4 = fig.add_subplot(gs[1, 1])

    # Find round where accuracy reaches 90% of final value
    convergence_rounds = []
    for item in data:
        final_acc = item['df']['accuracy'].iloc[-1]
        target_acc = 0.9 * final_acc
        conv_round = item['df'][item['df']
                                ['accuracy'] >= target_acc]['round'].min()
        convergence_rounds.append(conv_round if not pd.isna(
            conv_round) else len(item['df']))

    ax4.bar(client_counts, convergence_rounds, color=colors,
            alpha=0.7, edgecolor='black')
    ax4.set_xlabel("Number of Clients", fontsize=12)
    ax4.set_ylabel("Rounds to 90% Final Accuracy", fontsize=12)
    ax4.set_title("Convergence Speed", fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle("Client Scaling Analysis", fontsize=18, y=0.995)

    save_path = os.path.join(save_dir, "client_scaling_study.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Client scaling study plot saved to {save_path}")
    return save_path


def plot_strategy_comparison(
    log_files: List[str],
    configs: List[dict],
    save_dir: str = "./plots"
):
    """Specialized plot for strategy comparison"""
    os.makedirs(save_dir, exist_ok=True)

    # Group by strategy
    data = []
    for log_file, config in zip(log_files, configs):
        df = pd.read_csv(log_file.replace(".json", ".csv"))
        strategy = config.get('strategy', 'unknown')
        data.append({
            'strategy': strategy,
            'df': df,
            'config': config
        })

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    colors = sns.color_palette("Set2", len(data))

    # 1. Loss comparison
    ax1 = fig.add_subplot(gs[0, 0])
    for i, item in enumerate(data):
        ax1.plot(
            item['df']['round'], item['df']['loss'],
            marker='o', label=item['strategy'].upper(),
            linewidth=2, color=colors[i]
        )
    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss Comparison", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy comparison
    ax2 = fig.add_subplot(gs[0, 1])
    for i, item in enumerate(data):
        ax2.plot(
            item['df']['round'], item['df']['accuracy'],
            marker='o', label=item['strategy'].upper(),
            linewidth=2, color=colors[i]
        )
    ax2.set_xlabel("Round", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy Comparison", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Final performance comparison
    ax3 = fig.add_subplot(gs[1, 0])
    strategies = [item['strategy'].upper() for item in data]
    final_accs = [item['df']['accuracy'].iloc[-1] for item in data]
    best_accs = [item['df']['accuracy'].max() for item in data]

    x = np.arange(len(strategies))
    width = 0.35

    ax3.bar(x - width/2, final_accs, width, label='Final Accuracy',
            color=colors, alpha=0.7, edgecolor='black')
    ax3.bar(x + width/2, best_accs, width, label='Best Accuracy',
            color=colors, alpha=0.4, edgecolor='black')

    ax3.set_ylabel("Accuracy", fontsize=12)
    ax3.set_title("Performance Summary", fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies, rotation=15)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Robustness metrics
    ax4 = fig.add_subplot(gs[1, 1])

    # Calculate variance in accuracy across rounds
    acc_std = [item['df']['accuracy'].std() for item in data]

    ax4.bar(strategies, acc_std, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel("Accuracy Std Dev", fontsize=12)
    ax4.set_title("Training Stability", fontsize=14)
    ax4.set_xticklabels(strategies, rotation=15)
    ax4.grid(True, alpha=0.3, axis='y')

    malicious_info = ""
    if len(data) > 0 and 'malicious_ratio' in data[0]['config']:
        malicious_ratio = data[0]['config']['malicious_ratio']
        malicious_info = f" (with {malicious_ratio *
                                   100:.0f}% malicious clients)"

    plt.suptitle(f"Strategy Comparison{malicious_info}", fontsize=18, y=0.995)

    save_path = os.path.join(save_dir, "strategy_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Strategy comparison plot saved to {save_path}")
    return save_path


def create_summary_report(
    log_files: List[str],
    configs: List[dict],
    save_dir: str = "./plots"
):
    """Create a summary table of all experiments"""
    os.makedirs(save_dir, exist_ok=True)

    # Collect summary data
    summary_data = []
    for log_file, config in zip(log_files, configs):
        summary = get_experiment_summary(log_file)
        summary_data.append({
            'Strategy': config.get('strategy', 'N/A'),
            'Clients': config.get('num_clients', 'N/A'),
            'Malicious %': f"{config.get('malicious_ratio', 0)*100:.0f}",
            'Rounds': config.get('num_rounds', 'N/A'),
            'Final Acc': f"{summary.get('final_accuracy', 0):.4f}",
            'Best Acc': f"{summary.get('best_accuracy', 0):.4f}",
            'Final Loss': f"{summary.get('final_loss', 0):.4f}",
        })

    df = pd.DataFrame(summary_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.5)))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.12] * len(df.columns)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title("Experiment Summary", fontsize=16, pad=20, weight='bold')

    save_path = os.path.join(save_dir, "experiment_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Also save as CSV
    csv_path = os.path.join(save_dir, "experiment_summary.csv")
    df.to_csv(csv_path, index=False)

    print(f"Summary report saved to {save_path}")
    print(f"Summary CSV saved to {csv_path}")

    return save_path


def plot_experiments_auto(
    log_files: List[str],
    configs: List[dict],
    save_dir: str = "./plots"
):
    """
    Automatically detect comparison type and create appropriate plots
    """
    if not log_files or not configs:
        print("No experiments to plot")
        return

    # Detect varying parameter
    varying_param = detect_varying_parameter(configs)

    print(f"\nDetected varying parameter: {varying_param}")
    print(f"Creating specialized visualizations...\n")

    # Create appropriate plots based on what's varying
    if varying_param == 'malicious_ratio':
        plot_malicious_ratio_study(log_files, configs, save_dir)
    elif varying_param == 'num_clients':
        plot_client_scaling_study(log_files, configs, save_dir)
    elif varying_param == 'strategy':
        plot_strategy_comparison(log_files, configs, save_dir)
    else:
        # Generic comparison for other parameters
        labels = []
        for config in configs:
            if varying_param:
                value = config.get(varying_param, 'N/A')
                labels.append(f"{varying_param}={value}")
            else:
                labels.append(f"Exp {configs.index(config) + 1}")

        plot_comparison(log_files, labels, save_dir,
                        title=f"Comparison: {varying_param or 'Multiple Parameters'}")

    # Always create summary report
    create_summary_report(log_files, configs, save_dir)

    # Plot individual experiments
    for log_file in log_files:
        plot_single_experiment(log_file, save_dir)
