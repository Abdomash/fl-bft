import json
import os
from datetime import datetime
import pandas as pd
import hashlib
from typing import Optional


class ExperimentLogger:
    """Logger for experiment metrics"""

    def __init__(self, experiment_name, log_dir="./logs"):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            log_dir,
            f"{experiment_name}_{timestamp}.json"
        )

        self.metrics = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "rounds": [],
            "config": {},
        }

    def set_config(self, config_dict: dict):
        """Store experiment configuration"""
        self.metrics["config"] = config_dict

    def log_round(self, round_num, metrics):
        """Log metrics for a specific round"""
        self.metrics["rounds"].append({
            "round": round_num,
            **metrics
        })

    def save(self):
        """Save logs to file"""
        with open(self.log_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

        # Also save as CSV for easy plotting
        csv_file = self.log_file.replace(".json", ".csv")
        df = pd.DataFrame(self.metrics["rounds"])
        df.to_csv(csv_file, index=False)

        print(f"Logs saved to {self.log_file}")
        return self.log_file


def compute_config_hash(config_dict: dict) -> str:
    """Compute hash of configuration for caching"""
    # Convert config to sorted JSON string for consistent hashing
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def find_existing_experiment(
    config_dict: dict, log_dir: str = "./logs"
) -> Optional[str]:
    """Find existing experiment with same configuration"""
    config_hash = compute_config_hash(config_dict)

    if not os.path.exists(log_dir):
        return None

    for filename in os.listdir(log_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(log_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    stored_config = data.get("config", {})
                    if compute_config_hash(stored_config) == config_hash:
                        return filepath
            except:
                continue

    return None


def load_experiment_logs(log_file):
    """Load experiment logs from file"""
    with open(log_file, "r") as f:
        return json.load(f)


def load_all_experiments(log_dir="./logs"):
    """Load all experiments from log directory"""
    experiments = []

    if not os.path.exists(log_dir):
        return experiments

    for filename in os.listdir(log_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(log_dir, filename)
            try:
                experiments.append(load_experiment_logs(filepath))
            except:
                continue

    return experiments


def get_experiment_summary(log_file: str) -> dict:
    """Get summary statistics from experiment"""
    data = load_experiment_logs(log_file)
    rounds = data.get("rounds", [])

    if not rounds:
        return {}

    df = pd.DataFrame(rounds)

    return {
        "final_loss": df["loss"].iloc[-1] if "loss" in df else None,
        "final_accuracy": df["accuracy"].iloc[-1] if "accuracy" in df else None,
        "best_accuracy": df["accuracy"].max() if "accuracy" in df else None,
        "mean_loss": df["loss"].mean() if "loss" in df else None,
        "convergence_round": df["accuracy"].idxmax() if "accuracy" in df else None,
    }
