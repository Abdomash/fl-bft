import json
import os
from datetime import datetime
import pandas as pd


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
        }

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


def load_experiment_logs(log_file):
    """Load experiment logs from file"""
    with open(log_file, "r") as f:
        return json.load(f)


def load_all_experiments(log_dir="./logs"):
    """Load all experiments from log directory"""
    experiments = []

    for filename in os.listdir(log_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(log_dir, filename)
            experiments.append(load_experiment_logs(filepath))

    return experiments
