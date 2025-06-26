import flwr as fl
import csv
from flwr.server.strategy import FedAvg

# Weighted average function for aggregated metrics
def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"accuracy": 0.0}

    weighted_acc = sum(num_examples * metric["accuracy"] for num_examples, metric in metrics) / total_examples
    return {"accuracy": weighted_acc}


# Custom strategy to log metrics into CSV
class SaveMetricsStrategy(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics_file = "metrics.csv"
        # Initialize CSV with header
        with open(self.metrics_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "loss", "accuracy"])

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_result = super().aggregate_evaluate(rnd, results, failures)
        if aggregated_result is not None:
            loss, metrics = aggregated_result  # Must unpack correctly
            accuracy = metrics.get("accuracy", 0.0)
            with open(self.metrics_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([rnd, loss, accuracy])
        return aggregated_result


# Start Flower server
def start_server():
    strategy = SaveMetricsStrategy(evaluate_metrics_aggregation_fn=weighted_average)

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    start_server()
