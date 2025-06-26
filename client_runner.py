import flwr as fl
from fl_client import LungCancerClient
import sys

if __name__ == "__main__":
    client_id = sys.argv[1] if len(sys.argv) > 1 else "client1"
    client_path = f"LungCancerFLData1/{client_id}"
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=LungCancerClient(client_path).to_client(),
    )
