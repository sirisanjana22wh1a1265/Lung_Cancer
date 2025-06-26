# LungCancerFLProject

## Setup
1. Clone/unzip this folder to your PC.
2. Make sure `LungCancerFLData/` is alongside the scripts.
3. Install dependencies:
   pip install -r requirements.txt

## Usage

### 1. Start Server (Terminal 1)
python server.py

### 2. Start Clients (Terminal 2, 3, 4, ...)
python client_runner.py client1
python client_runner.py client2
python client_runner.py client3
python client_runner.py client4

Replace `client1` with `client2`, etc., for each client.

## What Happens
- Server orchestrates 3 FL rounds.
- Each client trains locally on its data.
- Global aggregation occurs between rounds.
- Client evaluation results will print in terminal logs.

## Next Steps
- Save the aggregated model inside `server.py` (add callback).
- Add Streamlit GUI to test predictions.
- Compare federated performance vs central training.

Enjoy your Federated Lung Cancer project! 🎉
