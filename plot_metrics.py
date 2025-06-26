import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv("metrics.csv")  # Change path if needed

# Plotting
plt.figure(figsize=(10, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(df["round"], df["accuracy"], marker='o', color='green')
plt.title("Accuracy per Round")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(df["round"], df["loss"], marker='o', color='red')
plt.title("Loss per Round")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.grid(True)

plt.tight_layout()
plt.show()
