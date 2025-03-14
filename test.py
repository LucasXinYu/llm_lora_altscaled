import os

results_dir = "/root/autodl-tmp/OpenFedLLM/save/gsm8k_20000_fedprox_c1s1_i10_b1a1_l512_r4a8_altTrue_20250314081905/"
#os.makedirs(results_dir, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load training loss data
training_loss = np.load("/root/autodl-tmp/OpenFedLLM/save/gsm8k_20000_fedprox_c1s1_i10_b1a1_l512_r4a8_altTrue_20250314081905/training_loss.npy")

# Replace -1 with NaN for better visualization
training_loss = np.where(training_loss == -1, np.nan, training_loss)
print(training_loss)
# Plot and save heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(training_loss, cmap="coolwarm", annot=False, linewidths=0.5)
plt.xlabel("Round")
plt.ylabel("Client")
plt.title("Training Loss Across Clients and Rounds")
plt.savefig(os.path.join(results_dir, "training_loss_heatmap.png"), dpi=300)
plt.close()


for client_id in range(training_loss.shape[0]):
    plt.figure(figsize=(8, 5))
    plt.plot(training_loss[client_id], marker="o", linestyle="-")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Trend for Client {client_id}")
    plt.grid()
    plt.savefig(os.path.join(results_dir, f"client_{client_id}_loss.png"), dpi=300)
    plt.close()
