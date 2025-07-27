import matplotlib.pyplot as plt
import pandas as pd

# Path to your log file
log_path = "loss_log.csv"

# Load the CSV
df = pd.read_csv(log_path)

# Convert loss columns to float (just in case)
df['train_loss'] = df['train_loss'].astype(float)
df['val_loss'] = df['val_loss'].astype(float)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='x')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
