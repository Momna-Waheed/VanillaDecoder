import csv
import os

# --- Existing loss history (fill from your log) ---
history = [
    (1, 2.0274, 1.5859),
    (2, 1.5588, 1.3771),
    (3, 1.4173, 1.2785),
    (4, 1.3274, 1.1803),
    (5, 1.2341, 1.1101),
    (6, 1.1823, 1.0782),
    (7, 1.1506, 1.0548),
    (8, 1.1263, 1.0362),
    (9, 1.1068, 1.0219),
    (10, 1.0897, 1.0087),
]

LOG_PATH = "loss_log.csv"
write_header = not os.path.exists(LOG_PATH)

with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss"])
    for epoch, train_loss, val_loss in history:
        writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}"])

print(f"✅ Wrote {len(history)} rows to {LOG_PATH}")
