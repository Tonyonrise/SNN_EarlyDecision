import os
import numpy as np
import matplotlib.pyplot as plt

# ======================
# create output folder
# ======================
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ======================
# load log safely
# ======================
data = []

with open("early_log.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        if "index" in line and "true" in line:
            continue

        if "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]

        if len(parts) < 5:
            continue

        try:
            true = int(parts[1])
            pred = int(parts[2])
            conf = float(parts[3])
            stop = int(parts[4])
        except:
            continue

        data.append((true, pred, conf, stop))

data = np.array(data)

y_true = data[:, 0].astype(int)
y_pred = data[:, 1].astype(int)
conf = data[:, 2].astype(float)
stop = data[:, 3].astype(int)

print("Total samples:", len(data))


# ======================
# 1. accuracy vs stop frame
# ======================
max_t = int(np.max(stop))
acc_curve = []

for t in range(5, max_t + 1):
    mask = stop <= t
    acc_curve.append(np.mean(y_true[mask] == y_pred[mask]) if np.sum(mask) > 0 else 0)

plt.figure()
plt.plot(range(5, max_t + 1), acc_curve)
plt.xlabel("Stop Frame Threshold")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Early Stop Time")
plt.grid()

plt.savefig(os.path.join(FIG_DIR, "accuracy_vs_stop_frame.png"), dpi=300, bbox_inches="tight")
plt.close()


# ======================
# 2. per-class stop frame
# ======================
classes = np.unique(y_true)
mean_stop = []

for c in classes:
    mask = y_true == c
    mean_stop.append(np.mean(stop[mask]) if np.sum(mask) > 0 else 0)

plt.figure()
plt.bar(classes, mean_stop)
plt.xlabel("Class")
plt.ylabel("Mean Stop Frame")
plt.title("Per-Class Early Stop Time")
plt.grid()

plt.savefig(os.path.join(FIG_DIR, "per_class_stop_frame.png"), dpi=300, bbox_inches="tight")
plt.close()


# ======================
# 3. confidence distribution
# ======================
plt.figure()
plt.hist(conf, bins=30)
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.title("Confidence Distribution")
plt.grid()

plt.savefig(os.path.join(FIG_DIR, "confidence_distribution.png"), dpi=300, bbox_inches="tight")
plt.close()


# ======================
# summary
# ======================
print("\n===== SUMMARY =====")
print("Accuracy:", np.mean(y_true == y_pred))
print("Avg stop frame:", np.mean(stop))
print("Saved figures to:", FIG_DIR)