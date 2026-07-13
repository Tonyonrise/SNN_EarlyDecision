import os

import matplotlib.pyplot as plt
import numpy as np


LOG_PATH = "results/early_snn_log.txt"
FIG_DIR = "figures/snn"
CLASS_NAMES = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
    4: "wave",
}


def load_snn_log(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("idx") or line.startswith("EARLY"):
                continue
            if line.startswith("AVG"):
                continue

            parts = line.split()
            if len(parts) != 6:
                continue

            try:
                idx, true, pred, conf, frames, correct = parts
                rows.append(
                    (
                        int(idx),
                        int(true),
                        int(pred),
                        float(conf),
                        int(frames),
                        int(correct),
                    )
                )
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No valid rows found in {path}")

    return np.array(rows, dtype=float)


def class_labels(classes):
    return [CLASS_NAMES.get(int(c), str(int(c))) for c in classes]


def save_accuracy_stop_curve(y_true, y_pred, stop):
    min_t = int(stop.min())
    max_t = int(stop.max())
    xs = np.arange(min_t, max_t + 1)
    coverage = []
    acc = []

    for t in xs:
        mask = stop <= t
        coverage.append(mask.mean())
        acc.append((y_true[mask] == y_pred[mask]).mean() if mask.any() else np.nan)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(xs, acc, marker="o", label="Accuracy")
    ax1.set_xlabel("Stop frame threshold")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, coverage, marker="s", color="#d55e00", label="Coverage")
    ax2.set_ylabel("Coverage")
    ax2.set_ylim(0, 1.05)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="lower right")
    plt.title("SNN Early Decision: Accuracy and Coverage")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "accuracy_coverage_vs_stop_frame.png"), dpi=300)
    plt.close()


def save_stop_distribution(stop, correct):
    plt.figure(figsize=(8, 5))
    bins = np.arange(int(stop.min()), int(stop.max()) + 2) - 0.5
    plt.hist(stop[correct == 1], bins=bins, alpha=0.75, label="Correct")
    plt.hist(stop[correct == 0], bins=bins, alpha=0.75, label="Wrong")
    plt.xlabel("Stop frame")
    plt.ylabel("Sample count")
    plt.title("SNN Stop Frame Distribution")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "stop_frame_distribution.png"), dpi=300)
    plt.close()


def save_per_class_bars(y_true, correct, stop):
    classes = np.array(sorted(np.unique(y_true).astype(int)))
    labels = class_labels(classes)
    acc = []
    mean_stop = []

    for c in classes:
        mask = y_true == c
        acc.append(correct[mask].mean())
        mean_stop.append(stop[mask].mean())

    x = np.arange(len(classes))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x - width / 2, acc, width, label="Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, mean_stop, width, color="#009e73", label="Mean stop")
    ax2.set_ylabel("Mean stop frame")

    lines = ax1.patches[:1] + ax2.patches[:1]
    ax1.legend(lines, ["Accuracy", "Mean stop"], loc="lower right")
    plt.title("SNN Per-Class Accuracy and Stop Time")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "per_class_accuracy_stop.png"), dpi=300)
    plt.close()


def save_confidence_distribution(conf, correct):
    plt.figure(figsize=(8, 5))
    plt.hist(conf[correct == 1], bins=20, alpha=0.75, label="Correct")
    plt.hist(conf[correct == 0], bins=20, alpha=0.75, label="Wrong")
    plt.xlabel("Confidence")
    plt.ylabel("Sample count")
    plt.title("SNN Confidence Distribution")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confidence_distribution.png"), dpi=300)
    plt.close()


def save_confusion_matrix(y_true, y_pred):
    classes = np.array(sorted(np.unique(np.concatenate([y_true, y_pred])).astype(int)))
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for true, pred in zip(y_true, y_pred):
        cm[class_to_idx[int(true)], class_to_idx[int(pred)]] += 1

    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="Recall-normalized count")
    plt.xticks(np.arange(len(classes)), class_labels(classes), rotation=30, ha="right")
    plt.yticks(np.arange(len(classes)), class_labels(classes))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("SNN Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data = load_snn_log(LOG_PATH)

    y_true = data[:, 1].astype(int)
    y_pred = data[:, 2].astype(int)
    conf = data[:, 3]
    stop = data[:, 4].astype(int)
    correct = data[:, 5].astype(int)

    save_accuracy_stop_curve(y_true, y_pred, stop)
    save_stop_distribution(stop, correct)
    save_per_class_bars(y_true, correct, stop)
    save_confidence_distribution(conf, correct)
    save_confusion_matrix(y_true, y_pred)

    print("===== SNN SUMMARY =====")
    print(f"Samples: {len(data)}")
    print(f"Accuracy: {correct.mean():.4f}")
    print(f"Avg stop frame: {stop.mean():.2f}")
    print(f"Median stop frame: {np.median(stop):.1f}")
    print(f"Avg confidence: {conf.mean():.4f}")
    print("Saved figures to:", FIG_DIR)


if __name__ == "__main__":
    main()
