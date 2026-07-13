import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.snn_model import SNNModel


DATA_X = "dataset/processed/X.npy"
DATA_Y = "dataset/processed/y.npy"
WEIGHT_PATH = "weights/snn_model.pth"
LOG_PATH = "results/early_snn_log.txt"
MIN_FRAMES = 5
CONF_TH = 0.90
STABLE_K = 3
DEVICE = "cpu"


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model = SNNModel(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model_state_dict"])
        mean = checkpoint.get("mean", 0.0)
        std = checkpoint.get("std", 1.0)
    else:
        model = SNNModel(input_size=63, hidden_size=64, num_classes=5)
        model.load_state_dict(checkpoint)
        mean = 0.0
        std = 1.0

    model.to(DEVICE)
    model.eval()
    return model, float(mean), float(std)


def early_predict(model, sample, mean, std):
    x = sample.reshape(sample.shape[0], -1).astype(np.float32)
    x = (x - mean) / (std + 1e-6)

    stable_count = 0
    last_pred = None
    last_conf = 0.0
    last_prob = None

    with torch.no_grad():
        for t in range(MIN_FRAMES, x.shape[0] + 1):
            xb = torch.tensor(x[:t], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits = model(xb)
            prob = F.softmax(logits, dim=1)
            conf, pred = prob.max(dim=1)

            pred = int(pred.item())
            conf = float(conf.item())
            last_conf = conf
            last_prob = prob

            stable_count = stable_count + 1 if pred == last_pred else 1
            last_pred = pred

            if conf >= CONF_TH and stable_count >= STABLE_K:
                return t, pred, conf

    return x.shape[0], int(last_pred), last_conf


def main():
    model, mean, std = load_checkpoint(WEIGHT_PATH)
    X = np.load(DATA_X)
    y = np.load(DATA_Y)

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    results = []

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("idx true pred conf frames correct\n")

        for i in range(len(X)):
            stop_t, pred, conf = early_predict(model, X[i], mean, std)
            correct = int(pred == int(y[i]))
            results.append((int(y[i]), pred, conf, stop_t, correct))

            line = f"{i} {int(y[i])} {pred} {conf:.6f} {stop_t} {correct}"
            print(line)
            f.write(line + "\n")

        acc = sum(r[-1] for r in results) / len(results)
        avg_stop = sum(r[3] for r in results) / len(results)
        summary = f"\nEARLY ACC: {acc:.4f}\nAVG STOP FRAME: {avg_stop:.2f}\n"
        print(summary)
        f.write(summary)

    print("Saved to:", LOG_PATH)


if __name__ == "__main__":
    main()
