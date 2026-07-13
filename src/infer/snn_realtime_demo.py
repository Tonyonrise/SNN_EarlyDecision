import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from cvzone.HandTrackingModule import HandDetector

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.snn_model import SNNModel


WEIGHT_PATH = "weights/snn_model.pth"
DEVICE = "cpu"
CAMERA_ID = 0

MIN_FRAMES = 5
MAX_FRAMES = 20
CONF_TH = 0.90
STABLE_K = 3

CLASS_NAMES = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
    4: "wave",
}


def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model = SNNModel(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model_state_dict"])
        mean = float(checkpoint.get("mean", 0.0))
        std = float(checkpoint.get("std", 1.0))
    else:
        model = SNNModel(input_size=63, hidden_size=64, num_classes=5)
        model.load_state_dict(checkpoint)
        mean = 0.0
        std = 1.0

    model.to(DEVICE)
    model.eval()
    return model, mean, std


def normalize_sequence(sequence):
    seq = np.asarray(sequence, dtype=np.float32)
    return seq - seq[:, 0:1, :]


def predict_prefix(model, sequence, mean, std):
    seq = normalize_sequence(sequence)
    x = seq.reshape(seq.shape[0], -1)
    x = (x - mean) / (std + 1e-6)
    xb = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        logits = model(xb)
        prob = F.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)

    prob_np = prob.squeeze(0).cpu().numpy()
    return int(pred.item()), float(conf.item()), prob_np


def draw_panel(img, label, conf, frames, status, stable_count, fps, top_probs=None):
    color = (0, 200, 0) if status in {"LOCKED", "PAUSED"} else (0, 180, 255)

    h, w = img.shape[:2]
    overlay = img.copy()

    cv2.rectangle(overlay, (8, 8), (min(w - 8, 430), 78), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    cv2.putText(img, f"{label}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
    cv2.putText(img, f"{status}  conf {conf:.2f}", (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    cv2.putText(
        img,
        f"frames {frames}/{MAX_FRAMES}  stable {stable_count}/{STABLE_K}  fps {fps:.1f}",
        (20, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
    )

    bottom_y = h - 56
    cv2.rectangle(overlay, (8, bottom_y), (w - 8, h - 8), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    cv2.putText(
        img,
        "space: pause/start   q: quit",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (190, 190, 190),
        1,
    )

    if top_probs is not None:
        top_idx = np.argsort(top_probs)[::-1][:3]
        x = 20
        for row, idx in enumerate(top_idx):
            name = CLASS_NAMES.get(int(idx), str(int(idx)))
            text = f"{row + 1}. {name}: {top_probs[idx]:.3f}"
            cv2.putText(img, text, (x, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
            x += 150


def main():
    if not os.path.exists(WEIGHT_PATH):
        raise FileNotFoundError(f"Missing SNN checkpoint: {WEIGHT_PATH}")

    model, mean, std = load_model(WEIGHT_PATH)

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_ID}")

    detector = HandDetector(maxHands=1, detectionCon=0.7, minTrackCon=0.7)

    sequence = []
    locked = False
    locked_pred = None
    locked_conf = 0.0
    last_pred = None
    stable_count = 0
    running = False
    display_label = "waiting"
    display_conf = 0.0
    display_status = "PAUSED"
    display_probs = None
    last_time = time.time()
    fps = 0.0

    print("Realtime SNN demo started")
    print("Press SPACE when your hand is ready, then perform one gesture. Press SPACE again to pause/start.")

    while True:
        success, img = cap.read()
        if not success:
            break

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
        last_time = now

        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img)

        if not running:
            display_status = "PAUSED"
        elif hands and not locked:
            lm_list = np.array(hands[0]["lmList"], dtype=np.float32)
            sequence.append(lm_list)
            display_status = "COLLECTING"

            if len(sequence) >= MIN_FRAMES:
                pred, conf, probs = predict_prefix(model, sequence, mean, std)
                display_label = CLASS_NAMES.get(pred, str(pred))
                display_conf = conf
                display_probs = probs

                stable_count = stable_count + 1 if pred == last_pred else 1
                last_pred = pred

                if (conf >= CONF_TH and stable_count >= STABLE_K) or len(sequence) >= MAX_FRAMES:
                    locked = True
                    locked_pred = pred
                    locked_conf = conf
                    display_status = "LOCKED"
                    display_label = CLASS_NAMES.get(pred, str(pred))
                    display_conf = conf
            else:
                display_label = "collecting"
                display_conf = 0.0
                display_probs = None
                stable_count = 0
                last_pred = None

        elif locked:
            display_label = CLASS_NAMES.get(locked_pred, str(locked_pred))
            display_conf = locked_conf
            display_status = "LOCKED"
        else:
            display_label = "waiting"
            display_conf = 0.0
            display_probs = None
            display_status = "NO HAND"
            stable_count = 0
            last_pred = None

        draw_panel(
            img,
            display_label,
            display_conf,
            len(sequence),
            display_status,
            stable_count,
            fps,
            display_probs,
        )

        cv2.imshow("Realtime SNN Early Decision", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord(" "):
            if running:
                running = False
                display_status = "PAUSED"
            else:
                running = True
                sequence = []
                locked = False
                locked_pred = None
                locked_conf = 0.0
                last_pred = None
                stable_count = 0
                display_label = "waiting"
                display_conf = 0.0
                display_probs = None
                display_status = "COLLECTING"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
