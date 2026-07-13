import os
import numpy as np

RAW_PATH = "dataset/raw"
OUT_PATH = "dataset/processed"

TARGET_LEN = 40  # 统一长度（你可以改 40 / 60 / 80）

def uniform_sample(seq, T):
    L = len(seq)

    if L == T:
        return seq

    if L < T:
        # padding（用最后一帧补齐）
        pad = np.repeat(seq[-1:], T - L, axis=0)
        return np.concatenate([seq, pad], axis=0)

    # uniform sampling
    idx = np.linspace(0, L - 1, T).astype(int)
    return seq[idx]


def normalize(seq):
    # 以手腕为原点（关键优化）
    return seq - seq[:, 0:1, :]


def build():
    X, y = [], []

    label_map = {}
    classes = sorted(os.listdir(RAW_PATH))

    for i, cls in enumerate(classes):
        label_map[cls] = i

        cls_path = os.path.join(RAW_PATH, cls)

        if not os.path.isdir(cls_path):
            continue

        for f in os.listdir(cls_path):
            if not f.endswith(".npy"):
                continue

            seq = np.load(os.path.join(cls_path, f))

            # ❌ filter too short
            if len(seq) < 20:
                continue

            # ✔ normalize
            seq = normalize(seq)

            # ✔ temporal alignment
            seq = uniform_sample(seq, TARGET_LEN)

            X.append(seq)
            y.append(i)

    X = np.array(X)
    y = np.array(y)

    os.makedirs(OUT_PATH, exist_ok=True)

    np.save(os.path.join(OUT_PATH, "X.npy"), X)
    np.save(os.path.join(OUT_PATH, "y.npy"), y)

    print("Done")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("label map:", label_map)


if __name__ == "__main__":
    build()
