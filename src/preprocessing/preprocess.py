import os
import numpy as np

label_map = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "wave": 4
}

def fix_length(seq, T=20):
    L = seq.shape[0]

    if L > T:
        seq = seq[:T]
    else:
        pad = np.zeros((T - L, 21, 3))
        seq = np.concatenate([seq, pad], axis=0)

    return seq

def normalize(seq):
    return seq - seq[:, 0:1, :]

def build_dataset(raw_path="dataset/raw", out_path="dataset/processed", T=20):
    X, y = [], []

    for cls in os.listdir(raw_path):
        cls_path = os.path.join(raw_path, cls)

        if not os.path.isdir(cls_path):
            continue

        label = label_map[cls]

        for f in os.listdir(cls_path):
            if not f.endswith(".npy"):
                continue

            seq = np.load(os.path.join(cls_path, f))

            # 1. 归一化
            seq = normalize(seq)

            # 2. 固定长度
            seq = fix_length(seq, T)

            X.append(seq)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    os.makedirs(out_path, exist_ok=True)

    np.save(os.path.join(out_path, "X.npy"), X)
    np.save(os.path.join(out_path, "y.npy"), y)

    print("Done!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

if __name__ == "__main__":
    build_dataset()