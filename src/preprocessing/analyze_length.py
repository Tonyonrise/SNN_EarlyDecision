import os
import numpy as np

def analyze(path="dataset/raw"):
    lengths = []

    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)

        if not os.path.isdir(cls_path):
            continue

        for f in os.listdir(cls_path):
            if not f.endswith(".npy"):
                continue

            seq = np.load(os.path.join(cls_path, f))
            lengths.append(len(seq))

    lengths = np.array(lengths)

    print("Total samples:", len(lengths))
    print("Min length:", lengths.min())
    print("Max length:", lengths.max())
    print("Mean length:", lengths.mean())
    print("Std:", lengths.std())

    print("\nPercentiles:")
    print("50%:", np.percentile(lengths, 50))
    print("80%:", np.percentile(lengths, 80))
    print("90%:", np.percentile(lengths, 90))
    print("95%:", np.percentile(lengths, 95))

if __name__ == "__main__":
    analyze()