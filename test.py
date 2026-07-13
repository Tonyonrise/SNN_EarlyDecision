import numpy as np
y = np.load("dataset/processed/y.npy")
print("unique y:", np.unique(y))
print("min/max:", y.min(), y.max())
print("count:", len(y))