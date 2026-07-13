import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.snn_model import SNNModel


DATA_X = "dataset/processed/X.npy"
DATA_Y = "dataset/processed/y.npy"
SAVE_PATH = "weights/snn_model.pth"
SEED = 42
BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-3


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            pred = logits.argmax(dim=1)

            total_loss += loss.item() * yb.size(0)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
            all_true.extend(yb.cpu().numpy().tolist())
            all_pred.extend(pred.cpu().numpy().tolist())

    return total_loss / total, correct / total, np.array(all_true), np.array(all_pred)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X = np.load(DATA_X)
    y = np.load(DATA_Y)

    if X.ndim != 4:
        raise ValueError(f"Expected X shape (N, T, 21, 3), got {X.shape}")

    n, t, joints, coords = X.shape
    X = X.reshape(n, t, joints * coords).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    mean = X_train.mean()
    std = X_train.std() + 1e-6
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model_kwargs = {
        "input_size": X_train.shape[-1],
        "hidden_size": 128,
        "num_classes": int(y.max()) + 1,
        "beta": 0.9,
        "decay": 0.9,
    }
    model = SNNModel(**model_kwargs)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_acc = 0.0
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * yb.size(0)

        train_loss, train_acc, _, _ = evaluate(model, train_loader, loss_fn)
        test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, loss_fn)

        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "mean": float(mean),
                    "std": float(std),
                    "seq_len": int(t),
                    "test_acc": float(test_acc),
                    "epoch": epoch,
                },
                SAVE_PATH,
            )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={total_loss / len(X_train):.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"test_acc={test_acc:.4f} | "
            f"best={best_acc:.4f}"
        )

    print("\nConfusion matrix on last epoch:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nSaved best checkpoint to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
