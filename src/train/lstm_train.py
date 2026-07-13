import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# ======================
# 1. Load data
# ======================
X = np.load("dataset/processed/X.npy")  # (N, 40, 21, 3)
y = np.load("dataset/processed/y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# ======================
# 2. flatten (21,3) -> 63
# ======================
N, T, J, C = X.shape
X = X.reshape(N, T, J * C)

# ======================
# 3. train/test split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================
# 4. to tensor
# ======================
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# ======================
# 5. model
# ======================
class LSTMModel(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = LSTMModel()

# ======================
# 6. loss + optimizer
# ======================
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ======================
# 7. accuracy function
# ======================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            pred_cls = torch.argmax(pred, dim=1)

            correct += (pred_cls == yb).sum().item()
            total += yb.size(0)

    return correct / total

# ======================
# 8. training loop
# ======================
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_acc = evaluate(model, train_loader)
    test_acc = evaluate(model, test_loader)

    print(f"Epoch {epoch:02d} | "
          f"loss={total_loss:.4f} | "
          f"train_acc={train_acc:.4f} | "
          f"test_acc={test_acc:.4f}")
    torch.save(model.state_dict(), "lstm_model.pth")
    print("Model saved!")