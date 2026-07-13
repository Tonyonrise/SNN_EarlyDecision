import numpy as np
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


model = LSTMModel()
model.load_state_dict(torch.load("lstm_model.pth", map_location="cpu"))
model.eval()

softmax = nn.Softmax(dim=1)


def early_predict(sample, threshold=0.95):
    T = len(sample)

    for t in range(5, T + 1):
        x = torch.tensor(sample[:t], dtype=torch.float32).view(1, t, -1)

        with torch.no_grad():
            logits = model(x)
            prob = softmax(logits)
            conf, pred = torch.max(prob, dim=1)

        conf = conf.item()
        pred = pred.item()

        if conf > threshold:
            return t, pred, conf

    return T, pred, conf


# ======================
# run + SAVE LOG
# ======================
X = np.load("dataset/processed/X.npy")
y = np.load("dataset/processed/y.npy")

log_path = "early_log.txt"

with open(log_path, "w", encoding="utf-8") as f:
    f.write("index | true | pred | conf | stop\n")

    for i in range(len(X)):
        stop_t, pred, conf = early_predict(X[i])

        line = f"{i} | {y[i]} | {pred} | {conf:.3f} | {stop_t}"
        print(line)
        f.write(line + "\n")

print("\nSaved to:", log_path)