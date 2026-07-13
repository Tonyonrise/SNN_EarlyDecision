import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNModel(nn.Module):
    def __init__(
        self,
        input_size=63,
        hidden_size=128,
        num_classes=5,
        beta=0.9,
        decay=0.9,
    ):
        super().__init__()

        self.decay = decay
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_all=False):
        """
        x: (B, T, 63)
        """
        mem = self.lif1.init_leaky()
        logits_accum = None
        logits_steps = []

        for t in range(x.shape[1]):
            cur = self.fc1(x[:, t, :])
            spk, mem = self.lif1(cur, mem)
            logits = self.fc2(mem)
            logits_accum = logits if logits_accum is None else logits_accum * self.decay + logits
            logits_steps.append(logits_accum)

        if return_all:
            return torch.stack(logits_steps, dim=1)

        return logits_accum
