from math import sqrt
from torch import nn
import torch

class Sine(nn.Module):
    def __init__(self, omega_0: float):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, input):
        if self.omega_0 == 1:
            return torch.sin(input)
        else:
            return torch.sin(input * self.omega_0)


def init_weights_(module: nn.Linear, omega_0: float, is_first: bool = True):
    assert isinstance(module, nn.Linear), module
    with torch.no_grad():
        mag = (
            1 / module.in_features
            if is_first else
            sqrt(6 / module.in_features) / omega_0
        )
        module.weight.uniform_(-mag, mag)
