import torch
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)
