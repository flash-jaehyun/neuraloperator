"""Small real neuralop FNO with exact Dirichlet boundary lifting."""

import torch
from torch import nn
from neuralop.models import FNO


class TransportFNO(nn.Module):
    def __init__(self, width=24, modes=12, layers=3):
        super().__init__()
        self.fno = FNO(
            n_modes=(modes,),
            in_channels=13,
            out_channels=2,
            hidden_channels=width,
            n_layers=layers,
            positional_embedding=None,
            domain_padding=0.25,
            use_channel_mlp=False,
            lifting_channel_ratio=2,
            projection_channel_ratio=2,
        )

    def forward(self, fields):
        x = fields[:, -1:]
        left, right = fields[:, 8:10], fields[:, 10:12]
        return left * (1 - x) + right * x + 4 * x * (1 - x) * self.fno(fields)


def torch_physics(c, fields):
    """Same discretization as the reference, fully differentiable."""
    h = 1 / (c.shape[-1] - 1)
    D, v, k, source = (
        0.1 * fields[:, :2].exp(),
        0.2 * fields[:, 2:4],
        0.8 * fields[:, 4:6],
        0.2 * fields[:, 6:8],
    )
    Df = 2 * D[..., :-1] * D[..., 1:] / (D[..., :-1] + D[..., 1:])
    vf = 0.5 * (v[..., :-1] + v[..., 1:])
    J = (
        -Df * torch.diff(c, dim=-1) / h
        + vf.clamp(min=0) * c[..., :-1]
        + vf.clamp(max=0) * c[..., 1:]
    )
    r = torch.diff(J, dim=-1) / h + k[..., 1:-1] * c[..., 1:-1] - source[..., 1:-1]
    balance = (
        J[..., -1]
        - J[..., 0]
        + h * (k[..., 1:-1] * c[..., 1:-1] - source[..., 1:-1]).sum(-1)
    )
    return r, balance
