from typing import cast

import torch
import torch.nn as nn


def squash(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    norm = x.norm(dim=dim, keepdim=True)
    return cast(torch.Tensor, (x**2 / (1 + norm**2)) * (x / (norm + 1e-8)))


class PrimaryCaps(nn.Module):
    def __init__(self, in_ch: int, out_caps: int, cap_dim: int, kernel: int = 9, stride: int = 2):
        super().__init__()
        self.out_caps = out_caps
        self.cap_dim = cap_dim
        self.conv = nn.Conv2d(in_ch, out_caps * cap_dim, kernel, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        out = self.conv(x)
        out = out.view(b, self.out_caps, self.cap_dim, -1).permute(0, 1, 3, 2)
        return squash(out, dim=-1)


class DigitCaps(nn.Module):
    def __init__(self, in_caps: int, in_dim: int, out_caps: int, out_dim: int, n_routes: int = 3):
        super().__init__()
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.n_routes = n_routes
        self.W = nn.Parameter(torch.randn(in_caps, out_caps, out_dim, in_dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n_in, in_dim = x.shape
        u_hat = torch.einsum("niod,bnd->bnio", self.W, x)
        b_ij = torch.zeros(b, n_in, self.out_caps, 1, device=x.device)
        for _ in range(self.n_routes):
            c_ij = b_ij.softmax(dim=2)
            s_j = (c_ij * u_hat).sum(dim=1)
            v_j = squash(s_j, dim=-1)
            if _ < self.n_routes - 1:
                b_ij = b_ij + (u_hat * v_j.unsqueeze(1)).sum(dim=-1, keepdim=True)
        return v_j


class CapsNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        n_classes: int = 10,
        primary_caps: int = 32,
        cap_dim: int = 8,
        digit_dim: int = 16,
        primary_spatial: int = 36,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 256, 9)
        self.primary = PrimaryCaps(256, primary_caps, cap_dim)
        in_caps = primary_caps * primary_spatial
        self.digit = DigitCaps(in_caps, cap_dim, n_classes, digit_dim)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        caps = self.primary(h)
        caps = caps.reshape(caps.size(0), -1, caps.size(-1))
        out = self.digit(caps)
        return cast(torch.Tensor, out.norm(dim=-1))
