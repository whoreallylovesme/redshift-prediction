from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(s: torch.Tensor, dim: int | tuple[int, ...] = -1, eps: float = 1e-8) -> torch.Tensor:
    sq = (s * s).sum(dim=dim, keepdim=True)
    norm = torch.sqrt(sq + eps)
    return (sq / (1.0 + sq)) * (s / norm)

class Conv2DCaps(nn.Module):

    def __init__(
        self,
        in_caps: int,
        in_atoms: int,
        out_caps: int,
        out_atoms: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.in_caps = in_caps
        self.in_atoms = in_atoms
        self.out_caps = out_caps
        self.out_atoms = out_atoms
        pad = kernel_size // 2
        self.conv = nn.Conv2d(
            in_caps * in_atoms,
            out_caps * out_atoms,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h, w = x.shape[-2], x.shape[-1]
        x_flat = x.reshape(b, self.in_caps * self.in_atoms, h, w)
        out = self.conv(x_flat)
        h_out, w_out = out.shape[-2], out.shape[-1]
        out = out.reshape(b, self.out_caps, self.out_atoms, h_out, w_out)
        return squash(out, dim=2)

class ConvCapsuleLayer3D(nn.Module):

    def __init__(
        self,
        in_caps: int,
        in_atoms: int,
        out_caps: int,
        out_atoms: int,
        kernel_size: int = 3,
        stride: int = 1,
        routings: int = 3,
    ) -> None:
        super().__init__()
        self.in_caps = in_caps
        self.in_atoms = in_atoms
        self.out_caps = out_caps
        self.out_atoms = out_atoms
        self.kernel_size = kernel_size
        self.stride = stride
        self.routings = routings

        self.weight = nn.Parameter(
            torch.empty(out_caps * out_atoms, 1, in_atoms, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")

        self.bias = nn.Parameter(torch.full((out_caps, out_atoms, 1, 1), 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, in_caps, in_atoms, h, w = x.shape
        x_flat = x.reshape(b, 1, in_caps * in_atoms, h, w)
        pad = self.kernel_size // 2
        out = F.conv3d(
            x_flat,
            self.weight,
            stride=(in_atoms, self.stride, self.stride),
            padding=(0, pad, pad),
        )

        h_out, w_out = out.shape[-2], out.shape[-1]
        out = out.reshape(b, self.out_caps, self.out_atoms, in_caps, h_out, w_out)

        votes = out.permute(0, 3, 1, 2, 4, 5).contiguous()

        b_log = torch.zeros(b, in_caps, self.out_caps, h_out, w_out, device=x.device)
        v = votes.new_zeros(b, self.out_caps, self.out_atoms, h_out, w_out)
        for r in range(self.routings):

            b_flat = b_log.reshape(b, in_caps, -1)
            c_flat = F.softmax(b_flat, dim=-1)
            c = c_flat.reshape(b, in_caps, self.out_caps, h_out, w_out)

            preact = (c.unsqueeze(3) * votes).sum(dim=1) + self.bias

            v = squash(preact, dim=(2, 3, 4))

            if r < self.routings - 1:

                agreement = (votes * v.unsqueeze(1)).sum(dim=3)
                b_log = b_log + agreement
        return v

def _flatten_caps(x: torch.Tensor) -> torch.Tensor:
    b, num_caps, dim_caps, h, w = x.shape
    return x.permute(0, 3, 4, 1, 2).reshape(b, h * w * num_caps, dim_caps)

class CapsuleLayer(nn.Module):

    def __init__(
        self,
        in_num_caps: int,
        in_dim_caps: int,
        num_caps: int,
        dim_caps: int,
        routings: int = 3,
    ) -> None:
        super().__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.routings = routings

        # W/B: имена параметров зафиксированы ключами state_dict существующих
        # чекпоинтов (edfn_morph_v1 и др.) - переименование ломает загрузку.
        self.W = nn.Parameter(torch.empty(num_caps, in_num_caps, dim_caps, in_dim_caps))
        nn.init.xavier_uniform_(self.W)
        self.B = nn.Parameter(torch.empty(num_caps, dim_caps))
        nn.init.xavier_uniform_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        u_hat = torch.einsum("bik,jidk->bjid", x, self.W)
        b_log = torch.zeros(x.shape[0], self.num_caps, self.in_num_caps, device=x.device)
        v = u_hat.new_zeros(x.shape[0], self.num_caps, self.dim_caps)
        for r in range(self.routings):

            c = F.softmax(b_log, dim=1)

            outputs = torch.einsum("bji,bjid->bjd", c, u_hat) + self.B
            v = squash(outputs, dim=-1)
            if r < self.routings - 1:

                agreement = torch.einsum("bjd,bjid->bji", v, u_hat)
                b_log = b_log + agreement
        return v

def mask_by_class(caps: torch.Tensor, target_class: torch.Tensor | None) -> torch.Tensor:
    auto = caps.norm(dim=-1).argmax(dim=1)
    if target_class is None:
        idx = auto
    else:
        idx = torch.where(target_class >= 0, target_class, auto)
    batch_idx = torch.arange(caps.shape[0], device=caps.device)
    return caps[batch_idx, idx]

def z_to_logistic(z: torch.Tensor, z_min: float, z_max: float) -> torch.Tensor:
    eps = 1e-6
    return torch.log((z - z_min).clamp(min=eps) / (z_max - z).clamp(min=eps))

def logistic_to_z(y: torch.Tensor, z_min: float, z_max: float) -> torch.Tensor:
    e = torch.exp(y)
    return (e * z_max + z_min) / (e + 1.0)

def _init_mdn_bias(bias: torch.Tensor, components: int) -> None:
    """Стартовый bias MDN-головы: равномерные логит-медианы + узкая сигма."""
    with torch.no_grad():
        for j in range(components):
            q = (j + 1) / (components + 1)
            bias[j + components] = math.log(q / (1 - q))
        bias[2 * components :] = -0.45  # сигма ~0.1 на старте


class CapsNet(nn.Module):

    @staticmethod
    def _build_stem(in_channels: int) -> tuple[nn.Conv2d, nn.BatchNorm2d]:
        conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(conv1.weight, nonlinearity="relu")
        assert conv1.bias is not None
        nn.init.zeros_(conv1.bias)
        return conv1, nn.BatchNorm2d(128)

    @staticmethod
    def _build_caps_block(
        in_caps: int, in_atoms: int, out_atoms: int, routings: int | None = None,
    ) -> tuple[Conv2DCaps, Conv2DCaps | ConvCapsuleLayer3D, Conv2DCaps, Conv2DCaps]:
        """Residual-блок капсул: downsample/skip/две свёртки (skip=3D при routings)."""
        down = Conv2DCaps(in_caps, in_atoms, 32, out_atoms, kernel_size=3, stride=2)
        skip: Conv2DCaps | ConvCapsuleLayer3D
        if routings is None:
            skip = Conv2DCaps(32, out_atoms, 32, out_atoms, kernel_size=3, stride=1)
        else:
            skip = ConvCapsuleLayer3D(
                in_caps=32, in_atoms=out_atoms, out_caps=32, out_atoms=out_atoms,
                kernel_size=3, stride=1, routings=routings,
            )
        a = Conv2DCaps(32, out_atoms, 32, out_atoms, kernel_size=3, stride=1)
        b = Conv2DCaps(32, out_atoms, 32, out_atoms, kernel_size=3, stride=1)
        return down, skip, a, b

    @staticmethod
    def _build_decoder_dense(dim_capsule: int, flat: int) -> tuple[nn.Linear, nn.PReLU]:
        dense = nn.Linear(dim_capsule, flat)
        nn.init.kaiming_normal_(dense.weight, nonlinearity="relu")
        nn.init.zeros_(dense.bias)
        return dense, nn.PReLU()

    @staticmethod
    def _build_decoder_conv(in_channels: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev_ch = in_channels
        for ch in (64, 32, 16, 8):
            ct = nn.ConvTranspose2d(prev_ch, ch, kernel_size=3, padding=1)
            nn.init.kaiming_normal_(ct.weight, nonlinearity="relu")
            assert ct.bias is not None
            nn.init.zeros_(ct.bias)
            layers.append(ct)
            layers.append(nn.PReLU())
            prev_ch = ch
        ct_out = nn.ConvTranspose2d(prev_ch, in_channels, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(ct_out.weight, nonlinearity="relu")
        assert ct_out.bias is not None
        nn.init.zeros_(ct_out.bias)
        layers.append(ct_out)
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    @staticmethod
    def _build_regressor(dim_capsule: int, mdn_components: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = dim_capsule
        for h in (128, 64, 32, 16):
            lin = nn.Linear(prev, h)
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            layers.append(nn.PReLU())
            prev = h
        out_dim = 3 * mdn_components if mdn_components > 0 else 1
        out_lin = nn.Linear(prev, out_dim)
        nn.init.kaiming_normal_(out_lin.weight, nonlinearity="relu")
        nn.init.zeros_(out_lin.bias)
        if mdn_components > 0:
            _init_mdn_bias(out_lin.bias, mdn_components)
        layers.append(out_lin)
        return nn.Sequential(*layers)

    def __init__(
        self,
        in_channels: int = 5,
        input_size: int = 64,
        num_class: int = 2,
        dim_capsule: int = 16,
        routings: int = 3,
        mdn_components: int = 0,
    ) -> None:
        super().__init__()
        if input_size % 16 != 0:
            raise ValueError(f"input_size must be divisible by 16, got {input_size}")
        self.in_channels = in_channels
        self.input_size = input_size
        self.num_class = num_class
        self.dim_capsule = dim_capsule

        self.conv1, self.bn1 = self._build_stem(in_channels)

        self.b1_down, self.b1_skip, self.b1_a, self.b1_b = self._build_caps_block(128, 1, 4)
        self.b2_down, self.b2_skip, self.b2_a, self.b2_b = self._build_caps_block(32, 4, 8)
        self.b3_down, self.b3_skip, self.b3_a, self.b3_b = self._build_caps_block(32, 8, 8)
        self.b4_down, self.b4_skip, self.b4_a, self.b4_b = self._build_caps_block(
            32, 8, 8, routings=routings
        )

        l1_h = input_size // 8
        l2_h = input_size // 16
        total_caps = 32 * l1_h * l1_h + 32 * l2_h * l2_h
        self.digit_caps = CapsuleLayer(
            in_num_caps=total_caps, in_dim_caps=8,
            num_caps=num_class, dim_caps=dim_capsule, routings=routings,
        )

        flat = in_channels * input_size * input_size
        self.decoder_dense, self.decoder_dense_act = self._build_decoder_dense(
            dim_capsule, flat
        )
        self.decoder_conv = self._build_decoder_conv(in_channels)

        self.mdn_components = int(mdn_components)
        self.redshift = self._build_regressor(dim_capsule, self.mdn_components)

    def forward(
        self, x: torch.Tensor, target_class: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x.shape[0]

        h = F.relu(self.conv1(x))
        h = self.bn1(h)

        caps = h.unsqueeze(2)

        u = self.b1_down(caps)
        skip = self.b1_skip(u)
        u = self.b1_a(u)
        u = self.b1_b(u)
        u = u + skip

        u = self.b2_down(u)
        skip = self.b2_skip(u)
        u = self.b2_a(u)
        u = self.b2_b(u)
        u = u + skip

        u = self.b3_down(u)
        skip = self.b3_skip(u)
        u = self.b3_a(u)
        u = self.b3_b(u)
        u = u + skip
        l1 = u

        u = self.b4_down(u)
        skip = self.b4_skip(u)
        u = self.b4_a(u)
        u = self.b4_b(u)
        u = u + skip
        l2 = u

        flat = torch.cat([_flatten_caps(l2), _flatten_caps(l1)], dim=1)

        digit_caps = self.digit_caps(flat)
        probs = digit_caps.norm(dim=-1)

        active = mask_by_class(digit_caps, target_class)

        d = self.decoder_dense_act(self.decoder_dense(active))
        d = d.reshape(b, self.in_channels, self.input_size, self.input_size)
        recon = self.decoder_conv(d)

        z_out: torch.Tensor = self.redshift(active)
        if self.mdn_components > 0:
            return z_out, probs, recon
        return z_out.squeeze(-1), probs, recon


def margin_loss(
    probs: torch.Tensor,
    target_onehot: torch.Tensor,
    m_plus: float = 0.9,
    m_minus: float = 0.1,
    lam: float = 0.5,
) -> torch.Tensor:
    pos = target_onehot * (m_plus - probs).clamp(min=0.0).pow(2)
    neg = lam * (1.0 - target_onehot) * (probs - m_minus).clamp(min=0.0).pow(2)
    return (pos + neg).sum(dim=1).mean()


def mdn_decode(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """raw [B,3K] -> (w [B,K] softmax, mu [B,K] in (0,1), sigma [B,K] > 0)."""
    k = raw.shape[-1] // 3
    w = torch.softmax(raw[..., :k], dim=-1)
    mu = torch.sigmoid(raw[..., k : 2 * k])
    sigma = 0.005 + 0.245 * torch.sigmoid(raw[..., 2 * k :])
    return w, mu, sigma


def mdn_nll(raw: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """NLL истинного логистического y под гауссовой смесью."""
    w, mu, sigma = mdn_decode(raw)
    comp = -0.5 * ((y.unsqueeze(-1) - mu) / sigma) ** 2 - torch.log(sigma) - 0.918
    log_prob = torch.logsumexp(comp + torch.log(w.clamp_min(1e-8)), dim=-1)
    return -log_prob.mean()


def mdn_pz(raw: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Плотность p(z) на решётке z: [B, G]."""
    w, mu, sigma = mdn_decode(raw)
    g = grid.view(1, 1, -1)
    norm = sigma.unsqueeze(-1) * 2.5066
    gauss = torch.exp(-0.5 * ((g - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)) ** 2)
    return (w.unsqueeze(-1) * gauss / norm).sum(dim=1)
