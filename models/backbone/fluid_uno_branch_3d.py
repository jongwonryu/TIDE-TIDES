from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm_channels(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        g = _group_norm_channels(channels)
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.block(x))


class SpectralConv3D(nn.Module):
    """
    3D Fourier operator: FFT -> complex mode mixing -> iFFT.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_d: int,
        modes_h: int,
        modes_w: int,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_d = int(modes_d)
        self.modes_h = int(modes_h)
        self.modes_w = int(modes_w)

        scale = 1.0 / max(1.0, float(in_channels * out_channels))
        shape = (in_channels, out_channels, self.modes_d, self.modes_h, self.modes_w)
        self.w1 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w3 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w4 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    @staticmethod
    def _mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bidhw,iodhw->bodhw", a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, d, h, w = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        w_half = (w // 2) + 1

        m1 = min(self.modes_d, d)
        m2 = min(self.modes_h, h)
        m3 = min(self.modes_w, w_half)

        out_ft = torch.zeros(
            bsz, self.out_channels, d, h, w_half, dtype=torch.cfloat, device=x.device
        )
        if m1 <= 0 or m2 <= 0 or m3 <= 0:
            return torch.fft.irfftn(out_ft, s=(d, h, w))

        out_ft[:, :, :m1, :m2, :m3] = self._mul(
            x_ft[:, :, :m1, :m2, :m3], self.w1[:, :, :m1, :m2, :m3]
        )
        out_ft[:, :, -m1:, :m2, :m3] = self._mul(
            x_ft[:, :, -m1:, :m2, :m3], self.w2[:, :, :m1, :m2, :m3]
        )
        out_ft[:, :, :m1, -m2:, :m3] = self._mul(
            x_ft[:, :, :m1, -m2:, :m3], self.w3[:, :, :m1, :m2, :m3]
        )
        out_ft[:, :, -m1:, -m2:, :m3] = self._mul(
            x_ft[:, :, -m1:, -m2:, :m3], self.w4[:, :, :m1, :m2, :m3]
        )
        return torch.fft.irfftn(out_ft, s=(d, h, w))


class SpectralLocalResidualBlock3D(nn.Module):
    """
    Stage block: spectral operator + local conv residual.
    """

    def __init__(self, channels: int, modes: Tuple[int, int, int]) -> None:
        super().__init__()
        self.spectral = SpectralConv3D(channels, channels, *modes)
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.local = ResidualConvBlock3D(channels)
        g = _group_norm_channels(channels)
        self.norm = nn.GroupNorm(g, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.pointwise(x)
        y = self.norm(y)
        y = F.gelu(y)
        y = self.local(y)
        return x + y


class Downsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(_group_norm_channels(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_norm_channels(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class UNOOperator3D(nn.Module):
    """
    3-stage U-NO style operator.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_width: int = 16,
        blocks_per_stage: int = 2,
        stage_modes: Sequence[Tuple[int, int, int]] = ((6, 12, 12), (8, 16, 16), (10, 20, 20)),
    ) -> None:
        super().__init__()
        if len(stage_modes) != 3:
            raise ValueError("stage_modes must contain exactly 3 (D,H,W) tuples.")

        w0 = int(base_width)
        w1 = int(base_width * 2)
        w2 = int(base_width * 4)
        n = int(max(1, blocks_per_stage))

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, w0, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_norm_channels(w0), w0),
            nn.GELU(),
        )
        self.enc0 = nn.Sequential(
            *[SpectralLocalResidualBlock3D(w0, stage_modes[0]) for _ in range(n)]
        )
        self.down1 = Downsample3D(w0, w1)
        self.enc1 = nn.Sequential(
            *[SpectralLocalResidualBlock3D(w1, stage_modes[1]) for _ in range(n)]
        )
        self.down2 = Downsample3D(w1, w2)
        self.latent = nn.Sequential(
            *[SpectralLocalResidualBlock3D(w2, stage_modes[2]) for _ in range(n)]
        )
        self.up1 = Upsample3D(w2, w1, w1)
        self.dec1 = nn.Sequential(
            *[SpectralLocalResidualBlock3D(w1, stage_modes[1]) for _ in range(n)]
        )
        self.up2 = Upsample3D(w1, w0, w0)
        self.dec0 = nn.Sequential(
            *[SpectralLocalResidualBlock3D(w0, stage_modes[0]) for _ in range(n)]
        )
        self.head = nn.Conv3d(w0, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.enc0(self.stem(x))
        s1 = self.enc1(self.down1(s0))
        z = self.latent(self.down2(s1))
        y = self.dec1(self.up1(z, s1))
        y = self.dec0(self.up2(y, s0))
        return self.head(y)


class FluidUNOBranch3D(nn.Module):
    """
    Fluid branch with explicit encode/update separation.
    """

    def __init__(
        self,
        input_channels: int,
        latent_channels: int = 32,
        control_channels: int = 0,
        base_width: int = 16,
        blocks_per_stage: int = 2,
        stage_modes: Sequence[Tuple[int, int, int]] = ((6, 12, 12), (8, 16, 16), (10, 20, 20)),
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.control_channels = int(max(0, control_channels))
        self.encoder = UNOOperator3D(
            in_channels=int(input_channels),
            out_channels=self.latent_channels,
            base_width=base_width,
            blocks_per_stage=blocks_per_stage,
            stage_modes=stage_modes,
        )
        update_in = self.latent_channels + self.control_channels
        self.self_update_operator = UNOOperator3D(
            in_channels=update_in,
            out_channels=self.latent_channels,
            base_width=base_width,
            blocks_per_stage=blocks_per_stage,
            stage_modes=stage_modes,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def self_update(
        self,
        z_f: torch.Tensor,
        control_grid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if control_grid is None:
            if self.control_channels > 0:
                zeros = torch.zeros(
                    z_f.shape[0],
                    self.control_channels,
                    z_f.shape[2],
                    z_f.shape[3],
                    z_f.shape[4],
                    device=z_f.device,
                    dtype=z_f.dtype,
                )
                x = torch.cat([z_f, zeros], dim=1)
            else:
                x = z_f
        else:
            if control_grid.shape[-3:] != z_f.shape[-3:]:
                control_grid = F.interpolate(
                    control_grid, size=z_f.shape[-3:], mode="trilinear", align_corners=False
                )
            x = torch.cat([z_f, control_grid], dim=1)
        return self.self_update_operator(x)
