from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentPressureCorrectionOperator(nn.Module):
    """
    Pressure-like latent correction operator.

    Input:
    - z_tilde: tentative fluid latent after bulk/interface exchange.

    Output:
    - z_corr: incompressibility-aware corrected latent.
    """

    def __init__(
        self,
        latent_channels: int,
        correction_scale: float = 0.2,
        hidden_channels: int = 0,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.correction_scale = float(correction_scale)
        h = int(hidden_channels) if int(hidden_channels) > 0 else max(1, self.latent_channels // 2)

        self.global_redistribution = nn.Conv3d(self.latent_channels, self.latent_channels, kernel_size=1, bias=False)
        self.potential_head = nn.Sequential(
            nn.Conv3d(self.latent_channels, h, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(h, 1, kernel_size=1),
        )
        self.grad_to_latent = nn.Conv3d(3, self.latent_channels, kernel_size=1, bias=False)
        self.velocity_like_proj = nn.Conv3d(self.latent_channels, 3, kernel_size=1, bias=False)

    @staticmethod
    def _finite_diff(x: torch.Tensor, dim: int) -> torch.Tensor:
        slices_hi = [slice(None)] * x.ndim
        slices_lo = [slice(None)] * x.ndim
        slices_hi[dim] = slice(1, None)
        slices_lo[dim] = slice(0, -1)
        diff = x[tuple(slices_hi)] - x[tuple(slices_lo)]
        pad_shape = list(x.shape)
        pad_shape[dim] = 1
        pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([diff, pad], dim=dim)

    @classmethod
    def _scalar_grad3d(cls, scalar_b1dhw: torch.Tensor) -> torch.Tensor:
        dphi_dz = cls._finite_diff(scalar_b1dhw, dim=2)
        dphi_dy = cls._finite_diff(scalar_b1dhw, dim=3)
        dphi_dx = cls._finite_diff(scalar_b1dhw, dim=4)
        return torch.cat([dphi_dx, dphi_dy, dphi_dz], dim=1)

    @classmethod
    def _divergence_3d(cls, vel_b3dhw: torch.Tensor) -> torch.Tensor:
        du_dx = cls._finite_diff(vel_b3dhw[:, 0:1], dim=4)
        dv_dy = cls._finite_diff(vel_b3dhw[:, 1:2], dim=3)
        dw_dz = cls._finite_diff(vel_b3dhw[:, 2:3], dim=2)
        return du_dx + dv_dy + dw_dz

    def forward(
        self,
        z_tilde: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        global_ctx = F.adaptive_avg_pool3d(z_tilde, output_size=1)
        z_global = z_tilde + self.global_redistribution(global_ctx.expand_as(z_tilde))

        phi = self.potential_head(z_global)
        grad_phi = self._scalar_grad3d(phi)
        pressure_corr = self.grad_to_latent(grad_phi)
        z_corr = z_global - self.correction_scale * pressure_corr

        if not return_diagnostics:
            return z_corr

        vel_pre = self.velocity_like_proj(z_global)
        vel_post = self.velocity_like_proj(z_corr)
        div_pre = torch.mean(torch.abs(self._divergence_3d(vel_pre)), dim=(1, 2, 3, 4))
        div_post = torch.mean(torch.abs(self._divergence_3d(vel_post)), dim=(1, 2, 3, 4))

        diagnostics = {
            "pressure_correction_norm": torch.mean(torch.abs(pressure_corr), dim=(1, 2, 3, 4)),
            "pressure_divergence_pre": div_pre,
            "pressure_divergence_post": div_post,
        }
        return z_corr, diagnostics
