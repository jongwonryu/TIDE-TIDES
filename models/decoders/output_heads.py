from __future__ import annotations

import torch
import torch.nn as nn


def _group_norm_channels(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _Head3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_channels) if hidden_channels is not None else int(in_channels)
        g = _group_norm_channels(hidden)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, hidden),
            nn.GELU(),
            nn.Conv3d(hidden, out_channels, kernel_size=1),
        )

    def forward(self, z_f: torch.Tensor) -> torch.Tensor:
        return self.net(z_f)


class VelocityHead3D(_Head3D):
    def __init__(self, in_channels: int, out_channels: int = 3, hidden_channels: int | None = None) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels)


class PhaseOccupancyHead3D(_Head3D):
    def __init__(self, in_channels: int, num_phases: int = 3, hidden_channels: int | None = None) -> None:
        super().__init__(in_channels=in_channels, out_channels=num_phases, hidden_channels=hidden_channels)

    def forward(self, z_f: torch.Tensor) -> torch.Tensor:
        # Return raw logits for CE/BCE-style supervision.
        return super().forward(z_f)


class PressureHead3D(_Head3D):
    def __init__(self, in_channels: int, hidden_channels: int | None = None) -> None:
        super().__init__(in_channels=in_channels, out_channels=1, hidden_channels=hidden_channels)


class InterfaceAwareVelocityProjection(nn.Module):
    """
    Lightweight output-space velocity correction block.

    Inputs are tentative velocity and interface/boundary conditions.
    """

    def __init__(self, in_channels: int = 9, hidden_channels: int = 32, correction_scale: float = 0.1) -> None:
        super().__init__()
        hidden = int(max(8, hidden_channels))
        self.correction_scale = float(max(0.0, correction_scale))
        self.net = nn.Sequential(
            nn.Conv3d(int(in_channels), hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_norm_channels(hidden), hidden),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_norm_channels(hidden), hidden),
            nn.GELU(),
            nn.Conv3d(hidden, 3, kernel_size=1, bias=False),
        )

    def forward(
        self,
        velocity_tilde: torch.Tensor,
        interface_sdf: torch.Tensor,
        boundary_mask: torch.Tensor,
        boundary_velocity: torch.Tensor,
        interaction_intensity: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.cat(
            [velocity_tilde, interface_sdf, boundary_mask, boundary_velocity, interaction_intensity], dim=1
        )
        delta_u = self.net(x)
        velocity_corr = velocity_tilde + self.correction_scale * delta_u
        # Hard boundary overwrite for robust no-slip/no-penetration handling.
        velocity_corr = velocity_corr * (1.0 - boundary_mask) + boundary_velocity * boundary_mask
        diag = {
            "correction_norm": torch.mean(torch.abs(velocity_corr - velocity_tilde), dim=(1, 2, 3, 4)),
        }
        return velocity_corr, diag
