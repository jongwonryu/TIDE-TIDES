from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def _group_norm_channels(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class InterfaceBranchEncoder3D(nn.Module):
    """
    Encodes interface fields into interface tokens.

    Supported grid keys in interface_inputs:
    - fluid_solid_interface: [B,1,D,H,W] (optional)
    - liquid_liquid_interface: [B,1,D,H,W] (optional)
    - interface_sdf: [B,1,D,H,W] (optional)
    - interface_normal: [B,3,D,H,W] (optional)
    - boundary_velocity: [B,3,D,H,W] (optional)
    - boundary_mask: [B,1,D,H,W] (optional)
    """

    def __init__(
        self,
        token_dim: int,
        input_channels: int = 10,
        object_patch_channels: int = 7,
        base_width: int = 16,
        token_grid: Tuple[int, int, int] = (2, 4, 4),
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.input_channels = int(input_channels)
        self.object_patch_channels = int(object_patch_channels)
        self.token_grid = tuple(int(x) for x in token_grid)

        w0 = int(base_width)
        w1 = int(base_width * 2)
        self.stem = nn.Sequential(
            nn.Conv3d(self.input_channels, w0, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_norm_channels(w0), w0),
            nn.GELU(),
            nn.Conv3d(w0, w1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_norm_channels(w1), w1),
            nn.GELU(),
            nn.Conv3d(w1, w1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_norm_channels(w1), w1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool3d(self.token_grid)
        self.proj = nn.Linear(w1, self.token_dim)
        self.object_patch_encoder = nn.Sequential(
            nn.Conv3d(self.object_patch_channels, w0, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_norm_channels(w0), w0),
            nn.GELU(),
            nn.Conv3d(w0, w1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_norm_channels(w1), w1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.object_patch_proj = nn.Linear(w1, self.token_dim)
        self.norm = nn.LayerNorm(self.token_dim)

    def _collect_grids(
        self, interface_inputs: Dict[str, torch.Tensor] | None, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        grids = []
        if interface_inputs is not None:
            for key in (
                "fluid_solid_interface",
                "liquid_liquid_interface",
                "interface_sdf",
                "interface_normal",
                "boundary_velocity",
                "boundary_mask",
            ):
                v = interface_inputs.get(key, None)
                if v is not None:
                    grids.append(v.float())

        if not grids:
            # fallback: zero field
            return torch.zeros(batch_size, self.input_channels, 8, 16, 16, device=device)

        x = torch.cat(grids, dim=1).to(device)
        c = x.shape[1]
        if c == self.input_channels:
            return x
        if c > self.input_channels:
            return x[:, : self.input_channels]
        pad = torch.zeros(
            x.shape[0],
            self.input_channels - c,
            x.shape[2],
            x.shape[3],
            x.shape[4],
            device=x.device,
            dtype=x.dtype,
        )
        return torch.cat([x, pad], dim=1)

    def _collect_object_patches(
        self, interface_inputs: Dict[str, torch.Tensor] | None
    ) -> torch.Tensor | None:
        if interface_inputs is None:
            return None
        patches = []
        for key in (
            "object_interface_patch",
            "object_interface_normal_patch",
            "object_boundary_velocity_patch",
        ):
            v = interface_inputs.get(key, None)
            if v is not None:
                patches.append(v.float())
        if not patches:
            return None
        x = torch.cat(patches, dim=2)  # [B,N,C,P,P,P]
        c = x.shape[2]
        if c > self.object_patch_channels:
            return x[:, :, : self.object_patch_channels]
        if c < self.object_patch_channels:
            pad = torch.zeros(
                x.shape[0],
                x.shape[1],
                self.object_patch_channels - c,
                x.shape[3],
                x.shape[4],
                x.shape[5],
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([x, pad], dim=2)
        return x

    def forward(
        self,
        interface_inputs: Dict[str, torch.Tensor] | None,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._collect_grids(interface_inputs, batch_size=batch_size, device=device)
        h = self.pool(self.stem(x))
        # [B, C, D, H, W] -> [B, N, C]
        h = h.flatten(2).transpose(1, 2)
        tokens_global = self.proj(h)
        mask_global = torch.ones(
            tokens_global.shape[0], tokens_global.shape[1], dtype=torch.bool, device=tokens_global.device
        )

        obj_patch = self._collect_object_patches(interface_inputs)
        if obj_patch is not None:
            obj_patch = obj_patch.to(device)
            bsz, nobj = obj_patch.shape[:2]
            patch_feat = self.object_patch_encoder(
                obj_patch.reshape(
                    bsz * nobj,
                    obj_patch.shape[2],
                    obj_patch.shape[3],
                    obj_patch.shape[4],
                    obj_patch.shape[5],
                )
            ).reshape(bsz, nobj, -1)
            tokens_obj = self.object_patch_proj(patch_feat)
            mask_obj = torch.ones(bsz, nobj, dtype=torch.bool, device=tokens_obj.device)
            tokens = torch.cat([tokens_global, tokens_obj], dim=1)
            mask = torch.cat([mask_global, mask_obj], dim=1)
        else:
            tokens = tokens_global
            mask = mask_global
        tokens = self.norm(tokens)
        return tokens, mask
