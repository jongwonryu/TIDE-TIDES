from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class TokenBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        q = self.norm1(x).transpose(0, 1)  # [N,B,C]
        y, _ = self.attn(q, q, q, key_padding_mask=key_padding_mask)
        y = y.transpose(0, 1)  # [B,N,C]
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class ObjectBranchEncoder(nn.Module):
    """
    Encodes per-object states into object tokens.

    Expected keys in object_inputs:
    - type_id: [B, N] (long, optional)
    - geometry: [B, N, G] (optional)
    - pose: [B, N, P] (optional)
    - velocity: [B, N, 3] (optional)
    - trajectory: [B, N, T] (optional)
    - control: [B, N, C] (optional)
    - mask: [B, N] bool (optional, True=valid)
    """

    def __init__(
        self,
        token_dim: int,
        num_object_types: int = 3,
        dynamic_input_dim: int = 64,
        geometry_input_dim: int = 32,
        patch_input_channels: int = 8,
        geometry_stream_scale: float = 1.0,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.patch_input_channels = int(max(1, patch_input_channels))
        self.geometry_stream_scale = float(max(0.0, geometry_stream_scale))
        self.type_emb = nn.Embedding(int(num_object_types), self.token_dim)
        self.dynamic_proj = nn.Sequential(
            nn.Linear(int(dynamic_input_dim), self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.geometry_summary_proj = nn.Sequential(
            nn.Linear(int(geometry_input_dim), self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        patch_hidden = max(8, self.token_dim // 2)
        self.patch_encoder = nn.Sequential(
            nn.Conv3d(self.patch_input_channels, patch_hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, patch_hidden),
            nn.GELU(),
            nn.Conv3d(patch_hidden, self.token_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, self.token_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.geometry_fuse = nn.Sequential(
            nn.Linear(self.token_dim * 2, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.geometry_gate = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.Sigmoid(),
        )
        # Keep dynamic and geometry streams disentangled through token refinement.
        self.dyn_blocks = nn.ModuleList(
            [TokenBlock(dim=self.token_dim, num_heads=num_heads) for _ in range(int(num_layers))]
        )
        self.geo_blocks = nn.ModuleList(
            [TokenBlock(dim=self.token_dim, num_heads=num_heads) for _ in range(int(num_layers))]
        )
        self.dyn_norm = nn.LayerNorm(self.token_dim)
        self.geo_norm = nn.LayerNorm(self.token_dim)
        self.norm = nn.LayerNorm(self.token_dim)

    def _collect_dynamic_numeric(self, object_inputs: Dict[str, torch.Tensor], b: int, n: int) -> torch.Tensor:
        feats = []
        for key in ("pose", "velocity_local", "trajectory_local", "control", "local_frame", "center_voxel"):
            v = object_inputs.get(key, None)
            if v is None and key == "velocity_local":
                v = object_inputs.get("velocity", None)
            if v is None and key == "trajectory_local":
                v = object_inputs.get("trajectory", None)
            if v is None:
                continue
            vf = v.float()
            if vf.ndim > 3:
                vf = vf.reshape(vf.shape[0], vf.shape[1], -1)
            feats.append(vf)
        if feats:
            return torch.cat(feats, dim=-1)
        return torch.zeros(b, n, 1, device=next(self.parameters()).device)

    def _collect_geometry_summary(self, object_inputs: Dict[str, torch.Tensor], b: int, n: int) -> torch.Tensor:
        v = object_inputs.get("geometry", None)
        if v is None:
            return torch.zeros(b, n, 1, device=next(self.parameters()).device)
        vg = v.float()
        if vg.ndim > 3:
            vg = vg.reshape(vg.shape[0], vg.shape[1], -1)
        return vg

    def _collect_patch(self, object_inputs: Dict[str, torch.Tensor], b: int, n: int) -> torch.Tensor | None:
        patches = []
        for key in (
            "object_sdf_patch",
            "object_surface_patch",
            "object_interface_patch",
            "object_interface_normal_patch",
            "object_boundary_velocity_patch",
        ):
            v = object_inputs.get(key, None)
            if v is not None:
                patches.append(v.float())
        if not patches:
            return None
        return torch.cat(patches, dim=2)  # [B,N,C,Px,Py,Pz]

    def _fit_patch_channels(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[2]
        if c == self.patch_input_channels:
            return x
        if c > self.patch_input_channels:
            return x[:, :, : self.patch_input_channels]
        pad = torch.zeros(
            x.shape[0],
            x.shape[1],
            self.patch_input_channels - c,
            x.shape[3],
            x.shape[4],
            x.shape[5],
            dtype=x.dtype,
            device=x.device,
        )
        return torch.cat([x, pad], dim=2)

    def _fit_numeric_dim(self, x: torch.Tensor, target_dim: int) -> torch.Tensor:
        c = x.shape[-1]
        if c == target_dim:
            return x
        if c > target_dim:
            return x[..., :target_dim]
        pad = torch.zeros(*x.shape[:-1], target_dim - c, device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=-1)

    def forward(
        self,
        object_inputs: Dict[str, torch.Tensor] | None,
        batch_size: int | None = None,
        device: torch.device | None = None,
        return_components: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if object_inputs is None:
            if batch_size is None or device is None:
                raise ValueError("batch_size and device are required when object_inputs is None.")
            tokens = torch.zeros(batch_size, 1, self.token_dim, device=device)
            mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            if return_components:
                z = torch.zeros_like(tokens)
                return tokens, mask, {"dyn": z, "geo": z, "joint": z}
            return tokens, mask

        type_id = object_inputs.get("type_id", None)
        if type_id is None:
            if batch_size is None:
                raise ValueError("batch_size is required when type_id is not provided.")
            n = int(object_inputs.get("num_objects", 1))
            type_id = torch.zeros(batch_size, n, dtype=torch.long, device=device)
        else:
            type_id = type_id.long()
            batch_size = type_id.shape[0]
            n = type_id.shape[1]
            device = type_id.device

        dyn_numeric = self._collect_dynamic_numeric(object_inputs, batch_size, n)
        dyn_numeric = self._fit_numeric_dim(dyn_numeric, self.dynamic_proj[0].in_features)
        geo_summary = self._collect_geometry_summary(object_inputs, batch_size, n)
        geo_summary = self._fit_numeric_dim(geo_summary, self.geometry_summary_proj[0].in_features)
        patch = self._collect_patch(object_inputs, batch_size, n)

        base = self.type_emb(type_id.clamp(min=0, max=self.type_emb.num_embeddings - 1))
        dyn_tok = self.dynamic_proj(dyn_numeric.to(base.device))
        geo_sum_tok = self.geometry_summary_proj(geo_summary.to(base.device))
        dyn_state = base + dyn_tok
        geo_state = geo_sum_tok
        if patch is not None:
            patch = self._fit_patch_channels(patch.to(base.device))
            bn = patch.shape[0] * patch.shape[1]
            pfeat = self.patch_encoder(
                patch.reshape(
                    bn,
                    patch.shape[2],
                    patch.shape[3],
                    patch.shape[4],
                    patch.shape[5],
                )
            ).reshape(patch.shape[0], patch.shape[1], self.token_dim)
            geo_tok = self.geometry_fuse(torch.cat([geo_sum_tok, pfeat], dim=-1))
            geo_gate = self.geometry_gate(pfeat)
            geo_state = geo_gate * geo_tok

        mask = object_inputs.get("mask", None)
        if mask is None:
            mask = torch.ones(batch_size, n, dtype=torch.bool, device=base.device)
        else:
            mask = mask.bool().to(base.device)

        key_padding_mask = ~mask  # True means ignore for nn.MultiheadAttention
        dyn_tokens = dyn_state
        geo_tokens = geo_state
        for block in self.dyn_blocks:
            dyn_tokens = block(dyn_tokens, key_padding_mask=key_padding_mask)
        for block in self.geo_blocks:
            geo_tokens = block(geo_tokens, key_padding_mask=key_padding_mask)
        dyn_tokens = self.dyn_norm(dyn_tokens)
        geo_tokens = self.geo_norm(geo_tokens)

        # For geometry-off ablations we want the geometry stream to be truly absent,
        # not just down-weighted in the joint readout.
        if self.geometry_stream_scale <= 0.0:
            geo_tokens = torch.zeros_like(geo_tokens)
            tokens = dyn_tokens
        else:
            tokens = dyn_tokens + self.geometry_stream_scale * geo_tokens
        tokens = self.norm(tokens)
        if return_components:
            return tokens, mask, {"dyn": dyn_tokens, "geo": geo_tokens, "joint": tokens}
        return tokens, mask
