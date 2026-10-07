from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.fluid_uno_branch_3d import FluidUNOBranch3D
from ..decoders.output_heads import InterfaceAwareVelocityProjection
from .latent_pressure_correction import LatentPressureCorrectionOperator


class LatentCouplingRolloutCore(nn.Module):
    """
    One-step latent coupling update (interface-first):
    (z_f^t, z_obj^t, z_int^t) -> (z_f^{t+1}, z_obj^{t+1}, z_int^{t+1})
    """

    def __init__(
        self,
        fluid_branch: FluidUNOBranch3D,
        latent_channels: int,
        control_dim: int = 0,
        control_channels: int = 0,
        token_grid: Tuple[int, int, int] = (2, 4, 4),
        num_heads: int = 4,
        coupling_scale: float = 1.0,
        coupling_variant: str = "strict_mediator",
        direct_obj_to_fluid_scale: float = 0.0,
        direct_fluid_to_obj_scale: float = 0.0,
        exchange_reaction_scale: float = 1.0,
        geo_interface_condition_scale: float = 1.0,
        geo_operator_init_scale: float = 0.1,
        pressure_correction_scale: float = 0.2,
        pressure_post_p2g_enabled: bool = True,
        pressure_post_p2g_scale: float = 1.0,
        output_pressure_correction_enabled: bool = True,
        output_pressure_correction_scale: float = 0.1,
        output_pressure_feedback_scale: float = 0.1,
        dual_coupling_enabled: bool = True,
        dual_particle_step_scale: float = 1.0,
        dual_p2g_scale: float = 0.2,
        interface_near_aware_enabled: bool = False,
        interface_near_gate_strength: float = 1.0,
        interface_near_basis_scales: Sequence[float] = (0.10, 0.25, 0.50),
        interface_near_learned_refine_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.fluid_branch = fluid_branch
        self.latent_channels = int(latent_channels)
        self.control_dim = int(max(0, control_dim))
        self.control_channels = int(max(0, control_channels))
        self.token_grid = tuple(int(x) for x in token_grid)
        self.coupling_scale = float(coupling_scale)
        self.coupling_variant = str(coupling_variant).lower().strip()
        valid_variants = {"direct", "interface_mediated", "strict_mediator"}
        if self.coupling_variant not in valid_variants:
            raise ValueError(
                f"coupling_variant must be one of {sorted(valid_variants)}, got {coupling_variant!r}"
            )
        self.direct_obj_to_fluid_scale = float(direct_obj_to_fluid_scale)
        self.direct_fluid_to_obj_scale = float(direct_fluid_to_obj_scale)
        self.exchange_reaction_scale = float(exchange_reaction_scale)
        self.geo_interface_condition_scale = float(max(0.0, geo_interface_condition_scale))
        self.geo_operator_init_scale = float(max(0.0, geo_operator_init_scale))
        self.pressure_correction_scale = float(pressure_correction_scale)
        self.pressure_post_p2g_enabled = bool(pressure_post_p2g_enabled)
        self.pressure_post_p2g_scale = float(max(0.0, pressure_post_p2g_scale))
        self.output_pressure_correction_enabled = bool(output_pressure_correction_enabled)
        self.output_pressure_correction_scale = float(max(0.0, output_pressure_correction_scale))
        self.output_pressure_feedback_scale = float(max(0.0, output_pressure_feedback_scale))
        self.dual_coupling_enabled = bool(dual_coupling_enabled)
        self.dual_particle_step_scale = float(dual_particle_step_scale)
        self.dual_p2g_scale = float(dual_p2g_scale)
        self.interface_near_aware_enabled = bool(interface_near_aware_enabled)
        self.interface_near_gate_strength = float(max(0.0, min(1.0, interface_near_gate_strength)))
        self.interface_near_basis_scales = tuple(float(max(1e-4, s)) for s in interface_near_basis_scales)
        self.interface_near_learned_refine_enabled = bool(interface_near_learned_refine_enabled)

        self.obj_proj = nn.Linear(self.latent_channels, self.latent_channels)
        self.obj_dyn_proj = nn.Linear(self.latent_channels, self.latent_channels)
        self.obj_geo_proj = nn.Linear(self.latent_channels, self.latent_channels)
        # Geometry-conditioned interface coupling: geo stream modulates Q/K/V of object->interface attention.
        self.obj_geo_q_proj = nn.Linear(self.latent_channels, self.latent_channels)
        self.obj_geo_k_proj = nn.Linear(self.latent_channels, self.latent_channels)
        self.obj_geo_k_bias_proj = nn.Linear(self.latent_channels, self.latent_channels)
        self.obj_geo_v_proj = nn.Linear(self.latent_channels, self.latent_channels)
        self.obj_geo_out_bias_proj = nn.Linear(self.latent_channels, self.latent_channels)
        # Learnable small-init modulation scales for geometry-conditioned operator.
        init_scale = float(self.geo_operator_init_scale)
        self.obj_geo_q_scale = nn.Parameter(torch.tensor(init_scale))
        self.obj_geo_k_scale = nn.Parameter(torch.tensor(init_scale))
        self.obj_geo_v_scale = nn.Parameter(torch.tensor(init_scale))
        self.obj_geo_out_scale = nn.Parameter(torch.tensor(init_scale))
        self.obj_to_int = nn.MultiheadAttention(embed_dim=self.latent_channels, num_heads=num_heads)
        self.obj_to_fluid = nn.MultiheadAttention(embed_dim=self.latent_channels, num_heads=num_heads)
        self.int_to_fluid = nn.MultiheadAttention(embed_dim=self.latent_channels, num_heads=num_heads)
        self.fluid_to_int = nn.MultiheadAttention(embed_dim=self.latent_channels, num_heads=num_heads)
        self.fluid_to_int_exchange = nn.MultiheadAttention(embed_dim=self.latent_channels, num_heads=num_heads)
        self.fluid_to_obj = nn.MultiheadAttention(embed_dim=self.latent_channels, num_heads=num_heads)
        self.int_to_obj = nn.MultiheadAttention(embed_dim=self.latent_channels, num_heads=num_heads)
        self.token_norm = nn.LayerNorm(self.latent_channels)
        self.exchange_gain_proj = nn.Linear(self.latent_channels, self.latent_channels)
        self.exchange_react_proj = nn.Linear(self.latent_channels, self.latent_channels)

        self.impulse_refine = nn.Sequential(
            nn.Conv3d(self.latent_channels, self.latent_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, self.latent_channels),
            nn.GELU(),
            nn.Conv3d(self.latent_channels, self.latent_channels, kernel_size=3, padding=1, bias=False),
        )
        self.intensity_head = nn.Sequential(
            nn.Conv3d(self.latent_channels, max(1, self.latent_channels // 2), kernel_size=1),
            nn.GELU(),
            nn.Conv3d(max(1, self.latent_channels // 2), 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.interface_near_refine = nn.Sequential(
            nn.Conv3d(len(self.interface_near_basis_scales), max(4, len(self.interface_near_basis_scales) * 2), kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv3d(max(4, len(self.interface_near_basis_scales) * 2), 1, kernel_size=1, bias=True),
            nn.Tanh(),
        )
        self.pressure_operator = LatentPressureCorrectionOperator(
            latent_channels=self.latent_channels,
            correction_scale=self.pressure_correction_scale,
        )
        self.pressure_operator_post = LatentPressureCorrectionOperator(
            latent_channels=self.latent_channels,
            correction_scale=self.pressure_correction_scale * self.pressure_post_p2g_scale,
        )
        self.dual_velocity_head = nn.Conv3d(self.latent_channels, 3, kernel_size=1, bias=False)
        self.output_velocity_projection = InterfaceAwareVelocityProjection(
            in_channels=9,
            hidden_channels=max(16, self.latent_channels),
            correction_scale=self.output_pressure_correction_scale,
        )
        self.output_velocity_to_latent = nn.Sequential(
            nn.Conv3d(4, self.latent_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, self.latent_channels),
            nn.GELU(),
            nn.Conv3d(self.latent_channels, self.latent_channels, kernel_size=1, bias=False),
        )
        self.dual_p2g_refine = nn.Sequential(
            nn.Conv3d(self.latent_channels + 4, self.latent_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, self.latent_channels),
            nn.GELU(),
            nn.Conv3d(self.latent_channels, self.latent_channels, kernel_size=3, padding=1, bias=False),
        )
        self.post_norm = nn.GroupNorm(1, self.latent_channels)

        if self.control_dim > 0:
            if self.control_channels <= 0:
                raise ValueError("control_channels must be > 0 when control_dim > 0")
            self.control_to_grid = nn.Linear(self.control_dim, self.control_channels)
            self.control_to_token = nn.Linear(self.control_dim, self.latent_channels)
        else:
            self.control_to_grid = None
            self.control_to_token = None

    def _effective_direct_obj_to_fluid_scale(self) -> float:
        if self.coupling_variant != "direct":
            return 0.0
        return float(max(0.0, self.direct_obj_to_fluid_scale))

    def _effective_direct_fluid_to_obj_scale(self) -> float:
        if self.coupling_variant == "strict_mediator":
            return 0.0
        return float(max(0.0, self.direct_fluid_to_obj_scale))

    def _grid_to_tokens(self, z_f: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        pooled = F.adaptive_avg_pool3d(z_f, self.token_grid)
        d, h, w = pooled.shape[-3:]
        tokens = pooled.flatten(2).transpose(1, 2)  # [B, N, C]
        return tokens, (d, h, w)

    def _tokens_to_grid(
        self,
        tokens: torch.Tensor,
        token_shape: Tuple[int, int, int],
        out_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        bsz = tokens.shape[0]
        d, h, w = token_shape
        x = tokens.transpose(1, 2).reshape(bsz, self.latent_channels, d, h, w)
        if (d, h, w) != out_shape:
            x = F.interpolate(x, size=out_shape, mode="trilinear", align_corners=False)
        return x

    def _control(
        self,
        control_vec: Optional[torch.Tensor],
        z_f: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.control_dim <= 0:
            return None, None
        bsz = z_f.shape[0]
        if control_vec is None:
            control_vec = torch.zeros(bsz, self.control_dim, device=z_f.device, dtype=z_f.dtype)
        else:
            control_vec = control_vec.to(z_f.device, dtype=z_f.dtype)
            if control_vec.ndim != 2 or control_vec.shape[1] != self.control_dim:
                raise ValueError(
                    f"control_vec must be [B,{self.control_dim}], got {tuple(control_vec.shape)}"
                )
        control_grid = self.control_to_grid(control_vec)[:, :, None, None, None].expand(
            bsz, self.control_channels, z_f.shape[-3], z_f.shape[-2], z_f.shape[-1]
        )
        control_token = self.control_to_token(control_vec)[:, None, :]  # [B,1,C]
        return control_grid, control_token

    @staticmethod
    def _sample_at_points(field_bcdhw: torch.Tensor, pts_bnx: torch.Tensor) -> torch.Tensor:
        # field: [B,C,D,H,W], pts: [B,N,3] in voxel coordinates (x,y,z)
        bsz, _, d, h, w = field_bcdhw.shape
        n = pts_bnx.shape[1]
        x = pts_bnx[..., 0].clamp(0.0, float(max(1, w - 1)))
        y = pts_bnx[..., 1].clamp(0.0, float(max(1, h - 1)))
        z = pts_bnx[..., 2].clamp(0.0, float(max(1, d - 1)))
        gx = 2.0 * x / max(1.0, float(w - 1)) - 1.0
        gy = 2.0 * y / max(1.0, float(h - 1)) - 1.0
        gz = 2.0 * z / max(1.0, float(d - 1)) - 1.0
        grid = torch.stack([gx, gy, gz], dim=-1).view(bsz, n, 1, 1, 3)
        sampled = F.grid_sample(
            field_bcdhw,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled[:, :, :, 0, 0].transpose(1, 2)  # [B,N,C]

    @staticmethod
    def _advect_particles(
        particle_pos: torch.Tensor,
        vel_like: torch.Tensor,
        step_scale: float,
        grid_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d, h, w = [int(x) for x in grid_shape]
        disp = LatentCouplingRolloutCore._sample_at_points(vel_like, particle_pos) * float(step_scale)
        p_next = particle_pos + disp
        p_next[..., 0].clamp_(0.0, float(max(1, w - 1)))
        p_next[..., 1].clamp_(0.0, float(max(1, h - 1)))
        p_next[..., 2].clamp_(0.0, float(max(1, d - 1)))
        return p_next, disp

    @staticmethod
    def _rasterize_particles_to_grid(
        p_prev: torch.Tensor,
        p_next: torch.Tensor,
        grid_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns:
        # - density: [B,1,D,H,W]
        # - mean_disp: [B,3,D,H,W]
        bsz, n, _ = p_prev.shape
        d, h, w = [int(x) for x in grid_shape]
        device = p_prev.device
        dtype = p_prev.dtype
        density = torch.zeros(bsz, 1, d, h, w, dtype=dtype, device=device)
        disp_sum = torch.zeros(bsz, 3, d, h, w, dtype=dtype, device=device)
        disp = p_next - p_prev

        ix = torch.round(p_next[..., 0]).long().clamp(0, max(0, w - 1))
        iy = torch.round(p_next[..., 1]).long().clamp(0, max(0, h - 1))
        iz = torch.round(p_next[..., 2]).long().clamp(0, max(0, d - 1))
        lin = iz * (h * w) + iy * w + ix  # [B,N]

        flat_size = d * h * w
        ones = torch.ones((n,), dtype=dtype, device=device)
        for b in range(bsz):
            cnt_flat = torch.zeros((flat_size,), dtype=dtype, device=device)
            cnt_flat.scatter_add_(0, lin[b], ones)
            density[b, 0] = cnt_flat.view(d, h, w)
            for c in range(3):
                sum_flat = torch.zeros((flat_size,), dtype=dtype, device=device)
                sum_flat.scatter_add_(0, lin[b], disp[b, :, c])
                disp_sum[b, c] = sum_flat.view(d, h, w)

        mean_disp = disp_sum / density.clamp_min(1.0)
        density = density / density.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)
        return density, mean_disp

    @staticmethod
    def _divergence_mean_abs(vel: torch.Tensor) -> torch.Tensor:
        # vel: [B,3,D,H,W], returns [B]
        vx = vel[:, 0]
        vy = vel[:, 1]
        vz = vel[:, 2]
        dx = vx[:, :, :, 1:] - vx[:, :, :, :-1]
        dy = vy[:, :, 1:, :] - vy[:, :, :-1, :]
        dz = vz[:, 1:, :, :] - vz[:, :-1, :, :]
        div = (
            F.pad(dx, (0, 1, 0, 0, 0, 0))
            + F.pad(dy, (0, 0, 0, 1, 0, 0))
            + F.pad(dz, (0, 0, 0, 0, 0, 1))
        )
        return torch.mean(torch.abs(div), dim=(1, 2, 3))

    @staticmethod
    def _boundary_velocity_mismatch(
        vel: torch.Tensor,
        boundary_vel: Optional[torch.Tensor],
        boundary_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if boundary_vel is None or boundary_mask is None:
            return torch.zeros(vel.shape[0], dtype=vel.dtype, device=vel.device)
        mask = boundary_mask.clamp(0.0, 1.0)
        denom = torch.sum(mask, dim=(1, 2, 3, 4)).clamp_min(1.0)
        err = torch.sum(torch.abs(vel - boundary_vel) * mask, dim=(1, 2, 3, 4)) / denom
        return err

    @staticmethod
    def _get_interface_grid(
        interface_grid: Optional[Dict[str, torch.Tensor]],
        key: str,
        channels: int,
        like: torch.Tensor,
    ) -> torch.Tensor:
        if interface_grid is None:
            return torch.zeros(
                like.shape[0],
                channels,
                like.shape[2],
                like.shape[3],
                like.shape[4],
                dtype=like.dtype,
                device=like.device,
            )
        v = interface_grid.get(key, None)
        if v is None:
            return torch.zeros(
                like.shape[0],
                channels,
                like.shape[2],
                like.shape[3],
                like.shape[4],
                dtype=like.dtype,
                device=like.device,
            )
        x = v.to(device=like.device, dtype=like.dtype)
        if x.ndim != 5:
            x = x.reshape(x.shape[0], x.shape[1], like.shape[2], like.shape[3], like.shape[4])
        if x.shape[2:] != like.shape[2:]:
            x = F.interpolate(x, size=like.shape[2:], mode="trilinear", align_corners=False)
        if x.shape[1] == channels:
            return x
        if x.shape[1] > channels:
            return x[:, :channels]
        pad = torch.zeros(
            x.shape[0],
            channels - x.shape[1],
            x.shape[2],
            x.shape[3],
            x.shape[4],
            dtype=x.dtype,
            device=x.device,
        )
        return torch.cat([x, pad], dim=1)

    def _interface_near_gate(
        self,
        interface_grid: Optional[Dict[str, torch.Tensor]],
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if interface_grid is None:
            ones = torch.ones(
                like.shape[0], 1, like.shape[2], like.shape[3], like.shape[4], dtype=like.dtype, device=like.device
            )
            return ones, ones
        prior_gate = self._get_interface_grid(interface_grid, "interface_near_weight", 1, like).clamp(0.0, 1.0)
        basis = self._get_interface_grid(
            interface_grid,
            "interface_near_basis",
            len(self.interface_near_basis_scales),
            like,
        ).clamp(0.0, 1.0)
        if not self.interface_near_aware_enabled:
            ones = torch.ones_like(prior_gate)
            return ones, prior_gate
        learned_residual = torch.zeros_like(prior_gate)
        if self.interface_near_learned_refine_enabled:
            learned_residual = 0.15 * self.interface_near_refine(basis)
        gate = (prior_gate + learned_residual).clamp(0.0, 1.0)
        blended = (1.0 - self.interface_near_gate_strength) + self.interface_near_gate_strength * gate
        return blended.clamp(0.0, 1.0), prior_gate

    def step(
        self,
        z_f: torch.Tensor,
        z_obj: torch.Tensor,
        z_int: torch.Tensor,
        z_obj_dyn: Optional[torch.Tensor] = None,
        z_obj_geo: Optional[torch.Tensor] = None,
        particle_pos: Optional[torch.Tensor] = None,
        object_mask: Optional[torch.Tensor] = None,
        interface_mask: Optional[torch.Tensor] = None,
        interface_grid: Optional[Dict[str, torch.Tensor]] = None,
        control_vec: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
            Dict[str, torch.Tensor],
        ],
    ]:
        fluid_tokens, token_shape = self._grid_to_tokens(z_f)  # [B,Nf,C]
        fluid_tokens_q = self.token_norm(fluid_tokens)

        if z_obj_dyn is None or z_obj_geo is None:
            obj_ctx = self.obj_proj(z_obj)
            obj_dyn_ctx = self.obj_dyn_proj(obj_ctx)
            obj_geo_ctx = self.obj_geo_proj(obj_ctx)
        else:
            obj_dyn_ctx = z_obj_dyn
            obj_geo_ctx = z_obj_geo
            obj_ctx = self.obj_proj(obj_dyn_ctx + obj_geo_ctx)

        obj_pad = None
        if object_mask is not None:
            obj_pad = ~object_mask.bool()
        int_pad = None
        if interface_mask is not None:
            int_pad = ~interface_mask.bool()

        control_grid, control_token = self._control(control_vec, z_f)
        direct_obj_to_fluid_scale = self._effective_direct_obj_to_fluid_scale()
        direct_fluid_to_obj_scale = self._effective_direct_fluid_to_obj_scale()

        # Optional direct object -> fluid path (used for ablation against strict mediator coupling).
        delta_obj_direct_grid = torch.zeros_like(z_f)
        if (obj_dyn_ctx.shape[1] > 0) and (direct_obj_to_fluid_scale > 0.0):
            obj_direct_tokens = self._attend(
                self.obj_to_fluid,
                fluid_tokens_q,
                self.token_norm(obj_dyn_ctx),
                self.token_norm(obj_dyn_ctx),
                key_padding_mask=obj_pad,
            )
            delta_obj_direct_grid = self._tokens_to_grid(
                self.token_norm(obj_direct_tokens),
                token_shape=token_shape,
                out_shape=tuple(z_f.shape[-3:]),
            )

        # 1) Interface-first state update from object and fluid context.
        int_query = self.token_norm(z_int)
        int_from_obj = torch.zeros_like(int_query)
        int_from_obj_raw = torch.zeros_like(int_query)
        obj_geo_gate_mean = torch.zeros(
            z_f.shape[0], dtype=z_f.dtype, device=z_f.device
        )
        obj_geo_to_int_raw_norm = torch.zeros(
            z_f.shape[0], dtype=z_f.dtype, device=z_f.device
        )
        obj_geo_to_int_gated_norm = torch.zeros(
            z_f.shape[0], dtype=z_f.dtype, device=z_f.device
        )
        obj_geo_effect_ratio = torch.zeros(
            z_f.shape[0], dtype=z_f.dtype, device=z_f.device
        )
        if obj_dyn_ctx.shape[1] > 0:
            dyn_ctx_n = self.token_norm(obj_dyn_ctx)
            geo_ctx_n = self.token_norm(obj_geo_ctx)

            if obj_pad is not None:
                valid = (~obj_pad).float().unsqueeze(-1)
                denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
                geo_summary = (geo_ctx_n * valid).sum(dim=1, keepdim=True) / denom
            else:
                geo_summary = geo_ctx_n.mean(dim=1, keepdim=True)

            # Base dynamics coupling (dominant path).
            int_from_obj_base = self._attend(
                self.obj_to_int,
                int_query,
                dyn_ctx_n,
                dyn_ctx_n,
                key_padding_mask=obj_pad,
            )

            # Geometry residual coupling (specificity path, not dominance):
            # - geometry modulates only value path (no full Q/K replacement),
            # - residual is scaled by a bounded gate to avoid saturation.
            geo_v_film = torch.tanh(self.obj_geo_v_proj(geo_ctx_n))
            obj_value_geo_res = dyn_ctx_n * geo_v_film
            int_from_obj_res_raw = self._attend(
                self.obj_to_int,
                int_query,
                dyn_ctx_n,
                obj_value_geo_res,
                key_padding_mask=obj_pad,
            )
            geo_out_bias = torch.tanh(self.obj_geo_out_bias_proj(geo_summary))
            int_from_obj_res = int_from_obj_res_raw + geo_out_bias

            # Bounded residual gate:
            #   gain in [0, 0.6], data-dependent gate in [0,1].
            # When geometry conditioning is explicitly disabled, force the
            # residual path to zero so geo-off ablations are exact.
            if self.geo_interface_condition_scale <= 0.0:
                geo_gate = torch.zeros_like(int_from_obj_res)
                int_from_obj = int_from_obj_base
            else:
                gain_cap = 0.6 * torch.sigmoid(
                    self.geo_interface_condition_scale * self.obj_geo_out_scale
                )
                gate_logits = torch.tanh(self.obj_geo_q_proj(geo_summary))
                geo_gate = gain_cap * torch.sigmoid(0.5 * gate_logits)
                int_from_obj = int_from_obj_base + geo_gate * int_from_obj_res
            obj_geo_gate_mean = torch.mean(geo_gate, dim=(1, 2))
            obj_geo_to_int_raw_norm = torch.mean(torch.abs(int_from_obj_res), dim=(1, 2))
            obj_geo_to_int_gated_norm = torch.mean(torch.abs(geo_gate * int_from_obj_res), dim=(1, 2))
            obj_geo_effect_ratio = obj_geo_to_int_gated_norm / obj_geo_to_int_raw_norm.clamp_min(1e-8)
            int_from_obj_raw = int_from_obj_base
        int_from_fluid = torch.zeros_like(int_query)
        if fluid_tokens_q.shape[1] > 0:
            int_from_fluid = self._attend(
                self.fluid_to_int, int_query, fluid_tokens_q, fluid_tokens_q
            )
        z_int_next = z_int + int_from_obj + int_from_fluid
        if control_token is not None:
            z_int_next = z_int_next + control_token

        # 2) Balanced interface-mediated latent exchange:
        #    (fluid gain, interface reaction) are computed as paired updates.
        delta_f_gain_tokens = torch.zeros_like(fluid_tokens_q)
        if z_int_next.shape[1] > 0:
            delta_f_gain_tokens = self._attend(
                self.int_to_fluid,
                fluid_tokens_q,
                self.token_norm(z_int_next),
                self.token_norm(z_int_next),
                key_padding_mask=int_pad,
            )
        delta_i_loss_tokens = torch.zeros_like(z_int_next)
        if z_int_next.shape[1] > 0 and fluid_tokens_q.shape[1] > 0:
            delta_i_loss_tokens = self._attend(
                self.fluid_to_int_exchange,
                self.token_norm(z_int_next),
                fluid_tokens_q,
                fluid_tokens_q,
            )
        z_int_balanced = z_int_next - self.exchange_reaction_scale * delta_i_loss_tokens

        delta_int_grid = self._tokens_to_grid(
            self.token_norm(delta_f_gain_tokens),
            token_shape=token_shape,
            out_shape=tuple(z_f.shape[-3:]),
        )
        delta_int_grid = self.impulse_refine(delta_int_grid)
        near_gate, near_gate_prior = self._interface_near_gate(interface_grid, delta_int_grid)
        interaction_intensity = self.intensity_head(torch.abs(delta_int_grid))
        delta_int_grid = delta_int_grid * near_gate * (1.0 + interaction_intensity)

        # 3) Bulk transport update + interface correction.
        delta_self = self.fluid_branch.self_update(z_f, control_grid=control_grid)
        delta_bulk = delta_self + direct_obj_to_fluid_scale * delta_obj_direct_grid

        z_f_tilde = z_f + delta_bulk + self.coupling_scale * delta_int_grid
        # 4) Pressure-like latent correction operator (stage-1, before dual P2G).
        if return_diagnostics:
            z_f_corr, pressure_diag_stage1 = self.pressure_operator(z_f_tilde, return_diagnostics=True)
        else:
            z_f_corr = self.pressure_operator(z_f_tilde, return_diagnostics=False)
            pressure_diag_stage1 = {}
        z_f_next = F.gelu(self.post_norm(z_f_corr))

        # 5) Explicit dual-space coupling: G2P advection + P2G latent feedback.
        particle_next = particle_pos
        dual_disp = None
        delta_p2g_grid = torch.zeros_like(z_f_next)
        if self.dual_coupling_enabled and (particle_pos is not None):
            vel_like = self.dual_velocity_head(z_f_next)
            particle_next, dual_disp = self._advect_particles(
                particle_pos=particle_pos,
                vel_like=vel_like,
                step_scale=self.dual_particle_step_scale,
                grid_shape=tuple(z_f_next.shape[-3:]),
            )
            density_grid, mean_disp_grid = self._rasterize_particles_to_grid(
                p_prev=particle_pos,
                p_next=particle_next,
                grid_shape=tuple(z_f_next.shape[-3:]),
            )
            if z_int_balanced.shape[1] > 0:
                int_ctx = self.token_norm(z_int_balanced).mean(dim=1)[:, :, None, None, None]
                int_ctx = int_ctx.expand(-1, -1, z_f_next.shape[-3], z_f_next.shape[-2], z_f_next.shape[-1])
            else:
                int_ctx = torch.zeros_like(z_f_next)
            p2g_input = torch.cat([density_grid, mean_disp_grid, int_ctx], dim=1)
            delta_p2g_grid = self.dual_p2g_refine(p2g_input)
            z_f_next = z_f_next + self.dual_p2g_scale * delta_p2g_grid

        # 6) Pressure-like latent correction operator (stage-2, after dual P2G).
        if self.pressure_post_p2g_enabled:
            if return_diagnostics:
                z_f_post, pressure_diag_stage2 = self.pressure_operator_post(z_f_next, return_diagnostics=True)
            else:
                z_f_post = self.pressure_operator_post(z_f_next, return_diagnostics=False)
                pressure_diag_stage2 = {}
            # Keep this projection-like to preserve correction effect.
            z_f_next = z_f_post
        else:
            pressure_diag_stage2 = {}

        # 7) Output-space interface-aware projection correction.
        u_tilde = self.dual_velocity_head(z_f_next)
        out_pressure_norm = torch.zeros(z_f_next.shape[0], dtype=z_f_next.dtype, device=z_f_next.device)
        out_div_pre = self._divergence_mean_abs(u_tilde)
        out_div_post = out_div_pre
        out_bnd_pre = torch.zeros_like(out_div_pre)
        out_bnd_post = torch.zeros_like(out_div_pre)
        if self.output_pressure_correction_enabled:
            interface_sdf = self._get_interface_grid(interface_grid, "interface_sdf", 1, u_tilde)
            boundary_mask = self._get_interface_grid(interface_grid, "boundary_mask", 1, u_tilde).clamp(0.0, 1.0)
            boundary_vel = self._get_interface_grid(interface_grid, "boundary_velocity", 3, u_tilde)
            u_corr, out_diag = self.output_velocity_projection(
                velocity_tilde=u_tilde,
                interface_sdf=interface_sdf,
                boundary_mask=boundary_mask,
                boundary_velocity=boundary_vel,
                interaction_intensity=interaction_intensity,
            )
            out_pressure_norm = out_diag["correction_norm"]
            out_div_post = self._divergence_mean_abs(u_corr)
            out_bnd_pre = self._boundary_velocity_mismatch(u_tilde, boundary_vel, boundary_mask)
            out_bnd_post = self._boundary_velocity_mismatch(u_corr, boundary_vel, boundary_mask)
            feedback = self.output_velocity_to_latent(
                torch.cat([u_corr - u_tilde, interaction_intensity], dim=1)
            )
            z_f_next = z_f_next + self.output_pressure_feedback_scale * feedback

        fluid_tokens_next, _ = self._grid_to_tokens(z_f_next)
        fluid_tokens_next = self.token_norm(fluid_tokens_next)

        # Object state: interface-dominant update, fluid feedback secondary.
        z_obj_dyn_next = obj_dyn_ctx
        z_obj_geo_next = obj_geo_ctx
        z_obj_next = z_obj
        if obj_dyn_ctx.shape[1] > 0:
            obj_from_int = self._attend(
                self.int_to_obj,
                self.token_norm(obj_dyn_ctx),
                self.token_norm(z_int_balanced),
                self.token_norm(z_int_balanced),
                key_padding_mask=int_pad,
            )
            fb_obj = self._attend(
                self.fluid_to_obj,
                self.token_norm(obj_dyn_ctx),
                fluid_tokens_next,
                fluid_tokens_next,
            )
            z_obj_dyn_next = obj_dyn_ctx + obj_from_int + direct_fluid_to_obj_scale * fb_obj
            z_obj_next = z_obj_dyn_next + z_obj_geo_next

        if delta_f_gain_tokens.shape[1] > 0:
            gain_summary = self.exchange_gain_proj(delta_f_gain_tokens.mean(dim=1))
        else:
            gain_summary = torch.zeros(
                z_f.shape[0], self.latent_channels, dtype=z_f.dtype, device=z_f.device
            )
        if delta_i_loss_tokens.shape[1] > 0:
            react_summary = self.exchange_react_proj(delta_i_loss_tokens.mean(dim=1))
        else:
            react_summary = torch.zeros(
                z_f.shape[0], self.latent_channels, dtype=z_f.dtype, device=z_f.device
            )
        exchange_balance_error = torch.mean(torch.abs(gain_summary + react_summary), dim=1)

        if return_diagnostics:
            if dual_disp is not None:
                dual_particle_step_norm = torch.mean(torch.abs(dual_disp), dim=(1, 2))
            else:
                dual_particle_step_norm = torch.zeros(
                    z_f_next.shape[0], dtype=z_f_next.dtype, device=z_f_next.device
                )

            zero_b = torch.zeros(z_f_next.shape[0], dtype=z_f_next.dtype, device=z_f_next.device)
            p1_norm = pressure_diag_stage1.get("pressure_correction_norm", zero_b)
            p1_pre = pressure_diag_stage1.get("pressure_divergence_pre", zero_b)
            p1_post = pressure_diag_stage1.get("pressure_divergence_post", zero_b)
            p2_norm = pressure_diag_stage2.get("pressure_correction_norm", zero_b)
            p2_pre = pressure_diag_stage2.get("pressure_divergence_pre", p1_post)
            p2_post = pressure_diag_stage2.get("pressure_divergence_post", p1_post)
            # Keep backward-compatible fields as net pre/post over latent + output stages.
            p_net_pre = p1_pre
            p_net_post = out_div_post
            p_net_norm = p1_norm + p2_norm + out_pressure_norm

            diagnostics = {
                "bulk_delta_norm": torch.mean(torch.abs(delta_bulk), dim=(1, 2, 3, 4)),
                "interface_delta_norm": torch.mean(torch.abs(delta_int_grid), dim=(1, 2, 3, 4)),
                "exchange_reaction_norm": torch.mean(torch.abs(delta_i_loss_tokens), dim=(1, 2)),
                "exchange_balance_error": exchange_balance_error,
                "dual_particle_step_norm": dual_particle_step_norm,
                "dual_p2g_norm": torch.mean(torch.abs(delta_p2g_grid), dim=(1, 2, 3, 4)),
                "direct_obj_delta_norm": torch.mean(torch.abs(delta_obj_direct_grid), dim=(1, 2, 3, 4)),
                "interaction_intensity_mean": torch.mean(interaction_intensity, dim=(1, 2, 3, 4)),
                "interface_near_gate_mean": torch.mean(near_gate, dim=(1, 2, 3, 4)),
                "interface_near_prior_mean": torch.mean(near_gate_prior, dim=(1, 2, 3, 4)),
                "obj_dyn_norm": torch.mean(torch.abs(obj_dyn_ctx), dim=(1, 2)),
                "obj_geo_norm": torch.mean(torch.abs(obj_geo_ctx), dim=(1, 2)),
                "obj_geo_gate_mean": obj_geo_gate_mean,
                "obj_geo_to_int_raw_norm": obj_geo_to_int_raw_norm,
                "obj_geo_to_int_gated_norm": obj_geo_to_int_gated_norm,
                "obj_geo_effect_ratio": obj_geo_effect_ratio,
                "pressure_correction_norm": p_net_norm,
                "pressure_divergence_pre": p_net_pre,
                "pressure_divergence_post": p_net_post,
                "pressure_stage1_correction_norm": p1_norm,
                "pressure_stage1_divergence_pre": p1_pre,
                "pressure_stage1_divergence_post": p1_post,
                "pressure_stage2_correction_norm": p2_norm,
                "pressure_stage2_divergence_pre": p2_pre,
                "pressure_stage2_divergence_post": p2_post,
                "pressure_stage2_divergence_delta": p2_post - p2_pre,
                "output_pressure_correction_norm": out_pressure_norm,
                "output_pressure_divergence_pre": out_div_pre,
                "output_pressure_divergence_post": out_div_post,
                "output_pressure_divergence_delta": out_div_post - out_div_pre,
                "output_boundary_mismatch_pre": out_bnd_pre,
                "output_boundary_mismatch_post": out_bnd_post,
            }
            return z_f_next, z_obj_next, z_obj_dyn_next, z_obj_geo_next, z_int_balanced, particle_next, diagnostics
        return z_f_next, z_obj_next, z_obj_dyn_next, z_obj_geo_next, z_int_balanced, particle_next

    @staticmethod
    def _attend(
        attn: nn.MultiheadAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # nn.MultiheadAttention in torch 1.8 expects [S,B,C].
        q = query.transpose(0, 1)
        k = key.transpose(0, 1)
        v = value.transpose(0, 1)
        out, _ = attn(q, k, v, key_padding_mask=key_padding_mask)
        return out.transpose(0, 1)
