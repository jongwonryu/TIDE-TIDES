from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .latent_pressure_correction import LatentPressureCorrectionOperator


def _group_norm_channels(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class InterfaceResponseOperator3D(nn.Module):
    """
    Predict a localized latent forcing field near the interface band.

    The operator reads the current fluid latent and grid-aligned interface cues,
    then gates the response with a narrow-band mask derived from the object SDF.
    """

    def __init__(
        self,
        latent_channels: int,
        interface_feature_channels: int = 10,
        hidden_channels: int = 32,
        object_patch_channels: int = 8,
        band_width: float = 0.35,
        use_object_patch_context: bool = True,
        global_interface_response: bool = False,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.interface_feature_channels = int(interface_feature_channels)
        self.hidden_channels = int(max(8, hidden_channels))
        self.object_patch_channels = int(max(1, object_patch_channels))
        self.band_width = float(max(1e-4, band_width))
        self.use_object_patch_context = bool(use_object_patch_context)
        self.global_interface_response = bool(global_interface_response)

        in_channels = self.latent_channels + self.interface_feature_channels
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, self.hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_norm_channels(self.hidden_channels), self.hidden_channels),
            nn.GELU(),
            nn.Conv3d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_norm_channels(self.hidden_channels), self.hidden_channels),
            nn.GELU(),
        )
        self.out = nn.Conv3d(self.hidden_channels, self.latent_channels, kernel_size=1, bias=False)

        if self.use_object_patch_context:
            self.object_patch_encoder = nn.Sequential(
                nn.Conv3d(self.object_patch_channels, self.hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_group_norm_channels(self.hidden_channels), self.hidden_channels),
                nn.GELU(),
                nn.Conv3d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_group_norm_channels(self.hidden_channels), self.hidden_channels),
                nn.GELU(),
                nn.AdaptiveAvgPool3d(1),
            )
            self.object_to_gamma = nn.Linear(self.hidden_channels, self.hidden_channels)
            self.object_to_beta = nn.Linear(self.hidden_channels, self.hidden_channels)
        else:
            self.object_patch_encoder = None
            self.object_to_gamma = None
            self.object_to_beta = None

    @staticmethod
    def _zeros_like_grid(like: torch.Tensor, channels: int) -> torch.Tensor:
        return torch.zeros(
            like.shape[0],
            int(channels),
            like.shape[2],
            like.shape[3],
            like.shape[4],
            dtype=like.dtype,
            device=like.device,
        )

    @staticmethod
    def _fit_channels(x: torch.Tensor, channels: int) -> torch.Tensor:
        c = x.shape[1]
        if c == channels:
            return x
        if c > channels:
            return x[:, :channels]
        pad = torch.zeros(
            x.shape[0],
            channels - c,
            x.shape[2],
            x.shape[3],
            x.shape[4],
            dtype=x.dtype,
            device=x.device,
        )
        return torch.cat([x, pad], dim=1)

    def _collect_interface_features(
        self,
        interface_inputs: Optional[Dict[str, torch.Tensor]],
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if interface_inputs is None:
            zeros = self._zeros_like_grid(like, self.interface_feature_channels)
            band_mask = self._zeros_like_grid(like, 1)
            sdf = self._zeros_like_grid(like, 1)
            return zeros, band_mask, sdf

        keys = (
            "fluid_solid_interface",
            "interface_sdf",
            "interface_normal",
            "boundary_velocity",
            "boundary_mask",
        )
        feats = []
        for key in keys:
            value = interface_inputs.get(key, None)
            if value is None:
                continue
            v = value.to(device=like.device, dtype=like.dtype)
            if v.ndim == 4:
                v = v[:, None]
            feats.append(v)
        if feats:
            x = torch.cat(feats, dim=1)
            x = self._fit_channels(x, self.interface_feature_channels)
        else:
            x = self._zeros_like_grid(like, self.interface_feature_channels)

        sdf = interface_inputs.get("interface_sdf", None)
        if sdf is None:
            sdf_t = self._zeros_like_grid(like, 1)
        else:
            sdf_t = sdf.to(device=like.device, dtype=like.dtype)
            if sdf_t.ndim == 4:
                sdf_t = sdf_t[:, None]

        boundary_mask = interface_inputs.get("boundary_mask", None)
        if boundary_mask is None:
            boundary_mask_t = self._zeros_like_grid(like, 1)
        else:
            boundary_mask_t = boundary_mask.to(device=like.device, dtype=like.dtype)
            if boundary_mask_t.ndim == 4:
                boundary_mask_t = boundary_mask_t[:, None]

        if self.global_interface_response:
            band_mask = torch.ones_like(sdf_t)
        else:
            # Soft narrow-band around the interface plus hard boundary support.
            band_mask = torch.exp(-torch.abs(sdf_t) / self.band_width)
            band_mask = torch.maximum(band_mask, boundary_mask_t.clamp(0.0, 1.0))
        return x, band_mask.clamp(0.0, 1.0), sdf_t

    def _collect_object_context(
        self,
        object_inputs: Optional[Dict[str, torch.Tensor]],
        like: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.use_object_patch_context or object_inputs is None:
            return None

        patch_keys = (
            "object_sdf_patch",
            "object_interface_patch",
            "object_interface_normal_patch",
            "object_boundary_velocity_patch",
        )
        patches = []
        for key in patch_keys:
            v = object_inputs.get(key, None)
            if v is None:
                continue
            t = v.to(device=like.device, dtype=like.dtype)
            if t.ndim != 6:
                continue
            patches.append(t.float())
        if not patches:
            return None

        x = torch.cat(patches, dim=2)  # [B,N,C,P,P,P]
        x = x[:, 0]  # first object only for now
        x = self._fit_channels(x, self.object_patch_channels)
        ctx = self.object_patch_encoder(x).flatten(1)
        return ctx

    def forward(
        self,
        z_f: torch.Tensor,
        interface_inputs: Optional[Dict[str, torch.Tensor]],
        object_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        interface_feats, band_mask, sdf = self._collect_interface_features(interface_inputs, z_f)
        x = torch.cat([z_f, interface_feats], dim=1)
        h = self.stem(x)

        object_ctx = self._collect_object_context(object_inputs, z_f)
        if object_ctx is not None:
            gamma = self.object_to_gamma(object_ctx)[:, :, None, None, None]
            beta = self.object_to_beta(object_ctx)[:, :, None, None, None]
            h = h * (1.0 + torch.tanh(gamma)) + beta

        delta = self.out(h) * band_mask
        diagnostics = {
            "interaction_intensity_mean": torch.mean(torch.abs(delta), dim=(1, 2, 3, 4)),
            "obj_geo_norm": torch.mean(torch.abs(delta), dim=(1, 2, 3, 4)),
            "obj_geo_gate_mean": torch.mean(band_mask, dim=(1, 2, 3, 4)),
            "obj_geo_effect_ratio": torch.mean(band_mask, dim=(1, 2, 3, 4)),
            "interface_band_mean": torch.mean(band_mask, dim=(1, 2, 3, 4)),
            "interface_sdf_abs_mean": torch.mean(torch.abs(sdf), dim=(1, 2, 3, 4)),
        }
        return delta, diagnostics


class BulkInterfaceResponseRolloutCore(nn.Module):
    def __init__(
        self,
        fluid_branch: nn.Module,
        latent_channels: int,
        control_dim: int = 0,
        control_channels: int = 0,
        interface_hidden_channels: int = 32,
        interface_band_width: float = 0.35,
        pressure_correction_scale: float = 0.2,
        particle_step_scale: float = 1.0,
        use_projection: bool = True,
        use_object_patch_context: bool = True,
        global_interface_response: bool = False,
    ) -> None:
        super().__init__()
        self.fluid_branch = fluid_branch
        self.latent_channels = int(latent_channels)
        self.control_dim = int(max(0, control_dim))
        self.control_channels = int(max(0, control_channels))
        self.particle_step_scale = float(particle_step_scale)
        self.use_projection = bool(use_projection)

        self.interface_operator = InterfaceResponseOperator3D(
            latent_channels=self.latent_channels,
            hidden_channels=int(interface_hidden_channels),
            band_width=float(interface_band_width),
            use_object_patch_context=bool(use_object_patch_context),
            global_interface_response=bool(global_interface_response),
        )
        self.post_norm = nn.GroupNorm(_group_norm_channels(self.latent_channels), self.latent_channels)

        if self.use_projection:
            self.pressure_operator = LatentPressureCorrectionOperator(
                latent_channels=self.latent_channels,
                correction_scale=float(pressure_correction_scale),
            )
        else:
            self.pressure_operator = None

        if self.control_dim > 0:
            self.control_to_grid = nn.Linear(self.control_dim, self.control_channels)
        else:
            self.control_to_grid = None

    def _control_grid(self, control_vec: Optional[torch.Tensor], z_f: torch.Tensor) -> Optional[torch.Tensor]:
        if self.control_dim <= 0 or self.control_channels <= 0:
            return None
        bsz = z_f.shape[0]
        if control_vec is None:
            c = torch.zeros(bsz, self.control_dim, device=z_f.device, dtype=z_f.dtype)
        else:
            c = control_vec.to(device=z_f.device, dtype=z_f.dtype)
        g = self.control_to_grid(c)[:, :, None, None, None]
        return g.expand(bsz, self.control_channels, z_f.shape[-3], z_f.shape[-2], z_f.shape[-1])

    @staticmethod
    def sample_at_points(field_bcdhw: torch.Tensor, pts_bnx: torch.Tensor) -> torch.Tensor:
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
        return sampled[:, :, :, 0, 0].transpose(1, 2)

    @classmethod
    def advect_particles(
        cls,
        particle_pos: torch.Tensor,
        velocity_field: torch.Tensor,
        step_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, d, h, w = velocity_field.shape
        disp = cls.sample_at_points(velocity_field, particle_pos) * float(step_scale)
        p_next = particle_pos + disp
        p_next[..., 0].clamp_(0.0, float(max(1, w - 1)))
        p_next[..., 1].clamp_(0.0, float(max(1, h - 1)))
        p_next[..., 2].clamp_(0.0, float(max(1, d - 1)))
        return p_next, disp

    @staticmethod
    def divergence_mean_abs(vel: torch.Tensor) -> torch.Tensor:
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

    def step(
        self,
        z_f: torch.Tensor,
        object_inputs: Optional[Dict[str, torch.Tensor]] = None,
        interface_inputs: Optional[Dict[str, torch.Tensor]] = None,
        control_vec: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]] | torch.Tensor:
        control_grid = self._control_grid(control_vec, z_f)
        delta_bulk = self.fluid_branch.self_update(z_f, control_grid=control_grid)
        delta_interface, interface_diag = self.interface_operator(
            z_f=z_f,
            interface_inputs=interface_inputs,
            object_inputs=object_inputs,
        )
        z_tilde = z_f + delta_bulk + delta_interface

        pressure_diag = {}
        if self.pressure_operator is not None:
            if return_diagnostics:
                z_corr, pressure_diag = self.pressure_operator(z_tilde, return_diagnostics=True)
            else:
                z_corr = self.pressure_operator(z_tilde, return_diagnostics=False)
        else:
            z_corr = z_tilde

        z_next = F.gelu(self.post_norm(z_corr))

        if not return_diagnostics:
            return z_next

        zero = torch.zeros(z_f.shape[0], dtype=z_f.dtype, device=z_f.device)
        diagnostics = {
            "interaction_intensity_mean": interface_diag["interaction_intensity_mean"],
            "exchange_balance_error": zero,
            "pressure_correction_norm": pressure_diag.get("pressure_correction_norm", zero),
            "pressure_divergence_pre": pressure_diag.get("pressure_divergence_pre", zero),
            "pressure_divergence_post": pressure_diag.get("pressure_divergence_post", zero),
            "pressure_stage1_correction_norm": pressure_diag.get("pressure_correction_norm", zero),
            "pressure_stage1_divergence_pre": pressure_diag.get("pressure_divergence_pre", zero),
            "pressure_stage1_divergence_post": pressure_diag.get("pressure_divergence_post", zero),
            "pressure_stage2_correction_norm": zero,
            "pressure_stage2_divergence_pre": pressure_diag.get("pressure_divergence_post", zero),
            "pressure_stage2_divergence_post": pressure_diag.get("pressure_divergence_post", zero),
            "pressure_stage2_divergence_delta": zero,
            "output_pressure_correction_norm": zero,
            "output_pressure_divergence_pre": zero,
            "output_pressure_divergence_post": zero,
            "output_pressure_divergence_delta": zero,
            "output_boundary_mismatch_pre": zero,
            "output_boundary_mismatch_post": zero,
            "dual_particle_step_norm": zero,
            "dual_p2g_norm": zero,
            "obj_dyn_norm": zero,
            "obj_geo_norm": interface_diag["obj_geo_norm"],
            "obj_geo_dyn_ratio": zero,
            "obj_geo_gate_mean": interface_diag["obj_geo_gate_mean"],
            "obj_geo_to_int_raw_norm": interface_diag["obj_geo_norm"],
            "obj_geo_to_int_gated_norm": interface_diag["obj_geo_norm"],
            "obj_geo_effect_ratio": interface_diag["obj_geo_effect_ratio"],
            "direct_obj_delta_norm": zero,
        }
        return z_next, diagnostics
