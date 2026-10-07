from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .backbone import FluidUNOBranch3D
from .decoders import PhaseOccupancyHead3D, PressureHead3D, VelocityHead3D
from .rollout import BulkInterfaceResponseRolloutCore
from .solid_conditioning import build_interface_inputs_from_solid_state, solid_meta_at_step, solid_state_at_step


class BulkInterfaceResponse4DFlowPredictor(nn.Module):
    """
    Simpler rollout model:
    - bulk Eulerian update
    - localized interface response
    - one latent projection step
    """

    def __init__(
        self,
        fluid_input_channels: int,
        latent_channels: int = 32,
        control_dim: int = 0,
        control_channels: int = 8,
        base_width: int = 16,
        blocks_per_stage: int = 2,
        predict_phase: bool = False,
        num_phase_channels: int = 2,
        predict_pressure: bool = False,
        interface_hidden_channels: int = 32,
        interface_band_width: float = 0.35,
        pressure_correction_scale: float = 0.2,
        particle_step_scale: float = 1.0,
        use_projection: bool = True,
        use_object_patch_context: bool = True,
        global_interface_response: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.control_dim = int(max(0, control_dim))
        self.predict_phase = bool(predict_phase)
        self.predict_pressure = bool(predict_pressure)

        branch_control_channels = int(control_channels) if self.control_dim > 0 else 0
        self.fluid_branch = FluidUNOBranch3D(
            input_channels=int(fluid_input_channels),
            latent_channels=self.latent_channels,
            control_channels=branch_control_channels,
            base_width=base_width,
            blocks_per_stage=blocks_per_stage,
        )
        self.rollout_core = BulkInterfaceResponseRolloutCore(
            fluid_branch=self.fluid_branch,
            latent_channels=self.latent_channels,
            control_dim=self.control_dim,
            control_channels=branch_control_channels,
            interface_hidden_channels=int(interface_hidden_channels),
            interface_band_width=float(interface_band_width),
            pressure_correction_scale=float(pressure_correction_scale),
            particle_step_scale=float(particle_step_scale),
            use_projection=bool(use_projection),
            use_object_patch_context=bool(use_object_patch_context),
            global_interface_response=bool(global_interface_response),
        )
        self.velocity_head = VelocityHead3D(in_channels=self.latent_channels, out_channels=3)
        self.phase_head = (
            PhaseOccupancyHead3D(in_channels=self.latent_channels, num_phases=int(num_phase_channels))
            if self.predict_phase
            else None
        )
        self.pressure_head = PressureHead3D(in_channels=self.latent_channels) if self.predict_pressure else None

    def _control_at_step(
        self, control_sequence: Optional[torch.Tensor], step_idx: int
    ) -> Optional[torch.Tensor]:
        if control_sequence is None:
            return None
        if control_sequence.ndim == 2:
            return control_sequence
        if control_sequence.ndim == 3:
            if step_idx >= control_sequence.shape[1]:
                return control_sequence[:, -1, :]
            return control_sequence[:, step_idx, :]
        raise ValueError(
            f"control_sequence must be [B,C] or [B,T,C], got {tuple(control_sequence.shape)}"
        )

    def forward(
        self,
        fluid_init: torch.Tensor,
        solid_sequence: Dict[str, torch.Tensor] | None = None,
        object_inputs: Dict[str, torch.Tensor] | None = None,
        interface_inputs: Dict[str, torch.Tensor] | None = None,
        control_sequence: torch.Tensor | None = None,
        particle_pos_init: torch.Tensor | None = None,
        rollout_steps: int = 1,
        return_phase: bool = True,
        return_pressure: bool = False,
        return_latents: bool = False,
        return_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if fluid_init.ndim != 5:
            raise ValueError(f"fluid_init must be [B,C,D,H,W], got {tuple(fluid_init.shape)}")
        z_f = self.fluid_branch.encode(fluid_init)
        velocity_seq = []
        phase_seq = []
        pressure_seq = []
        diag_seq = []
        particle_seq = []
        p_state = particle_pos_init
        if p_state is not None:
            particle_seq.append(p_state)

        steps = int(max(1, rollout_steps))
        for t in range(steps):
            if solid_sequence is not None:
                solid_sdf_t, solid_velocity_t = solid_state_at_step(solid_sequence, t, like=z_f)
                solid_center_voxel_t, solid_radius_voxel_t = solid_meta_at_step(solid_sequence, t)
                step_object_inputs = None
                step_interface_inputs = build_interface_inputs_from_solid_state(
                    solid_sdf_t,
                    solid_velocity_t,
                    like=z_f,
                    solid_center_voxel=solid_center_voxel_t,
                    solid_radius_voxel=solid_radius_voxel_t,
                )
            else:
                step_object_inputs = object_inputs
                step_interface_inputs = interface_inputs

            c_t = self._control_at_step(control_sequence, t)
            if return_diagnostics:
                z_f, step_diag = self.rollout_core.step(
                    z_f=z_f,
                    object_inputs=step_object_inputs,
                    interface_inputs=step_interface_inputs,
                    control_vec=c_t,
                    return_diagnostics=True,
                )
            else:
                z_f = self.rollout_core.step(
                    z_f=z_f,
                    object_inputs=step_object_inputs,
                    interface_inputs=step_interface_inputs,
                    control_vec=c_t,
                    return_diagnostics=False,
                )
                step_diag = {}

            vel = self.velocity_head(z_f)
            velocity_seq.append(vel)
            if return_phase and self.phase_head is not None:
                phase_seq.append(self.phase_head(z_f))
            if return_pressure and self.pressure_head is not None:
                pressure_seq.append(self.pressure_head(z_f))

            if p_state is not None:
                p_state, disp = self.rollout_core.advect_particles(
                    particle_pos=p_state,
                    velocity_field=vel,
                    step_scale=self.rollout_core.particle_step_scale,
                )
                particle_seq.append(p_state)
                if return_diagnostics:
                    step_diag["dual_particle_step_norm"] = torch.mean(torch.abs(disp), dim=(1, 2))

            if return_diagnostics:
                diag_seq.append(step_diag)

        out: Dict[str, torch.Tensor] = {
            "velocity": torch.stack(velocity_seq, dim=1),
        }
        if phase_seq:
            out["phase"] = torch.stack(phase_seq, dim=1)
        if pressure_seq:
            out["pressure"] = torch.stack(pressure_seq, dim=1)
        if particle_seq:
            out["particle_pos"] = torch.stack(particle_seq, dim=1)
        if return_diagnostics and diag_seq:
            out["diagnostics"] = {
                k: torch.stack([d[k] for d in diag_seq], dim=1) for k in diag_seq[0].keys()
            }
        if return_latents:
            out["z_f_last"] = z_f
        return out
