from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .backbone import FluidUNOBranch3D
from .decoders import PhaseOccupancyHead3D, PressureHead3D, VelocityHead3D
from .encoders import InterfaceBranchEncoder3D, ObjectBranchEncoder
from .rollout import LatentCouplingRolloutCore
from .solid_conditioning import build_interface_inputs_from_solid_state, solid_meta_at_step, solid_state_at_step


class InteractionAwareHybrid4DFlowPredictor(nn.Module):
    """
    Interaction-Aware Hybrid 4D Flow Predictor

    Latent state at step t:
    - z_f^t: fluid latent grid
    - z_obj^t: object tokens
    - z_int^t: interface tokens
    """

    def __init__(
        self,
        fluid_input_channels: int,
        latent_channels: int = 32,
        control_dim: int = 0,
        control_channels: int = 8,
        base_width: int = 16,
        blocks_per_stage: int = 2,
        num_object_types: int = 3,
        interface_input_channels: int = 10,
        predict_phase: bool = True,
        num_phase_channels: int = 2,
        predict_pressure: bool = False,
        coupling_variant: str = "strict_mediator",
        direct_obj_to_fluid_scale: float = 0.0,
        direct_fluid_to_obj_scale: float = 0.0,
        exchange_reaction_scale: float = 1.0,
        geo_interface_condition_scale: float = 1.0,
        geo_operator_init_scale: float = 0.1,
        object_geometry_stream_scale: float = 1.0,
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
        interface_near_basis_scales: tuple[float, float, float] = (0.10, 0.25, 0.50),
        interface_near_learned_refine_enabled: bool = True,
        interface_near_use_size_normalization: bool = True,
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
        self.object_encoder = ObjectBranchEncoder(
            token_dim=self.latent_channels,
            num_object_types=num_object_types,
            geometry_stream_scale=float(object_geometry_stream_scale),
        )
        self.interface_encoder = InterfaceBranchEncoder3D(
            token_dim=self.latent_channels,
            input_channels=interface_input_channels,
        )
        self.rollout_core = LatentCouplingRolloutCore(
            fluid_branch=self.fluid_branch,
            latent_channels=self.latent_channels,
            control_dim=self.control_dim,
            control_channels=branch_control_channels,
            coupling_variant=coupling_variant,
            direct_obj_to_fluid_scale=float(direct_obj_to_fluid_scale),
            direct_fluid_to_obj_scale=float(direct_fluid_to_obj_scale),
            exchange_reaction_scale=float(exchange_reaction_scale),
            geo_interface_condition_scale=float(geo_interface_condition_scale),
            geo_operator_init_scale=float(geo_operator_init_scale),
            pressure_correction_scale=float(pressure_correction_scale),
            pressure_post_p2g_enabled=bool(pressure_post_p2g_enabled),
            pressure_post_p2g_scale=float(pressure_post_p2g_scale),
            output_pressure_correction_enabled=bool(output_pressure_correction_enabled),
            output_pressure_correction_scale=float(output_pressure_correction_scale),
            output_pressure_feedback_scale=float(output_pressure_feedback_scale),
            dual_coupling_enabled=bool(dual_coupling_enabled),
            dual_particle_step_scale=float(dual_particle_step_scale),
            dual_p2g_scale=float(dual_p2g_scale),
            interface_near_aware_enabled=bool(interface_near_aware_enabled),
            interface_near_gate_strength=float(interface_near_gate_strength),
            interface_near_basis_scales=tuple(float(x) for x in interface_near_basis_scales),
            interface_near_learned_refine_enabled=bool(interface_near_learned_refine_enabled),
        )
        self.interface_near_use_size_normalization = bool(interface_near_use_size_normalization)

        self.velocity_head = VelocityHead3D(in_channels=self.latent_channels, out_channels=3)
        self.phase_head = (
            PhaseOccupancyHead3D(in_channels=self.latent_channels, num_phases=int(num_phase_channels))
            if self.predict_phase
            else None
        )
        self.pressure_head = (
            PressureHead3D(in_channels=self.latent_channels) if self.predict_pressure else None
        )

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
        bsz = fluid_init.shape[0]
        device = fluid_init.device
        steps = int(max(1, rollout_steps))

        z_f = self.fluid_branch.encode(fluid_init)
        legacy_object_inputs = object_inputs if solid_sequence is None else None
        obj_enc_out = self.object_encoder(
            object_inputs=legacy_object_inputs,
            batch_size=bsz,
            device=device,
            return_components=True,
        )
        z_obj, obj_mask, obj_components = obj_enc_out
        z_obj_dyn = obj_components["dyn"]
        z_obj_geo = obj_components["geo"]
        z_obj_joint = obj_components.get("joint", z_obj)
        z_obj = z_obj_joint
        if solid_sequence is None:
            z_int_state, int_mask_state = self.interface_encoder(
                interface_inputs=interface_inputs,
                batch_size=bsz,
                device=device,
            )
        else:
            z_int_state = None
            int_mask_state = None

        velocity_seq = []
        phase_seq = []
        pressure_seq = []
        diag_seq = []
        particle_seq = []
        p_state = particle_pos_init
        if p_state is not None:
            particle_seq.append(p_state)

        for t in range(steps):
            if solid_sequence is not None:
                solid_sdf_t, solid_velocity_t = solid_state_at_step(solid_sequence, t, like=z_f)
                solid_center_voxel_t, solid_radius_voxel_t = solid_meta_at_step(solid_sequence, t)
                step_interface_inputs = build_interface_inputs_from_solid_state(
                    solid_sdf_t,
                    solid_velocity_t,
                    like=z_f,
                    solid_center_voxel=solid_center_voxel_t,
                    solid_radius_voxel=solid_radius_voxel_t,
                    use_size_normalization=self.interface_near_use_size_normalization,
                )
                z_int_cond, step_int_mask = self.interface_encoder(
                    interface_inputs=step_interface_inputs,
                    batch_size=bsz,
                    device=device,
                )
                if z_int_state is None:
                    z_int_in = z_int_cond
                else:
                    z_int_in = 0.5 * (z_int_state + z_int_cond)
            else:
                step_interface_inputs = interface_inputs
                step_int_mask = int_mask_state
                z_int_in = z_int_state

            c_t = self._control_at_step(control_sequence, t)
            step_out = self.rollout_core.step(
                z_f=z_f,
                z_obj=z_obj,
                z_int=z_int_in,
                z_obj_dyn=z_obj_dyn,
                z_obj_geo=z_obj_geo,
                particle_pos=p_state,
                object_mask=obj_mask,
                interface_mask=step_int_mask,
                interface_grid=step_interface_inputs,
                control_vec=c_t,
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                z_f, z_obj, z_obj_dyn, z_obj_geo, z_int_state, p_state, step_diag = step_out
                diag_seq.append(step_diag)
            else:
                z_f, z_obj, z_obj_dyn, z_obj_geo, z_int_state, p_state = step_out
            if p_state is not None:
                particle_seq.append(p_state)

            velocity_seq.append(self.velocity_head(z_f))
            if return_phase and self.phase_head is not None:
                phase_seq.append(self.phase_head(z_f))
            if return_pressure and self.pressure_head is not None:
                pressure_seq.append(self.pressure_head(z_f))

        out: Dict[str, torch.Tensor] = {
            "velocity": torch.stack(velocity_seq, dim=1),  # [B,T,3,D,H,W]
        }
        if phase_seq:
            out["phase"] = torch.stack(phase_seq, dim=1)  # [B,T,P,D,H,W]
        if pressure_seq:
            out["pressure"] = torch.stack(pressure_seq, dim=1)  # [B,T,1,D,H,W]
        if return_diagnostics and diag_seq:
            out["diagnostics"] = {
                k: torch.stack([d[k] for d in diag_seq], dim=1) for k in diag_seq[0].keys()
            }
        if particle_seq:
            out["particle_pos"] = torch.stack(particle_seq, dim=1)  # [B,T+1,P,3]
        if return_latents:
            out["z_f_last"] = z_f
            out["z_obj_last"] = z_obj
            out["z_obj_dyn_last"] = z_obj_dyn
            out["z_obj_geo_last"] = z_obj_geo
            out["z_obj_joint_last"] = z_obj
            out["z_int_last"] = z_int_state
        return out
