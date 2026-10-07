from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


_GRID_CACHE: Dict[Tuple[str, int, int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _take_step(tensor: Optional[torch.Tensor], step_idx: int) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.ndim == 3:
        if tensor.shape[1] == 0:
            return None
        idx = min(int(step_idx), int(tensor.shape[1]) - 1)
        return tensor[:, idx]
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 6:
        if tensor.shape[1] == 0:
            return None
        idx = min(int(step_idx), int(tensor.shape[1]) - 1)
        return tensor[:, idx]
    if tensor.ndim == 5:
        return tensor
    if tensor.ndim == 4:
        return tensor[:, None]
    raise ValueError(f"solid-sequence tensor must be 4D/5D/6D, got {tuple(tensor.shape)}")


def edge_map_3d(mask_b1dhw: torch.Tensor) -> torch.Tensor:
    x = (mask_b1dhw > 0.5).to(dtype=mask_b1dhw.dtype)
    e = torch.zeros_like(x)

    d_diff = torch.abs(x[:, :, 1:] - x[:, :, :-1])
    e[:, :, :-1] = torch.maximum(e[:, :, :-1], d_diff)
    e[:, :, 1:] = torch.maximum(e[:, :, 1:], d_diff)

    h_diff = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    e[:, :, :, :-1] = torch.maximum(e[:, :, :, :-1], h_diff)
    e[:, :, :, 1:] = torch.maximum(e[:, :, :, 1:], h_diff)

    w_diff = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
    e[:, :, :, :, :-1] = torch.maximum(e[:, :, :, :, :-1], w_diff)
    e[:, :, :, :, 1:] = torch.maximum(e[:, :, :, :, 1:], w_diff)
    return e


def safe_gradient_3d(sdf_b1dhw: torch.Tensor) -> torch.Tensor:
    zpad = F.pad(sdf_b1dhw, (0, 0, 0, 0, 1, 1), mode="replicate")
    ypad = F.pad(sdf_b1dhw, (0, 0, 1, 1, 0, 0), mode="replicate")
    xpad = F.pad(sdf_b1dhw, (1, 1, 0, 0, 0, 0), mode="replicate")
    dz = 0.5 * (zpad[:, :, 2:] - zpad[:, :, :-2])
    dy = 0.5 * (ypad[:, :, :, 2:] - ypad[:, :, :, :-2])
    dx = 0.5 * (xpad[:, :, :, :, 2:] - xpad[:, :, :, :, :-2])
    return torch.cat([dx, dy, dz], dim=1)


def solid_state_at_step(
    solid_sequence: Optional[Dict[str, torch.Tensor]],
    step_idx: int,
    *,
    like: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if solid_sequence is None:
        return None, None

    solid_sdf = _take_step(solid_sequence.get("solid_sdf"), step_idx)
    solid_velocity = _take_step(solid_sequence.get("solid_velocity"), step_idx)

    if like is not None:
        if solid_sdf is None:
            solid_sdf = torch.zeros(
                like.shape[0], 1, like.shape[2], like.shape[3], like.shape[4], dtype=like.dtype, device=like.device
            )
        if solid_velocity is None:
            solid_velocity = torch.zeros(
                like.shape[0], 3, like.shape[2], like.shape[3], like.shape[4], dtype=like.dtype, device=like.device
            )

    return solid_sdf, solid_velocity


def solid_meta_at_step(
    solid_sequence: Optional[Dict[str, torch.Tensor]],
    step_idx: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if solid_sequence is None:
        return None, None
    solid_center_voxel = _take_step(solid_sequence.get("solid_center_voxel"), step_idx)
    solid_radius_voxel = _take_step(solid_sequence.get("solid_radius_voxel"), step_idx)
    return solid_center_voxel, solid_radius_voxel


def _coord_grid(
    like_b1dhw: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d, h, w = int(like_b1dhw.shape[2]), int(like_b1dhw.shape[3]), int(like_b1dhw.shape[4])
    dev = like_b1dhw.device
    key = (str(dev), d, h, w)
    cached = _GRID_CACHE.get(key)
    if cached is None:
        z = torch.arange(d, device=dev, dtype=torch.float32).view(1, 1, d, 1, 1)
        y = torch.arange(h, device=dev, dtype=torch.float32).view(1, 1, 1, h, 1)
        x = torch.arange(w, device=dev, dtype=torch.float32).view(1, 1, 1, 1, w)
        cached = (x, y, z)
        _GRID_CACHE[key] = cached
    x, y, z = cached
    return x.to(dtype=like_b1dhw.dtype), y.to(dtype=like_b1dhw.dtype), z.to(dtype=like_b1dhw.dtype)


def _normalized_distance_from_sphere(
    solid_sdf: torch.Tensor,
    solid_center_voxel: Optional[torch.Tensor],
    solid_radius_voxel: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if solid_center_voxel is None or solid_radius_voxel is None:
        return None
    if solid_center_voxel.ndim != 2 or solid_center_voxel.shape[-1] != 3:
        return None
    if solid_radius_voxel.ndim == 1:
        radius = solid_radius_voxel[:, None]
    elif solid_radius_voxel.ndim == 2 and solid_radius_voxel.shape[-1] == 1:
        radius = solid_radius_voxel
    else:
        return None

    x, y, z = _coord_grid(solid_sdf)
    cx = solid_center_voxel[:, 0].to(device=solid_sdf.device, dtype=solid_sdf.dtype).view(-1, 1, 1, 1, 1)
    cy = solid_center_voxel[:, 1].to(device=solid_sdf.device, dtype=solid_sdf.dtype).view(-1, 1, 1, 1, 1)
    cz = solid_center_voxel[:, 2].to(device=solid_sdf.device, dtype=solid_sdf.dtype).view(-1, 1, 1, 1, 1)
    radius_t = radius.to(device=solid_sdf.device, dtype=solid_sdf.dtype).view(-1, 1, 1, 1, 1).clamp_min(1.0)
    signed_dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 + 1e-8) - radius_t
    return torch.abs(signed_dist) / radius_t


def _normalized_distance_fallback(solid_sdf: torch.Tensor) -> torch.Tensor:
    abs_sdf = torch.abs(solid_sdf)
    scale = abs_sdf.flatten(1).amax(dim=1).view(-1, 1, 1, 1, 1).clamp_min(1.0)
    return abs_sdf / scale


def _build_near_basis(
    distance_norm: torch.Tensor,
    basis_scales: Sequence[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    scales = [float(max(1e-4, s)) for s in basis_scales]
    basis = [torch.exp(-distance_norm / s) for s in scales]
    near_basis = torch.cat(basis, dim=1)
    near_weight = torch.mean(near_basis, dim=1, keepdim=True)
    return near_weight.clamp(0.0, 1.0), near_basis.clamp(0.0, 1.0)


def build_interface_inputs_from_solid_state(
    solid_sdf: Optional[torch.Tensor],
    solid_velocity: Optional[torch.Tensor],
    *,
    like: Optional[torch.Tensor] = None,
    solid_center_voxel: Optional[torch.Tensor] = None,
    solid_radius_voxel: Optional[torch.Tensor] = None,
    near_basis_scales: Sequence[float] = (0.10, 0.25, 0.50),
    use_size_normalization: bool = True,
) -> Optional[Dict[str, torch.Tensor]]:
    if solid_sdf is None and solid_velocity is None:
        return None

    if solid_sdf is None:
        if like is None:
            raise ValueError("like tensor is required when solid_sdf is missing.")
        solid_sdf = torch.zeros(
            like.shape[0], 1, like.shape[2], like.shape[3], like.shape[4], dtype=like.dtype, device=like.device
        )
    if solid_velocity is None:
        solid_velocity = torch.zeros(
            solid_sdf.shape[0],
            3,
            solid_sdf.shape[2],
            solid_sdf.shape[3],
            solid_sdf.shape[4],
            dtype=solid_sdf.dtype,
            device=solid_sdf.device,
        )

    solid_sdf = solid_sdf.float()
    solid_velocity = solid_velocity.float()
    solid_mask = (solid_sdf <= 0.0).to(dtype=solid_sdf.dtype)
    fluid_solid_interface = edge_map_3d(solid_mask)
    interface_normal = safe_gradient_3d(solid_sdf)
    distance_norm = None
    if use_size_normalization:
        distance_norm = _normalized_distance_from_sphere(solid_sdf, solid_center_voxel, solid_radius_voxel)
    if distance_norm is None:
        distance_norm = _normalized_distance_fallback(solid_sdf)
    interface_near_weight, interface_near_basis = _build_near_basis(distance_norm, near_basis_scales)
    interface_near_weight = torch.maximum(interface_near_weight, fluid_solid_interface)

    return {
        "fluid_solid_interface": fluid_solid_interface,
        "liquid_liquid_interface": torch.zeros_like(fluid_solid_interface),
        "interface_sdf": solid_sdf,
        "interface_normal": interface_normal,
        "boundary_velocity": solid_velocity,
        "boundary_mask": solid_mask,
        "interface_distance_norm": distance_norm,
        "interface_near_weight": interface_near_weight,
        "interface_near_basis": interface_near_basis,
    }
