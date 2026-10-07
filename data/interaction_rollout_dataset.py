from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class EpisodeEntry:
    episode_id: str
    split: str
    npz_path: Path
    sdf_path: Optional[Path]
    surface_state_path: Optional[Path]
    num_frames: int
    solid_size_voxel: float


def _edge_map_3d(mask_dhw: np.ndarray) -> np.ndarray:
    x = (mask_dhw > 0.5).astype(np.float32)
    e = np.zeros_like(x, dtype=np.float32)
    d_diff = np.abs(np.diff(x, axis=0))
    e[:-1] = np.maximum(e[:-1], d_diff)
    e[1:] = np.maximum(e[1:], d_diff)
    h_diff = np.abs(np.diff(x, axis=1))
    e[:, :-1] = np.maximum(e[:, :-1], h_diff)
    e[:, 1:] = np.maximum(e[:, 1:], h_diff)
    w_diff = np.abs(np.diff(x, axis=2))
    e[:, :, :-1] = np.maximum(e[:, :, :-1], w_diff)
    e[:, :, 1:] = np.maximum(e[:, :, 1:], w_diff)
    return e


def _safe_gradient_3d(sdf_dhw: np.ndarray) -> np.ndarray:
    if sdf_dhw.size == 0:
        return np.zeros((3, *sdf_dhw.shape), dtype=np.float32)
    dz, dy, dx = np.gradient(sdf_dhw.astype(np.float32), axis=(0, 1, 2))
    return np.stack([dx, dy, dz], axis=0).astype(np.float32)


class InteractionRolloutDataset(Dataset):
    """
    Dataset for autoregressive training of InteractionAwareHybrid4DFlowPredictor.
    Includes particle GT trajectories sampled from water_surface_state.npz.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        manifest_path: str | Path | None = None,
        split: str = "train",
        valid_status: Sequence[str] = ("ok", "skipped_existing"),
        rollout_steps: int = 8,
        control_dim: int = 16,
        velocity_scale: float = 1.0,
        max_episodes: int = 0,
        max_episode_cache: int = 1,
        particle_gt_num_points: int = 256,
        load_particle_gt: bool = True,
        precompute_derived_cache: bool = True,
        object_patch_size: int = 16,
        canonical_object_patches: bool = True,
        include_gas_channel: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        if manifest_path is None:
            self.manifest_path = self.dataset_root / "manifest.jsonl"
        else:
            p = Path(manifest_path)
            self.manifest_path = p if p.is_absolute() else (self.dataset_root / p)

        self.split = str(split)
        self.valid_status = set(str(x) for x in valid_status)
        self.rollout_steps = int(max(1, rollout_steps))
        self.control_dim = int(max(0, control_dim))
        self.velocity_scale = float(max(1e-8, velocity_scale))
        self.max_episode_cache = int(max(1, max_episode_cache))
        self.particle_gt_num_points = int(max(1, particle_gt_num_points))
        self.load_particle_gt = bool(load_particle_gt)
        self.precompute_derived_cache = bool(precompute_derived_cache)
        self.object_patch_size = int(max(4, object_patch_size))
        self.canonical_object_patches = bool(canonical_object_patches)
        self.include_gas_channel = bool(include_gas_channel)

        self.sim_root = self.dataset_root.parents[1]
        self.output_root = self.sim_root / "output"

        self.episodes = self._load_episodes(max_episodes=max_episodes)
        self.indices = self._build_indices()
        self._cache: OrderedDict[int, Dict[str, np.ndarray]] = OrderedDict()

    def _normalize_existing_path(self, raw: object) -> Optional[Path]:
        if raw is None:
            return None
        s = str(raw)
        candidates = [s]

        for cand in candidates:
            p = Path(cand)
            if not p.is_absolute():
                p = self.dataset_root / p
            if p.exists():
                return p.resolve()
        return None

    def _resolve_surface_state_path(self, row: Dict[str, object], case_id: str) -> Optional[Path]:
        direct = self._normalize_existing_path(row.get("surface_state"))
        if direct is not None:
            return direct

        fallback = self.output_root / case_id / "sim" / "water_surface_state.npz"
        if fallback.exists():
            return fallback.resolve()
        return None

    def _load_episodes(self, max_episodes: int) -> List[EpisodeEntry]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {self.manifest_path}")

        rows: List[Dict[str, object]] = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

        out: List[EpisodeEntry] = []
        for row in rows:
            split = str(row.get("split", ""))
            status = str(row.get("status", ""))
            rel_file = row.get("file")
            if split != self.split:
                continue
            if status and status not in self.valid_status:
                continue
            if not rel_file:
                continue

            npz_path = self.dataset_root / str(rel_file)
            if not npz_path.exists():
                continue

            sdf_rel = row.get("interaction_sdf_path") or row.get("cond_sdf_path")
            sdf_path = (self.dataset_root / str(sdf_rel)) if sdf_rel else None
            if sdf_path is not None and not sdf_path.exists():
                sdf_path = None

            num_frames = int(row.get("num_frames", 0))
            if num_frames <= 1:
                with np.load(npz_path) as data:
                    num_frames = int(data["target_flow"].shape[0])
            if num_frames <= (self.rollout_steps + 1):
                continue

            case_id = str(row.get("case_id") or row.get("episode_id") or npz_path.stem)
            episode_id = str(row.get("episode_id") or case_id)
            surface_state_path = self._resolve_surface_state_path(row, case_id=case_id)
            solid_size = float(row.get("solid_size_voxel", 0.0) or 0.0)

            out.append(
                EpisodeEntry(
                    episode_id=episode_id,
                    split=split,
                    npz_path=npz_path,
                    sdf_path=sdf_path,
                    surface_state_path=surface_state_path,
                    num_frames=num_frames,
                    solid_size_voxel=solid_size,
                )
            )

        if int(max_episodes) > 0:
            out = out[: int(max_episodes)]
        if not out:
            raise RuntimeError(f"No episodes loaded for split={self.split} from {self.manifest_path}.")
        return out

    def _build_indices(self) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for epi_idx, epi in enumerate(self.episodes):
            max_t0 = epi.num_frames - self.rollout_steps - 1
            for t0 in range(max_t0 + 1):
                out.append((epi_idx, t0))
        if not out:
            raise RuntimeError("No rollout indices built; reduce rollout_steps.")
        return out

    def _fallback_sdf(self, cond_volume: np.ndarray) -> np.ndarray:
        occ = cond_volume[:, 0].astype(np.float32)
        return (1.0 - 2.0 * occ)[:, None].astype(np.float32)

    def _load_episode(self, epi_idx: int) -> Dict[str, np.ndarray]:
        if epi_idx in self._cache:
            self._cache.move_to_end(epi_idx)
            return self._cache[epi_idx]

        epi = self.episodes[epi_idx]
        with np.load(epi.npz_path) as data:
            target_flow = data["target_flow"].astype(np.float32)  # [F,3,D,H,W]
            phase = data["phase_onehot"].astype(np.float32)  # [F,D,H,W,C]
            cond_volume = data["cond_volume"].astype(np.float32)  # [F,1,D,H,W]
            cond_flow = data["cond_flow"].astype(np.float32)  # [F,3,D,H,W]
            object_center = data["object_center_world"].astype(np.float32)  # [F,3]
            object_vel = data["object_velocity_voxel"].astype(np.float32)  # [F,3]
            bounds_xyz = data["bounds_xyz"].astype(np.float32)  # [6]
            voxel_size_xyz = data["voxel_size_xyz"].astype(np.float32)  # [3]
            dt = data["dt"].astype(np.float32) if "dt" in data else np.asarray([1.0], dtype=np.float32)
            fluid_mask = (
                data["fluid_mask"].astype(np.float32) if "fluid_mask" in data else phase[..., 0].astype(np.float32)
            )

        if phase.ndim != 5 or phase.shape[-1] < 2:
            raise ValueError(f"phase_onehot must be [F,D,H,W,C] with C>=2, got {phase.shape}")
        if not self.include_gas_channel and phase.shape[-1] > 2:
            phase = phase[..., :2]

        if epi.sdf_path is not None and epi.sdf_path.exists():
            sdf = np.load(epi.sdf_path).astype(np.float32)  # [F,1,D,H,W]
        else:
            sdf = self._fallback_sdf(cond_volume)

        particle_points = None
        particle_valid = None
        if self.load_particle_gt and epi.surface_state_path is not None and epi.surface_state_path.exists():
            try:
                with np.load(epi.surface_state_path) as pz:
                    particle_points = pz["points"].astype(np.float32)  # [F,N,3]
                    if "valid" in pz:
                        particle_valid = pz["valid"].astype(bool)
                    else:
                        particle_valid = np.ones(particle_points.shape[:2], dtype=bool)
            except Exception:
                particle_points = None
                particle_valid = None

        t = min(target_flow.shape[0], phase.shape[0], cond_volume.shape[0], cond_flow.shape[0], sdf.shape[0])
        if particle_points is not None and particle_valid is not None:
            t = min(t, particle_points.shape[0], particle_valid.shape[0])

        pack = {
            "target_flow": target_flow[:t],
            "phase": phase[:t],
            "fluid_mask": fluid_mask[:t],
            "cond_volume": cond_volume[:t],
            "cond_flow": cond_flow[:t],
            "sdf": sdf[:t],
            "object_center": object_center[:t],
            "object_vel": object_vel[:t],
            "bounds_xyz": bounds_xyz,
            "voxel_size_xyz": voxel_size_xyz,
            "particle_points": particle_points[:t] if particle_points is not None else None,
            "particle_valid": particle_valid[:t] if particle_valid is not None else None,
            "dt": dt,
        }

        if self.precompute_derived_cache:
            phase_t = pack["phase"]
            sdf_t = pack["sdf"]
            liquid_seq = phase_t[..., 0].astype(np.float32)  # [F,D,H,W]
            solid_seq = phase_t[..., 1].astype(np.float32)  # [F,D,H,W]
            fluid_solid_interface_seq = np.stack(
                [_edge_map_3d(solid_seq[i]) for i in range(t)], axis=0
            )[:, None].astype(np.float32)
            interface_normal_seq = np.stack(
                [_safe_gradient_3d(sdf_t[i, 0]) for i in range(t)], axis=0
            ).astype(np.float32)
            pack["fluid_solid_interface_seq"] = fluid_solid_interface_seq
            pack["interface_normal_seq"] = interface_normal_seq

        self._cache[epi_idx] = pack
        self._cache.move_to_end(epi_idx)
        while len(self._cache) > self.max_episode_cache:
            self._cache.popitem(last=False)
        return pack

    def _sample_particle_gt(
        self,
        pack: Dict[str, np.ndarray],
        t0: int,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = pack.get("particle_points", None)
        v = pack.get("particle_valid", None)
        pnum = self.particle_gt_num_points

        if p is None or v is None:
            return (
                np.zeros((pnum, 3), dtype=np.float32),
                np.zeros((k + 1, pnum, 3), dtype=np.float32),
                np.zeros((pnum,), dtype=np.float32),
            )

        window_valid = np.all(v[t0 : t0 + k + 1], axis=0)
        valid_idx = np.where(window_valid)[0]
        if valid_idx.size == 0:
            return (
                np.zeros((pnum, 3), dtype=np.float32),
                np.zeros((k + 1, pnum, 3), dtype=np.float32),
                np.zeros((pnum,), dtype=np.float32),
            )

        rng = np.random.default_rng()
        if valid_idx.size >= pnum:
            sel = rng.choice(valid_idx, size=pnum, replace=False)
        else:
            sel = rng.choice(valid_idx, size=pnum, replace=True)

        pos_world = p[t0 : t0 + k + 1, sel, :]  # [k+1,P,3]
        xmin, xmax, ymin, ymax, zmin, zmax = [float(x) for x in pack["bounds_xyz"]]
        sx, sy, sz = [max(1e-8, float(x)) for x in pack["voxel_size_xyz"]]

        x_vox = (pos_world[..., 0] - xmin) / sx
        y_vox = (pos_world[..., 1] - ymin) / sy
        z_vox = (pos_world[..., 2] - zmin) / sz
        pos_vox = np.stack([x_vox, y_vox, z_vox], axis=-1).astype(np.float32)
        pos_start = pos_vox[0]
        valid_mask = np.ones((pnum,), dtype=np.float32)
        return pos_start, pos_vox, valid_mask

    @staticmethod
    def _center_from_mask(mask_dhw: np.ndarray) -> np.ndarray:
        idx = np.argwhere(mask_dhw > 0.5)
        if idx.shape[0] == 0:
            d, h, w = mask_dhw.shape
            return np.asarray([0.5 * (w - 1), 0.5 * (h - 1), 0.5 * (d - 1)], dtype=np.float32)
        zyx = idx.astype(np.float32).mean(axis=0)
        return np.asarray([zyx[2], zyx[1], zyx[0]], dtype=np.float32)  # xyz

    @staticmethod
    def _velocity_to_local_frame(v_xyz: np.ndarray) -> np.ndarray:
        x_axis = np.asarray(v_xyz, dtype=np.float32)
        n = float(np.linalg.norm(x_axis))
        if n < 1e-6:
            return np.eye(3, dtype=np.float32)
        x_axis = x_axis / n
        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(x_axis, up))) > 0.95:
            up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        y_axis = np.cross(up, x_axis)
        y_axis /= max(1e-6, float(np.linalg.norm(y_axis)))
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= max(1e-6, float(np.linalg.norm(z_axis)))
        return np.stack([x_axis, y_axis, z_axis], axis=0).astype(np.float32)

    @staticmethod
    def _to_local(frame_3x3: np.ndarray, v_xyz: np.ndarray) -> np.ndarray:
        # frame rows are basis vectors in world coordinates -> local = R * world
        return (frame_3x3 @ np.asarray(v_xyz, dtype=np.float32).reshape(3,)).astype(np.float32)

    @staticmethod
    def _geometry_descriptor(
        solid_mask_dhw: np.ndarray,
        boundary_mask_dhw: np.ndarray,
        liquid_mask_dhw: np.ndarray,
    ) -> np.ndarray:
        # Geometry descriptor richer than simple global means.
        d, h, w = solid_mask_dhw.shape
        occ = (solid_mask_dhw > 0.5)
        idx = np.argwhere(occ)
        if idx.shape[0] == 0:
            return np.zeros((16,), dtype=np.float32)

        z = idx[:, 0].astype(np.float32)
        y = idx[:, 1].astype(np.float32)
        x = idx[:, 2].astype(np.float32)
        n = float(idx.shape[0])
        vol_frac = n / float(max(1, d * h * w))

        zmin, zmax = float(z.min()), float(z.max())
        ymin, ymax = float(y.min()), float(y.max())
        xmin, xmax = float(x.min()), float(x.max())
        bz = (zmax - zmin + 1.0) / float(max(1, d))
        by = (ymax - ymin + 1.0) / float(max(1, h))
        bx = (xmax - xmin + 1.0) / float(max(1, w))

        cz = float(z.mean()) / float(max(1, d - 1))
        cy = float(y.mean()) / float(max(1, h - 1))
        cx = float(x.mean()) / float(max(1, w - 1))

        xyz = np.stack([x, y, z], axis=1)
        xyz = xyz - xyz.mean(axis=0, keepdims=True)
        cov = (xyz.T @ xyz) / max(1.0, n - 1.0)
        evals = np.linalg.eigvalsh(cov).astype(np.float32)
        evals = np.sort(np.clip(evals, 0.0, None))[::-1]
        ev_sum = float(evals.sum()) + 1e-8
        evals_n = evals / ev_sum

        surface = _edge_map_3d(solid_mask_dhw).astype(np.float32)
        surface_frac = float(surface.mean())
        boundary_mean = float(np.mean(boundary_mask_dhw))
        liquid_mean = float(np.mean(liquid_mask_dhw))

        anis_xy = bx / max(1e-6, by)
        anis_xz = bx / max(1e-6, bz)
        anis_yz = by / max(1e-6, bz)

        desc = np.asarray(
            [
                vol_frac,
                bx,
                by,
                bz,
                cx,
                cy,
                cz,
                float(evals_n[0]),
                float(evals_n[1]),
                float(evals_n[2]),
                anis_xy,
                anis_xz,
                anis_yz,
                surface_frac,
                boundary_mean,
                liquid_mean,
            ],
            dtype=np.float32,
        )
        return desc

    @staticmethod
    def _extract_center_patch(
        volume_cdhw: np.ndarray,
        center_xyz: np.ndarray,
        patch_size: int,
    ) -> np.ndarray:
        # volume: [C,D,H,W], center_xyz in voxel coords.
        c, d, h, w = volume_cdhw.shape
        p = int(max(2, patch_size))
        half = p // 2
        cx = int(round(float(center_xyz[0])))
        cy = int(round(float(center_xyz[1])))
        cz = int(round(float(center_xyz[2])))

        pad_x = (max(0, half - cx), max(0, (cx + half) - (w - 1)))
        pad_y = (max(0, half - cy), max(0, (cy + half) - (h - 1)))
        pad_z = (max(0, half - cz), max(0, (cz + half) - (d - 1)))
        if any(v > 0 for v in (*pad_x, *pad_y, *pad_z)):
            volume_cdhw = np.pad(
                volume_cdhw,
                ((0, 0), pad_z, pad_y, pad_x),
                mode="constant",
                constant_values=0.0,
            )
            cx += pad_x[0]
            cy += pad_y[0]
            cz += pad_z[0]

        x0, x1 = cx - half, cx - half + p
        y0, y1 = cy - half, cy - half + p
        z0, z1 = cz - half, cz - half + p
        patch = volume_cdhw[:, z0:z1, y0:y1, x0:x1]
        if patch.shape[1:] != (p, p, p):
            out = np.zeros((c, p, p, p), dtype=volume_cdhw.dtype)
            zd = min(p, patch.shape[1])
            yd = min(p, patch.shape[2])
            xd = min(p, patch.shape[3])
            out[:, :zd, :yd, :xd] = patch[:, :zd, :yd, :xd]
            patch = out
        return patch.astype(np.float32)

    @staticmethod
    def _sample_trilinear(
        volume_cdhw: np.ndarray,
        xq: np.ndarray,
        yq: np.ndarray,
        zq: np.ndarray,
    ) -> np.ndarray:
        # volume: [C,D,H,W], query coordinates in voxel xyz.
        _, d, h, w = volume_cdhw.shape
        x = np.clip(xq.astype(np.float32), 0.0, float(max(1, w - 1)))
        y = np.clip(yq.astype(np.float32), 0.0, float(max(1, h - 1)))
        z = np.clip(zq.astype(np.float32), 0.0, float(max(1, d - 1)))

        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        z0 = np.floor(z).astype(np.int64)
        x1 = np.clip(x0 + 1, 0, max(0, w - 1))
        y1 = np.clip(y0 + 1, 0, max(0, h - 1))
        z1 = np.clip(z0 + 1, 0, max(0, d - 1))

        wx = x - x0.astype(np.float32)
        wy = y - y0.astype(np.float32)
        wz = z - z0.astype(np.float32)

        def gather(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
            return volume_cdhw[:, iz, iy, ix]

        c000 = gather(x0, y0, z0)
        c001 = gather(x0, y0, z1)
        c010 = gather(x0, y1, z0)
        c011 = gather(x0, y1, z1)
        c100 = gather(x1, y0, z0)
        c101 = gather(x1, y0, z1)
        c110 = gather(x1, y1, z0)
        c111 = gather(x1, y1, z1)

        w000 = (1.0 - wx) * (1.0 - wy) * (1.0 - wz)
        w001 = (1.0 - wx) * (1.0 - wy) * wz
        w010 = (1.0 - wx) * wy * (1.0 - wz)
        w011 = (1.0 - wx) * wy * wz
        w100 = wx * (1.0 - wy) * (1.0 - wz)
        w101 = wx * (1.0 - wy) * wz
        w110 = wx * wy * (1.0 - wz)
        w111 = wx * wy * wz

        out = (
            c000 * w000[None]
            + c001 * w001[None]
            + c010 * w010[None]
            + c011 * w011[None]
            + c100 * w100[None]
            + c101 * w101[None]
            + c110 * w110[None]
            + c111 * w111[None]
        )
        return out.astype(np.float32)

    @staticmethod
    def _extract_canonical_patch(
        volume_cdhw: np.ndarray,
        center_xyz: np.ndarray,
        frame_3x3: np.ndarray,
        patch_size: int,
    ) -> np.ndarray:
        # Canonical patch sampling in local object frame.
        p = int(max(2, patch_size))
        half = 0.5 * float(p - 1)
        grid = np.linspace(-half, half, p, dtype=np.float32)
        lx, ly, lz = np.meshgrid(grid, grid, grid, indexing="xy")

        r = np.asarray(frame_3x3, dtype=np.float32)
        # local -> world : world = center + R^T * local
        ox = r[0, 0] * lx + r[1, 0] * ly + r[2, 0] * lz
        oy = r[0, 1] * lx + r[1, 1] * ly + r[2, 1] * lz
        oz = r[0, 2] * lx + r[1, 2] * ly + r[2, 2] * lz

        cx, cy, cz = [float(center_xyz[i]) for i in range(3)]
        xq = cx + ox
        yq = cy + oy
        zq = cz + oz
        return InteractionRolloutDataset._sample_trilinear(volume_cdhw, xq=xq, yq=yq, zq=zq)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        epi_idx, t0 = self.indices[idx]
        e = self._load_episode(epi_idx)
        k = self.rollout_steps

        vel = e["target_flow"] / self.velocity_scale
        phase = e["phase"]
        cond_volume = e["cond_volume"]
        cond_flow = e["cond_flow"] / self.velocity_scale
        sdf = e["sdf"]
        obj_center = e["object_center"]
        obj_vel = e["object_vel"] / self.velocity_scale
        bounds_xyz = e["bounds_xyz"]
        voxel_size_xyz = e["voxel_size_xyz"]

        vel_t = vel[t0]
        phase_t = phase[t0]
        liquid_t = phase_t[..., 0].astype(np.float32)
        solid_t = phase_t[..., 1].astype(np.float32)
        boundary_mask_t = cond_volume[t0, 0].astype(np.float32)
        sdf_t = sdf[t0, 0].astype(np.float32)
        fluid_mask_t = e.get("fluid_mask", phase[..., 0])[t0].astype(np.float32)

        fluid_init = vel_t.astype(np.float32)

        local_frame = self._velocity_to_local_frame(obj_vel[t0])
        vel_local_t = self._to_local(local_frame, obj_vel[t0])
        future_obj_vel = obj_vel[(t0 + 1) : (t0 + k + 1)].reshape(-1).astype(np.float32)
        future_obj_vel_local = np.stack(
            [self._to_local(local_frame, obj_vel[t0 + i + 1]) for i in range(k)],
            axis=0,
        ).reshape(-1).astype(np.float32)
        future_obj_center = obj_center[(t0 + 1) : (t0 + k + 1)].astype(np.float32)
        eps = 1e-8
        future_obj_center_voxel = np.stack(
            [
                (future_obj_center[:, 0] - float(bounds_xyz[0])) / max(float(voxel_size_xyz[0]), eps),
                (future_obj_center[:, 1] - float(bounds_xyz[2])) / max(float(voxel_size_xyz[1]), eps),
                (future_obj_center[:, 2] - float(bounds_xyz[4])) / max(float(voxel_size_xyz[2]), eps),
            ],
            axis=-1,
        ).astype(np.float32)
        solid_radius_voxel = np.full((k, 1), float(self.episodes[epi_idx].solid_size_voxel), dtype=np.float32)
        geom = self._geometry_descriptor(solid_t, boundary_mask_t, liquid_t)
        pose = np.zeros((7,), dtype=np.float32)
        pose[:3] = obj_center[t0]
        pose[3] = 1.0
        object_center_vox = self._center_from_mask(solid_t)
        object_inputs = {
            "type_id": np.array([0], dtype=np.int64),
            "geometry": geom[None],
            "pose": pose[None],
            "velocity": obj_vel[t0][None],
            "velocity_local": vel_local_t[None],
            "trajectory": future_obj_vel[None],
            "trajectory_local": future_obj_vel_local[None],
            "control": np.zeros((1, 8), dtype=np.float32),
            "center_voxel": object_center_vox[None].astype(np.float32),
            "mask": np.array([True], dtype=np.bool_),
        }

        fluid_solid_interface = _edge_map_3d(solid_t)[None]
        liquid_liquid_interface = np.zeros_like(fluid_solid_interface, dtype=np.float32)
        interface_normal = _safe_gradient_3d(sdf_t)
        psize = int(self.object_patch_size)
        if self.canonical_object_patches:
            sdf_patch = self._extract_canonical_patch(
                sdf_t[None], object_center_vox, local_frame, psize
            )
            surface_patch = self._extract_canonical_patch(
                fluid_solid_interface.astype(np.float32), object_center_vox, local_frame, psize
            )
            interface_patch = self._extract_canonical_patch(
                fluid_solid_interface.astype(np.float32), object_center_vox, local_frame, psize
            )
            interface_normal_patch = self._extract_canonical_patch(
                interface_normal.astype(np.float32), object_center_vox, local_frame, psize
            )
            boundary_vel_patch = self._extract_canonical_patch(
                cond_flow[t0].astype(np.float32), object_center_vox, local_frame, psize
            )
        else:
            sdf_patch = self._extract_center_patch(sdf_t[None], object_center_vox, psize)
            surface_patch = self._extract_center_patch(
                fluid_solid_interface.astype(np.float32), object_center_vox, psize
            )
            interface_patch = self._extract_center_patch(
                fluid_solid_interface.astype(np.float32), object_center_vox, psize
            )
            interface_normal_patch = self._extract_center_patch(
                interface_normal.astype(np.float32), object_center_vox, psize
            )
            boundary_vel_patch = self._extract_center_patch(
                cond_flow[t0].astype(np.float32), object_center_vox, psize
            )

        interface_inputs = {
            "fluid_solid_interface": fluid_solid_interface.astype(np.float32),
            "liquid_liquid_interface": liquid_liquid_interface.astype(np.float32),
            "interface_sdf": sdf_t[None].astype(np.float32),
            "interface_normal": interface_normal.astype(np.float32),
            "boundary_velocity": cond_flow[t0].astype(np.float32),
            "boundary_mask": boundary_mask_t[None].astype(np.float32),
            "object_interface_patch": interface_patch[None].astype(np.float32),
            "object_interface_normal_patch": interface_normal_patch[None].astype(np.float32),
            "object_boundary_velocity_patch": boundary_vel_patch[None].astype(np.float32),
        }
        if self.include_gas_channel and phase_t.shape[-1] > 2:
            liquid_gas_interface = _edge_map_3d(liquid_t)[None]
            interface_inputs["liquid_gas_interface"] = liquid_gas_interface.astype(np.float32)
        object_inputs["object_sdf_patch"] = sdf_patch[None].astype(np.float32)
        object_inputs["object_surface_patch"] = surface_patch[None].astype(np.float32)
        object_inputs["object_interface_patch"] = interface_patch[None].astype(np.float32)
        object_inputs["object_interface_normal_patch"] = interface_normal_patch[None].astype(np.float32)
        object_inputs["object_boundary_velocity_patch"] = boundary_vel_patch[None].astype(np.float32)
        object_inputs["local_frame"] = local_frame[None].astype(np.float32)

        control_seq = np.zeros((k, self.control_dim), dtype=np.float32)
        if self.control_dim > 0:
            for i in range(k):
                frame = t0 + i + 1
                if self.control_dim >= 3:
                    control_seq[i, :3] = obj_vel[frame][:3]
                if self.control_dim >= 6:
                    control_seq[i, 3:6] = (obj_center[frame] - obj_center[t0])[:3]
                if self.control_dim >= 7:
                    control_seq[i, 6] = float(np.mean(cond_volume[frame, 0]))
                if self.control_dim >= 8:
                    control_seq[i, 7] = float(np.mean(cond_flow[frame]))

        solid_sequence = {
            "solid_sdf": sdf[(t0 + 1) : (t0 + k + 1)].astype(np.float32),
            "solid_velocity": cond_flow[(t0 + 1) : (t0 + k + 1)].astype(np.float32),
            "solid_center_voxel": future_obj_center_voxel,
            "solid_radius_voxel": solid_radius_voxel,
        }

        target_velocity = vel[(t0 + 1) : (t0 + k + 1)].astype(np.float32)
        target_phase = np.transpose(phase[(t0 + 1) : (t0 + k + 1)], (0, 4, 1, 2, 3)).astype(np.float32)
        target_boundary_velocity = cond_flow[(t0 + 1) : (t0 + k + 1)].astype(np.float32)
        target_boundary_mask = cond_volume[(t0 + 1) : (t0 + k + 1)].astype(np.float32)
        target_interface_sdf = sdf[(t0 + 1) : (t0 + k + 1)].astype(np.float32)
        target_interface_normal = np.stack(
            [_safe_gradient_3d(target_interface_sdf[i, 0]) for i in range(k)], axis=0
        ).astype(np.float32)
        target_fluid_mask = e.get("fluid_mask", phase[..., 0])[(t0 + 1) : (t0 + k + 1), None].astype(np.float32)

        particle_pos_start, particle_pos_seq, particle_valid = self._sample_particle_gt(e, t0=t0, k=k)

        return {
            "fluid_init": torch.from_numpy(fluid_init),
            "solid_sequence": {k0: torch.from_numpy(v0) for k0, v0 in solid_sequence.items()},
            "object_inputs": {k1: torch.from_numpy(v1) for k1, v1 in object_inputs.items()},
            "interface_inputs": {k2: torch.from_numpy(v2) for k2, v2 in interface_inputs.items()},
            "control_sequence": torch.from_numpy(control_seq),
            "target_velocity": torch.from_numpy(target_velocity),
            "target_phase": torch.from_numpy(target_phase),
            "target_boundary_velocity": torch.from_numpy(target_boundary_velocity),
            "target_boundary_mask": torch.from_numpy(target_boundary_mask),
            "target_interface_sdf": torch.from_numpy(target_interface_sdf),
            "target_object_sdf": torch.from_numpy(target_interface_sdf),
            "target_interface_normal": torch.from_numpy(target_interface_normal),
            "target_fluid_mask": torch.from_numpy(target_fluid_mask),
            "particle_pos_start": torch.from_numpy(particle_pos_start),
            "particle_pos_seq": torch.from_numpy(particle_pos_seq),
            "particle_valid": torch.from_numpy(particle_valid),
            "target_dt": torch.tensor(float(e["dt"][0]), dtype=torch.float32),
            "episode_index": torch.tensor(epi_idx, dtype=torch.long),
            "frame_index": torch.tensor(t0, dtype=torch.long),
        }
