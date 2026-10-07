from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runtime import dataset_batch, get_trainer, load_checkpoint


def npz_inputs(path: Path, device: torch.device):
    with np.load(path, allow_pickle=False) as data:
        required = ("fluid_init", "solid_sdf", "solid_velocity")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Input NPZ is missing {missing}.")
        accepted = (*required, "particle_pos_init", "solid_center_voxel", "solid_radius_voxel")
        values = {key: torch.as_tensor(data[key].copy(), dtype=torch.float32, device=device)
                  for key in accepted if key in data}
    fluid = values["fluid_init"]
    if fluid.ndim == 4:
        fluid = fluid.unsqueeze(0)
    if fluid.ndim != 5 or fluid.shape[1] != 3:
        raise ValueError("fluid_init must have shape [3,D,H,W] or [B,3,D,H,W].")
    solid = {key: values[key] for key in ("solid_sdf", "solid_velocity")}
    for key, channels in (("solid_sdf", 1), ("solid_velocity", 3)):
        if solid[key].ndim == 5:
            solid[key] = solid[key].unsqueeze(0)
        value = solid[key]
        if value.ndim != 6 or value.shape[2] != channels:
            raise ValueError(f"{key} must have shape [T,{channels},D,H,W] or [B,T,{channels},D,H,W].")
        if value.shape[0] != fluid.shape[0] or value.shape[-3:] != fluid.shape[-3:]:
            raise ValueError(f"{key} must share the fluid batch and grid dimensions.")
    if solid["solid_sdf"].shape[1] != solid["solid_velocity"].shape[1]:
        raise ValueError("Solid SDF and boundary velocity must have the same horizon.")
    for key in ("solid_center_voxel", "solid_radius_voxel"):
        if key in values:
            value = values[key]
            if value.ndim == 2:
                value = value.unsqueeze(0)
            solid[key] = value
    particles = values.get("particle_pos_init")
    if particles is not None:
        if particles.ndim == 2:
            particles = particles.unsqueeze(0)
        if particles.ndim != 3 or particles.shape[-1] != 3 or particles.shape[0] != fluid.shape[0]:
            raise ValueError("particle_pos_init must have shape [P,3] or [B,P,3].")
    return fluid, solid, particles


def main() -> None:
    parser = argparse.ArgumentParser(description="Free-running TIDE/TIDES inference.")
    parser.add_argument("--model", choices=("tide", "tides"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Prepared input NPZ; see README for array layout.")
    source.add_argument("--dataset-root", type=Path, help="Dataset with a manifest.jsonl.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive.")
    device = torch.device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, args.model, args.config, device)
    get_trainer(args.model).set_seed(int(cfg["train"].get("seed", 0)))
    if args.input is not None:
        fluid, solid, particles = npz_inputs(args.input, device)
    else:
        cfg["dataset"]["root"] = str(args.dataset_root.resolve())
        cfg["dataset"]["rollout_steps"] = args.steps
        batch = dataset_batch(cfg, args.split, args.index)
        trainer = get_trainer(args.model)
        fluid = batch["fluid_init"].to(device)
        solid = trainer.move_nested_to_device(batch["solid_sequence"], device)
        particles = batch["particle_pos_start"].to(device)
    if solid["solid_sdf"].shape[1] < args.steps:
        parser.error("The input must provide future solid states for every requested step.")
    if args.model == "tides" and cfg["model"].get("dual_coupling_enabled", True) and particles is None:
        parser.error("TIDES transport feedback requires particle_pos_init in the input NPZ.")
    with torch.inference_mode():
        output = model(
            fluid_init=fluid, solid_sequence=solid,
            particle_pos_init=particles, rollout_steps=args.steps,
            return_phase=False, return_pressure=False,
        )
    arrays = {key: value.detach().cpu().numpy()
              for key, value in output.items() if isinstance(value, torch.Tensor)}
    arrays["velocity_scale"] = np.asarray(cfg["dataset"].get("velocity_scale", 1.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Saved {args.steps}-step rollout: {args.output}")
    print(f"velocity shape: {arrays['velocity'].shape}")


if __name__ == "__main__":
    main()
