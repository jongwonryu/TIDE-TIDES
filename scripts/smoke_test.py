from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import build_model_from_config
from scripts.runtime import dataset_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run both models on the included compact example.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    for family in ("tide", "tides"):
        torch.manual_seed(0)
        cfg = json.loads((ROOT / "configs" / f"{family}_example.json").read_text())
        cfg["dataset"]["root"] = str(ROOT / "examples" / "mini_dataset")
        batch = dataset_batch(cfg, "train", 0)
        model = build_model_from_config(cfg["model"]).to(device).eval()
        solid = {key: value.to(device) for key, value in batch["solid_sequence"].items()}
        with torch.inference_mode():
            output = model(
                fluid_init=batch["fluid_init"].to(device),
                solid_sequence=solid,
                particle_pos_init=batch["particle_pos_start"].to(device),
                rollout_steps=2, return_phase=False,
            )
        velocity = output["velocity"]
        expected = (1, 2, 3, *batch["fluid_init"].shape[-3:])
        assert tuple(velocity.shape) == expected, (velocity.shape, expected)
        assert torch.isfinite(velocity).all(), f"{family} produced non-finite velocities"
        print(f"{family.upper()}: shape={tuple(velocity.shape)}, finite=True")
        del model, batch, output


if __name__ == "__main__":
    main()
