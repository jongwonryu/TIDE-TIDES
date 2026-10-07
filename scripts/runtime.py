from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import torch

from models import build_model_from_config

MODEL_NAMES = {
    "tide": "BulkInterfaceResponse4DFlowPredictor",
    "tides": "InteractionAwareHybrid4DFlowPredictor",
}


def get_trainer(family: str):
    module = "train_bulk_interface" if family == "tide" else "train_interaction_aware"
    return importlib.import_module(f"train.{module}")


def load_checkpoint(
    path: Path, family: str, config_path: Path | None, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if config_path is not None:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("cfg"), dict):
        cfg = checkpoint["cfg"]
    else:
        raise ValueError("Provide --config when the checkpoint has no embedded cfg.")
    if cfg["model"].get("name") != MODEL_NAMES[family]:
        raise ValueError(f"The configuration does not describe {family.upper()}.")
    model = build_model_from_config(cfg["model"])
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, cfg


def dataset_batch(cfg: dict[str, Any], split: str, index: int):
    family = next(key for key, name in MODEL_NAMES.items() if cfg["model"]["name"] == name)
    loader = get_trainer(family).build_loader(cfg, split=split, shuffle=False)
    if not len(loader.dataset):
        raise ValueError(f"No rollout windows found for split {split!r}.")
    from torch.utils.data import default_collate

    return default_collate([loader.dataset[index]])
