from __future__ import annotations

from typing import Any

from .bulk_interface_response_4d import BulkInterfaceResponse4DFlowPredictor
from .interaction_aware_hybrid_4d import InteractionAwareHybrid4DFlowPredictor

MODEL_REGISTRY = {
    "BulkInterfaceResponse4DFlowPredictor": BulkInterfaceResponse4DFlowPredictor,
    "InteractionAwareHybrid4DFlowPredictor": InteractionAwareHybrid4DFlowPredictor,
}


def build_model_from_config(model_cfg: dict[str, Any]):
    cfg = dict(model_cfg)
    name = str(cfg.pop("name", "BulkInterfaceResponse4DFlowPredictor"))
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**cfg)


__all__ = [
    "BulkInterfaceResponse4DFlowPredictor",
    "InteractionAwareHybrid4DFlowPredictor",
    "MODEL_REGISTRY",
    "build_model_from_config",
]
