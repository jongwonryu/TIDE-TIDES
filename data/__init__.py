from .interaction_rollout_dataset import InteractionRolloutDataset

__all__ = ["InteractionRolloutDataset"]

try:
    from .voxel_pair_dataset import DualSPHVoxelPairDataset  # type: ignore

    __all__.append("DualSPHVoxelPairDataset")
except ModuleNotFoundError:
    # Keep interaction-aware training usable even when voxel baseline file is absent.
    pass
