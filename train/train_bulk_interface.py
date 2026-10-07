from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import InteractionRolloutDataset
from models import BulkInterfaceResponse4DFlowPredictor, build_model_from_config

LOSS_TRACK_KEYS = (
    "total_loss",
    "l_vel",
    "l_vel_near_t1",
    "l_mag_t1",
    "l_resp_t1",
    "l_grad",
    "l_div",
    "l_traj",
    "l_dual",
    "l_particle_module",
    "l_exchange",
    "l_int",
    "l_roll",
    "l_phase",
    "rollout_endpoint_error",
    "trajectory_endpoint_error",
    "near_vel_error",
    "far_vel_error",
    "near_voxel_count",
    "far_voxel_count",
    "near_voxel_frac",
    "far_voxel_frac",
    "mask_valid_voxel_count",
    "mask_valid_voxel_frac",
    "near_sdf_threshold_used",
    "far_sdf_threshold_used",
    "mask_quantile_fallback_used",
    "near_sdf_mean",
    "far_sdf_mean",
    "near_far_overlap_frac",
    "pred_speed_mean_t1",
    "gt_speed_mean_t1",
    "pred_gt_speed_ratio_t1",
    "interaction_intensity_mean",
    "exchange_balance_error",
    "pressure_correction_norm",
    "pressure_divergence_pre",
    "pressure_divergence_post",
    "pressure_divergence_delta",
    "pressure_stage1_correction_norm",
    "pressure_stage1_divergence_pre",
    "pressure_stage1_divergence_post",
    "pressure_stage2_correction_norm",
    "pressure_stage2_divergence_pre",
    "pressure_stage2_divergence_post",
    "pressure_stage2_divergence_delta",
    "output_pressure_correction_norm",
    "output_pressure_divergence_pre",
    "output_pressure_divergence_post",
    "output_pressure_divergence_delta",
    "output_boundary_mismatch_pre",
    "output_boundary_mismatch_post",
    "dual_particle_step_norm",
    "dual_p2g_norm",
    "obj_dyn_norm",
    "obj_geo_norm",
    "obj_geo_dyn_ratio",
    "obj_geo_gate_mean",
    "obj_geo_to_int_raw_norm",
    "obj_geo_to_int_gated_norm",
    "obj_geo_effect_ratio",
    "direct_obj_delta_norm",
)

TIME_SERIES_KEYS = (
    "target_velocity",
    "target_phase",
    "target_boundary_velocity",
    "target_boundary_mask",
    "target_interface_sdf",
    "target_object_sdf",
    "target_interface_normal",
    "target_fluid_mask",
)


class TerminalProgressBar:
    def __init__(self, total: int, prefix: str, width: int = 28, enabled: bool = True) -> None:
        self.total = max(1, int(total))
        self.prefix = str(prefix)
        self.width = int(max(8, width))
        self.enabled = bool(enabled)
        self._last_len = 0
        self._started = False
        self._t0 = 0.0

    @staticmethod
    def _format_hms(seconds: float) -> str:
        s = max(0, int(round(float(seconds))))
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"

    def update(self, current: int, postfix: str = "") -> None:
        if not self.enabled:
            return
        cur = min(self.total, max(0, int(current)))
        if not self._started:
            self._t0 = time.time()
            self._started = True
        frac = float(cur) / float(self.total)
        filled = int(round(self.width * frac))
        bar = "#" * filled + "-" * (self.width - filled)

        elapsed = max(0.0, time.time() - self._t0)
        eta = 0.0
        if cur > 0:
            sec_per_step = elapsed / float(cur)
            eta = sec_per_step * float(self.total - cur)
        eta_str = self._format_hms(eta)
        elapsed_str = self._format_hms(elapsed)

        msg = (
            f"\r{self.prefix} [{bar}] {cur:4d}/{self.total:4d} "
            f"eta {eta_str} elap {elapsed_str} {postfix}"
        )
        pad = max(0, self._last_len - len(msg))
        print(msg + (" " * pad), end="", flush=True)
        self._last_len = len(msg)
        if cur >= self.total:
            self.close()

    def close(self) -> None:
        if self.enabled and self._started:
            print("", flush=True)
            self._started = False
            self._t0 = 0.0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bulk-plus-interface 4D flow predictor.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tide_train.json",
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--train-total-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--scheduler-step-size", type=int, default=None)
    parser.add_argument("--scheduler-gamma", type=float, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--vorticity-weight", type=float, default=None)
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--max-val-episodes", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--interface-hidden-channels", type=int, default=None)
    parser.add_argument("--interface-band-width", type=float, default=None)
    parser.add_argument("--particle-step-scale", type=float, default=None)
    parser.add_argument("--use-projection", type=int, default=None)
    parser.add_argument("--use-object-patch-context", type=int, default=None)
    parser.add_argument("--global-interface-response", type=int, default=None)
    parser.add_argument("--predict-phase", type=int, default=None)
    parser.add_argument(
        "--tier3-variant",
        type=str,
        default=None,
        choices=[
            "geo_off",
            "geo_strong",
            "geo_residual",
            "dual_off",
            "pressure_off",
            "pressure_latent_only",
            "ours_core",
            "ours_full",
        ],
        help="Preset bundle for Tier-3 mechanism experiments.",
    )
    parser.add_argument(
        "--cond-variant",
        type=str,
        default=None,
        choices=["hybrid", "euler", "object", "bulk_only"],
        help="Conditioning variant for Tier-1 experiments.",
    )
    parser.add_argument("--coupling-variant", type=str, default=None)
    parser.add_argument("--direct-obj-to-fluid-scale", type=float, default=None)
    parser.add_argument("--direct-fluid-to-obj-scale", type=float, default=None)
    parser.add_argument("--exchange-reaction-scale", type=float, default=None)
    parser.add_argument("--geo-interface-condition-scale", type=float, default=None)
    parser.add_argument("--geo-operator-init-scale", type=float, default=None)
    parser.add_argument("--object-geometry-stream-scale", type=float, default=None)
    parser.add_argument("--pressure-correction-scale", type=float, default=None)
    parser.add_argument("--pressure-post-p2g-enabled", type=int, default=None)
    parser.add_argument("--pressure-post-p2g-scale", type=float, default=None)
    parser.add_argument("--output-pressure-correction-enabled", type=int, default=None)
    parser.add_argument("--output-pressure-correction-scale", type=float, default=None)
    parser.add_argument("--output-pressure-feedback-scale", type=float, default=None)
    parser.add_argument("--dual-coupling-enabled", type=int, default=None)
    parser.add_argument("--dual-particle-step-scale", type=float, default=None)
    parser.add_argument("--dual-p2g-scale", type=float, default=None)
    parser.add_argument("--lambda-div", type=float, default=None)
    parser.add_argument("--lambda-traj", type=float, default=None)
    parser.add_argument("--lambda-int", type=float, default=None)
    parser.add_argument("--lambda-roll", type=float, default=None)
    parser.add_argument("--disable-curriculum", action="store_true")
    parser.add_argument("--curriculum-apply-to-val", action="store_true")
    parser.add_argument("--val-ablation-pressure-off", action="store_true")
    parser.add_argument("--val-ablation-geo-zero", action="store_true")
    parser.add_argument("--val-ablation-geo-shuffle", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_model(model_cfg: Dict[str, Any]) -> torch.nn.Module:
    return build_model_from_config(model_cfg)

def resolve_run_output_dir(base_dir: Path) -> Path:
    """Return a non-overlapping output directory for this run."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return base_dir
    parent = base_dir.parent
    stem = base_dir.name
    idx = 1
    while True:
        cand = parent / f"{stem}_run{idx:02d}"
        if not cand.exists():
            return cand
        idx += 1


def _apply_tier3_variant_preset(
    run_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    loss_cfg: Dict[str, Any],
    variant: str,
) -> None:
    v = str(variant).strip().lower()
    presets: Dict[str, Dict[str, Any]] = {
        "geo_off": {
            "cond_variant": "hybrid",
            "coupling_variant": "strict_mediator",
            "direct_obj_to_fluid_scale": 0.0,
            "direct_fluid_to_obj_scale": 0.0,
            "exchange_reaction_scale": 1.0,
            "geo_interface_condition_scale": 0.0,
            "geo_operator_init_scale": 0.0,
            "object_geometry_stream_scale": 0.0,
            "dual_coupling_enabled": True,
            "dual_particle_step_scale": 1.0,
            "dual_p2g_scale": 0.2,
            "pressure_correction_scale": 0.2,
            "pressure_post_p2g_enabled": False,
            "pressure_post_p2g_scale": 0.0,
            "output_pressure_correction_enabled": False,
            "output_pressure_correction_scale": 0.0,
            "output_pressure_feedback_scale": 0.0,
            "vorticity_weight": 0.0,
        },
        "geo_strong": {
            "cond_variant": "hybrid",
            "coupling_variant": "strict_mediator",
            "direct_obj_to_fluid_scale": 0.0,
            "direct_fluid_to_obj_scale": 0.0,
            "exchange_reaction_scale": 1.0,
            "geo_interface_condition_scale": 1.0,
            "geo_operator_init_scale": 0.1,
            "object_geometry_stream_scale": 1.25,
            "dual_coupling_enabled": True,
            "dual_particle_step_scale": 1.0,
            "dual_p2g_scale": 0.2,
            "pressure_correction_scale": 0.2,
            "pressure_post_p2g_enabled": False,
            "pressure_post_p2g_scale": 0.0,
            "output_pressure_correction_enabled": False,
            "output_pressure_correction_scale": 0.0,
            "output_pressure_feedback_scale": 0.0,
            "vorticity_weight": 0.0,
        },
        "geo_residual": {
            "cond_variant": "hybrid",
            "coupling_variant": "strict_mediator",
            "direct_obj_to_fluid_scale": 0.0,
            "direct_fluid_to_obj_scale": 0.0,
            "exchange_reaction_scale": 1.0,
            "geo_interface_condition_scale": 0.5,
            "geo_operator_init_scale": 0.03,
            "object_geometry_stream_scale": 0.8,
            "dual_coupling_enabled": True,
            "dual_particle_step_scale": 1.0,
            "dual_p2g_scale": 0.2,
            "pressure_correction_scale": 0.2,
            "pressure_post_p2g_enabled": False,
            "pressure_post_p2g_scale": 0.0,
            "output_pressure_correction_enabled": False,
            "output_pressure_correction_scale": 0.0,
            "output_pressure_feedback_scale": 0.0,
            "vorticity_weight": 0.0,
        },
        "dual_off": {
            "cond_variant": "hybrid",
            "coupling_variant": "strict_mediator",
            "direct_obj_to_fluid_scale": 0.0,
            "direct_fluid_to_obj_scale": 0.0,
            "exchange_reaction_scale": 1.0,
            "geo_interface_condition_scale": 0.5,
            "geo_operator_init_scale": 0.03,
            "object_geometry_stream_scale": 0.8,
            "dual_coupling_enabled": False,
            "dual_particle_step_scale": 1.0,
            "dual_p2g_scale": 0.0,
            "pressure_correction_scale": 0.2,
            "pressure_post_p2g_enabled": False,
            "pressure_post_p2g_scale": 0.0,
            "output_pressure_correction_enabled": False,
            "output_pressure_correction_scale": 0.0,
            "output_pressure_feedback_scale": 0.0,
            "vorticity_weight": 0.0,
        },
        "pressure_off": {
            "cond_variant": "hybrid",
            "coupling_variant": "strict_mediator",
            "direct_obj_to_fluid_scale": 0.0,
            "direct_fluid_to_obj_scale": 0.0,
            "exchange_reaction_scale": 1.0,
            "geo_interface_condition_scale": 0.5,
            "geo_operator_init_scale": 0.03,
            "object_geometry_stream_scale": 0.8,
            "dual_coupling_enabled": True,
            "dual_particle_step_scale": 1.0,
            "dual_p2g_scale": 0.2,
            "pressure_correction_scale": 0.0,
            "pressure_post_p2g_enabled": False,
            "pressure_post_p2g_scale": 0.0,
            "output_pressure_correction_enabled": False,
            "output_pressure_correction_scale": 0.0,
            "output_pressure_feedback_scale": 0.0,
            "vorticity_weight": 0.0,
        },
        "pressure_latent_only": {
            "cond_variant": "hybrid",
            "coupling_variant": "strict_mediator",
            "direct_obj_to_fluid_scale": 0.0,
            "direct_fluid_to_obj_scale": 0.0,
            "exchange_reaction_scale": 1.0,
            "geo_interface_condition_scale": 0.5,
            "geo_operator_init_scale": 0.03,
            "object_geometry_stream_scale": 0.8,
            "dual_coupling_enabled": True,
            "dual_particle_step_scale": 1.0,
            "dual_p2g_scale": 0.2,
            "pressure_correction_scale": 0.2,
            "pressure_post_p2g_enabled": False,
            "pressure_post_p2g_scale": 0.0,
            "output_pressure_correction_enabled": False,
            "output_pressure_correction_scale": 0.0,
            "output_pressure_feedback_scale": 0.0,
            "vorticity_weight": 0.0,
        },
        "ours_core": {
            "cond_variant": "hybrid",
            "coupling_variant": "strict_mediator",
            "direct_obj_to_fluid_scale": 0.0,
            "direct_fluid_to_obj_scale": 0.0,
            "exchange_reaction_scale": 1.0,
            "geo_interface_condition_scale": 0.5,
            "geo_operator_init_scale": 0.03,
            "object_geometry_stream_scale": 0.8,
            "dual_coupling_enabled": True,
            "dual_particle_step_scale": 1.0,
            "dual_p2g_scale": 0.2,
            "pressure_correction_scale": 0.2,
            "pressure_post_p2g_enabled": False,
            "pressure_post_p2g_scale": 0.0,
            "output_pressure_correction_enabled": False,
            "output_pressure_correction_scale": 0.0,
            "output_pressure_feedback_scale": 0.0,
            "vorticity_weight": 0.0,
        },
        "ours_full": {
            "cond_variant": "hybrid",
            "coupling_variant": "strict_mediator",
            "direct_obj_to_fluid_scale": 0.0,
            "direct_fluid_to_obj_scale": 0.0,
            "exchange_reaction_scale": 1.0,
            "geo_interface_condition_scale": 0.5,
            "geo_operator_init_scale": 0.03,
            "object_geometry_stream_scale": 0.8,
            "dual_coupling_enabled": True,
            "dual_particle_step_scale": 1.0,
            "dual_p2g_scale": 0.2,
            "pressure_correction_scale": 0.2,
            "pressure_post_p2g_enabled": True,
            "pressure_post_p2g_scale": 1.0,
            "output_pressure_correction_enabled": True,
            "output_pressure_correction_scale": 0.1,
            "output_pressure_feedback_scale": 0.1,
            "vorticity_weight": 0.0,
        },
    }
    if v not in presets:
        raise ValueError(f"unknown tier3_variant: {variant!r}")
    preset = presets[v]
    run_cfg["tier3_variant"] = v
    run_cfg["cond_variant"] = str(preset["cond_variant"])
    loss_cfg["vorticity_weight"] = float(preset["vorticity_weight"])
    for key, value in preset.items():
        if key in ("cond_variant", "vorticity_weight"):
            continue
        model_cfg[key] = value


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    run_cfg = cfg.setdefault("train", {})
    ds_cfg = cfg.setdefault("dataset", {})
    model_cfg = cfg.setdefault("model", {})
    loss_cfg = cfg.setdefault("loss", {})
    weights_cfg = cfg.setdefault("loss_weights", {})

    if args.seed is not None:
        run_cfg["seed"] = int(args.seed)
    if args.tier3_variant is not None:
        _apply_tier3_variant_preset(run_cfg, model_cfg, loss_cfg, args.tier3_variant)

    if args.epochs is not None:
        run_cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        run_cfg["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        run_cfg["lr"] = float(args.lr)
    if args.weight_decay is not None:
        run_cfg["weight_decay"] = float(args.weight_decay)
    if args.max_train_steps is not None:
        run_cfg["max_train_steps"] = int(args.max_train_steps)
    if args.max_val_steps is not None:
        run_cfg["max_val_steps"] = int(args.max_val_steps)
    if args.num_workers is not None:
        run_cfg["num_workers"] = int(args.num_workers)
    if args.log_interval is not None:
        run_cfg["log_interval"] = int(args.log_interval)
    if args.scheduler_step_size is not None:
        run_cfg["scheduler_step_size"] = int(args.scheduler_step_size)
    if args.scheduler_gamma is not None:
        run_cfg["scheduler_gamma"] = float(args.scheduler_gamma)
    if args.checkpoint_every_steps is not None:
        run_cfg["checkpoint_every_steps"] = int(args.checkpoint_every_steps)
    if args.rollout_steps is not None:
        ds_cfg["rollout_steps"] = int(args.rollout_steps)
    if args.vorticity_weight is not None:
        loss_cfg["vorticity_weight"] = float(args.vorticity_weight)
    if args.max_train_episodes is not None:
        ds_cfg["max_train_episodes"] = int(args.max_train_episodes)
    if args.max_val_episodes is not None:
        ds_cfg["max_val_episodes"] = int(args.max_val_episodes)
    if args.output_dir is not None:
        run_cfg["output_dir"] = str(args.output_dir)
    if args.resume_checkpoint is not None:
        run_cfg["resume_checkpoint"] = str(args.resume_checkpoint)
    if args.interface_hidden_channels is not None:
        model_cfg["interface_hidden_channels"] = int(args.interface_hidden_channels)
    if args.interface_band_width is not None:
        model_cfg["interface_band_width"] = float(args.interface_band_width)
    if args.particle_step_scale is not None:
        model_cfg["particle_step_scale"] = float(args.particle_step_scale)
    if args.use_projection is not None:
        model_cfg["use_projection"] = bool(int(args.use_projection))
    if args.use_object_patch_context is not None:
        model_cfg["use_object_patch_context"] = bool(int(args.use_object_patch_context))
    if args.global_interface_response is not None:
        model_cfg["global_interface_response"] = bool(int(args.global_interface_response))
    if args.predict_phase is not None:
        model_cfg["predict_phase"] = bool(int(args.predict_phase))
    if args.cond_variant is not None:
        run_cfg["cond_variant"] = str(args.cond_variant)
    if args.coupling_variant is not None:
        model_cfg["coupling_variant"] = str(args.coupling_variant)
    if args.direct_obj_to_fluid_scale is not None:
        model_cfg["direct_obj_to_fluid_scale"] = float(args.direct_obj_to_fluid_scale)
    if args.direct_fluid_to_obj_scale is not None:
        model_cfg["direct_fluid_to_obj_scale"] = float(args.direct_fluid_to_obj_scale)
    if args.exchange_reaction_scale is not None:
        model_cfg["exchange_reaction_scale"] = float(args.exchange_reaction_scale)
    if args.geo_interface_condition_scale is not None:
        model_cfg["geo_interface_condition_scale"] = float(args.geo_interface_condition_scale)
    if args.geo_operator_init_scale is not None:
        model_cfg["geo_operator_init_scale"] = float(args.geo_operator_init_scale)
    if args.object_geometry_stream_scale is not None:
        model_cfg["object_geometry_stream_scale"] = float(args.object_geometry_stream_scale)
    if args.pressure_correction_scale is not None:
        model_cfg["pressure_correction_scale"] = float(args.pressure_correction_scale)
    if args.pressure_post_p2g_enabled is not None:
        model_cfg["pressure_post_p2g_enabled"] = bool(int(args.pressure_post_p2g_enabled))
    if args.pressure_post_p2g_scale is not None:
        model_cfg["pressure_post_p2g_scale"] = float(args.pressure_post_p2g_scale)
    if args.output_pressure_correction_enabled is not None:
        model_cfg["output_pressure_correction_enabled"] = bool(int(args.output_pressure_correction_enabled))
    if args.output_pressure_correction_scale is not None:
        model_cfg["output_pressure_correction_scale"] = float(args.output_pressure_correction_scale)
    if args.output_pressure_feedback_scale is not None:
        model_cfg["output_pressure_feedback_scale"] = float(args.output_pressure_feedback_scale)
    if args.dual_coupling_enabled is not None:
        model_cfg["dual_coupling_enabled"] = bool(int(args.dual_coupling_enabled))
    if args.dual_particle_step_scale is not None:
        model_cfg["dual_particle_step_scale"] = float(args.dual_particle_step_scale)
    if args.dual_p2g_scale is not None:
        model_cfg["dual_p2g_scale"] = float(args.dual_p2g_scale)
    if args.lambda_div is not None:
        weights_cfg["lambda_div"] = float(args.lambda_div)
    if args.lambda_traj is not None:
        weights_cfg["lambda_traj"] = float(args.lambda_traj)
    if args.lambda_int is not None:
        weights_cfg["lambda_int"] = float(args.lambda_int)
    if args.lambda_roll is not None:
        weights_cfg["lambda_roll"] = float(args.lambda_roll)

    if args.disable_curriculum:
        rcfg = run_cfg.setdefault("rollout_curriculum", {})
        rcfg["enabled"] = False
    if args.curriculum_apply_to_val:
        rcfg = run_cfg.setdefault("rollout_curriculum", {})
        rcfg["apply_to_val"] = True

    if args.val_ablation_pressure_off:
        run_cfg["val_ablation_pressure_off"] = True
    if args.val_ablation_geo_zero:
        run_cfg["val_ablation_geo_zero"] = True
    if args.val_ablation_geo_shuffle:
        run_cfg["val_ablation_geo_shuffle"] = True

    if args.train_total_steps is not None:
        total_steps = int(args.train_total_steps)
        if total_steps <= 0:
            raise ValueError("--train-total-steps must be > 0")
        steps_per_epoch = int(run_cfg.get("max_train_steps", 0))
        if steps_per_epoch <= 0:
            raise ValueError(
                "--train-total-steps requires --max-train-steps (or config train.max_train_steps) > 0."
            )
        run_cfg["epochs"] = int(math.ceil(total_steps / float(steps_per_epoch)))
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_nested_to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_nested_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        typ = type(x)
        return typ(move_nested_to_device(v, device) for v in x)
    return x


def _shuffle_geometry_tensor(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    if y.numel() <= 1:
        return y
    if y.shape[0] > 1:
        perm = torch.randperm(y.shape[0], device=y.device)
        return y.index_select(0, perm)
    if y.ndim >= 2 and y.shape[1] > 1:
        perm = torch.randperm(y.shape[1], device=y.device)
        return y.index_select(1, perm)
    flat = y.reshape(-1)
    perm = torch.randperm(flat.numel(), device=y.device)
    return flat.index_select(0, perm).reshape_as(y)


def _ablate_object_geometry_inputs(object_inputs: Dict[str, Any], mode: str) -> Dict[str, Any]:
    m = str(mode).strip().lower()
    if m == "none":
        return object_inputs
    if m not in ("zero", "shuffle"):
        raise ValueError(f"object_ablation_mode must be one of ['none','zero','shuffle'], got {mode!r}")

    out: Dict[str, Any] = {}
    for k, v in object_inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        else:
            out[k] = v

    geo_keys = (
        "geometry",
        "object_sdf_patch",
        "object_surface_patch",
        "object_interface_patch",
        "object_interface_normal_patch",
        "object_boundary_velocity_patch",
        "local_frame",
    )
    for key in geo_keys:
        v = out.get(key, None)
        if not isinstance(v, torch.Tensor):
            continue
        if m == "zero":
            out[key] = torch.zeros_like(v)
        else:
            out[key] = _shuffle_geometry_tensor(v)
    return out


def _set_pressure_correction_scale(
    model: torch.nn.Module, new_scale: float
) -> Tuple[float, float, float, float, bool, bool, float, float] | None:
    rc = getattr(model, "rollout_core", None)
    if rc is None or not hasattr(rc, "pressure_operator"):
        return None
    prev_core = float(getattr(rc, "pressure_correction_scale", 0.0))
    prev_op1 = float(getattr(rc.pressure_operator, "correction_scale", 0.0))
    prev_post_scale = float(getattr(rc, "pressure_post_p2g_scale", 0.0))
    prev_post_enabled = bool(getattr(rc, "pressure_post_p2g_enabled", False))
    prev_out_enabled = bool(getattr(rc, "output_pressure_correction_enabled", False))
    prev_out_scale = float(getattr(rc, "output_pressure_correction_scale", 0.0))
    if hasattr(rc, "output_velocity_projection"):
        prev_out_proj_scale = float(getattr(rc.output_velocity_projection, "correction_scale", 0.0))
    else:
        prev_out_proj_scale = 0.0
    if hasattr(rc, "pressure_operator_post"):
        prev_op2 = float(getattr(rc.pressure_operator_post, "correction_scale", 0.0))
    else:
        prev_op2 = 0.0

    scale = float(new_scale)
    rc.pressure_correction_scale = scale
    rc.pressure_operator.correction_scale = scale
    if hasattr(rc, "pressure_operator_post"):
        rc.pressure_operator_post.correction_scale = scale * prev_post_scale
    if hasattr(rc, "output_pressure_correction_scale"):
        rc.output_pressure_correction_scale = scale
    if hasattr(rc, "output_velocity_projection"):
        rc.output_velocity_projection.correction_scale = scale
    if hasattr(rc, "output_pressure_correction_enabled"):
        rc.output_pressure_correction_enabled = bool(scale > 0.0)
    # keep original enabled flag
    return (
        prev_core,
        prev_op1,
        prev_op2,
        prev_post_scale,
        prev_post_enabled,
        prev_out_enabled,
        prev_out_scale,
        prev_out_proj_scale,
    )


def _restore_pressure_correction_scale(
    model: torch.nn.Module, prev: Tuple[float, float, float, float, bool, bool, float, float] | None
) -> None:
    if prev is None:
        return
    rc = getattr(model, "rollout_core", None)
    if rc is None or not hasattr(rc, "pressure_operator"):
        return
    rc.pressure_correction_scale = float(prev[0])
    rc.pressure_operator.correction_scale = float(prev[1])
    if hasattr(rc, "pressure_operator_post"):
        rc.pressure_operator_post.correction_scale = float(prev[2])
    if hasattr(rc, "pressure_post_p2g_scale"):
        rc.pressure_post_p2g_scale = float(prev[3])
    if hasattr(rc, "pressure_post_p2g_enabled"):
        rc.pressure_post_p2g_enabled = bool(prev[4])
    if hasattr(rc, "output_pressure_correction_enabled"):
        rc.output_pressure_correction_enabled = bool(prev[5])
    if hasattr(rc, "output_pressure_correction_scale"):
        rc.output_pressure_correction_scale = float(prev[6])
    if hasattr(rc, "output_velocity_projection"):
        rc.output_velocity_projection.correction_scale = float(prev[7])


def build_loader(cfg: Dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    ds_cfg = cfg["dataset"]
    run_cfg = cfg["train"]
    dataset = InteractionRolloutDataset(
        dataset_root=ds_cfg["root"],
        manifest_path=ds_cfg.get("manifest", "manifest.jsonl"),
        split=split,
        valid_status=tuple(ds_cfg.get("valid_status", ["ok", "skipped_existing"])),
        rollout_steps=int(ds_cfg.get("rollout_steps", 8)),
        control_dim=int(cfg["model"].get("control_dim", 16)),
        velocity_scale=float(ds_cfg.get("velocity_scale", 1.0)),
        max_episodes=int(ds_cfg.get(f"max_{split}_episodes", 0)),
        max_episode_cache=int(ds_cfg.get("max_episode_cache", 1)),
        particle_gt_num_points=int(ds_cfg.get("particle_gt_num_points", 256)),
        load_particle_gt=bool(ds_cfg.get("load_particle_gt", True)),
        object_patch_size=int(ds_cfg.get("object_patch_size", 16)),
        canonical_object_patches=bool(ds_cfg.get("canonical_object_patches", True)),
    )
    return DataLoader(
        dataset,
        batch_size=int(run_cfg.get("batch_size", 1)),
        shuffle=bool(shuffle),
        num_workers=int(run_cfg.get("num_workers", 0)),
        pin_memory=bool(run_cfg.get("pin_memory", False)),
        drop_last=False,
    )


def _finite_diff(x: torch.Tensor, dim: int) -> torch.Tensor:
    slices_hi = [slice(None)] * x.ndim
    slices_lo = [slice(None)] * x.ndim
    slices_hi[dim] = slice(1, None)
    slices_lo[dim] = slice(0, -1)
    diff = x[tuple(slices_hi)] - x[tuple(slices_lo)]
    pad_shape = list(x.shape)
    pad_shape[dim] = 1
    pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([diff, pad], dim=dim)


def _grad_xyz(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # v: [B,T,C,D,H,W]
    dv_dz = _finite_diff(v, dim=3)
    dv_dy = _finite_diff(v, dim=4)
    dv_dx = _finite_diff(v, dim=5)
    return dv_dx, dv_dy, dv_dz


def divergence_3d(v: torch.Tensor) -> torch.Tensor:
    # v: [B,T,3,D,H,W] -> [B,T,1,D,H,W]
    du_dx = _finite_diff(v[:, :, 0:1], dim=5)
    dv_dy = _finite_diff(v[:, :, 1:2], dim=4)
    dw_dz = _finite_diff(v[:, :, 2:3], dim=3)
    return du_dx + dv_dy + dw_dz


def vorticity_3d(v: torch.Tensor) -> torch.Tensor:
    # curl(u): [B,T,3,D,H,W]
    du_dz = _finite_diff(v[:, :, 0:1], dim=3)
    du_dy = _finite_diff(v[:, :, 0:1], dim=4)
    dv_dx = _finite_diff(v[:, :, 1:2], dim=5)
    dv_dz = _finite_diff(v[:, :, 1:2], dim=3)
    dw_dx = _finite_diff(v[:, :, 2:3], dim=5)
    dw_dy = _finite_diff(v[:, :, 2:3], dim=4)

    w_x = dw_dy - dv_dz
    w_y = du_dz - dw_dx
    w_z = dv_dx - du_dy
    return torch.cat([w_x, w_y, w_z], dim=2)


def _recon_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    norm: str = "l1",
    spatial_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if norm == "l2":
        diff = (pred - gt) ** 2
    else:
        diff = torch.abs(pred - gt)
    if spatial_weight is None:
        return torch.mean(diff)
    # spatial_weight: [B,1,D,H,W], diff: [B,C,D,H,W]
    w = spatial_weight.expand(-1, diff.shape[1], -1, -1, -1)
    denom = w.sum().clamp_min(1e-8)
    return (diff * w).sum() / denom


def rollout_weighted_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    weights: torch.Tensor,
    spatial_weight: torch.Tensor | None = None,
    norm: str = "l1",
) -> torch.Tensor:
    # pred,gt: [B,T,C,D,H,W], weights: [T]
    if norm == "l2":
        diff = (pred - gt) ** 2
    else:
        diff = torch.abs(pred - gt)
    if spatial_weight is None:
        per = torch.mean(diff, dim=(2, 3, 4, 5))
    else:
        # spatial_weight: [B,T,1,D,H,W]
        w_map = spatial_weight.expand(-1, -1, diff.shape[2], -1, -1, -1)
        denom = w_map.sum(dim=(2, 3, 4, 5)).clamp_min(1e-8)
        per = (diff * w_map).sum(dim=(2, 3, 4, 5)) / denom
    w = weights[None, :].to(per.device)
    return torch.mean(torch.sum(per * w, dim=1) / (torch.sum(w, dim=1).clamp_min(1e-12)))


def _interface_focus_weight(
    gt_phase: torch.Tensor,
    target_interface_sdf: torch.Tensor | None,
    reweight_cfg: Dict[str, Any],
) -> torch.Tensor | None:
    if not bool(reweight_cfg.get("enabled", False)):
        return None
    mode = str(reweight_cfg.get("mode", "phase_grad"))
    alpha = float(reweight_cfg.get("alpha", 2.0))
    clip = float(reweight_cfg.get("clip", 6.0))
    eps = 1e-8

    if mode in ("sdf", "object_sdf") and target_interface_sdf is not None:
        sigma = float(reweight_cfg.get("sigma", 0.25))
        sdf = torch.abs(target_interface_sdf)
        weight = 1.0 + alpha * torch.exp(-sdf / max(sigma, 1e-6))
        return weight.clamp(1.0, clip)

    # Default: phase-gradient focus.
    liq = gt_phase[:, :, 0:1]
    g_dx, g_dy, g_dz = _grad_xyz(liq)
    grad_mag = torch.sqrt(g_dx.pow(2) + g_dy.pow(2) + g_dz.pow(2) + eps)
    norm = grad_mag.mean(dim=(3, 4, 5), keepdim=True).clamp_min(eps)
    focus = (grad_mag / norm).clamp(0.0, clip)
    weight = 1.0 + alpha * focus
    return weight.clamp(1.0, clip)


def _phase_ce_loss(pred_phase: torch.Tensor | None, gt_phase: torch.Tensor) -> torch.Tensor:
    if pred_phase is None or pred_phase.shape[2] <= 1:
        return torch.tensor(0.0, dtype=gt_phase.dtype, device=gt_phase.device)
    bsz, ts, c, d, h, w = pred_phase.shape
    target_cls = torch.argmax(gt_phase, dim=2).long()  # [B,T,D,H,W]
    return F.cross_entropy(
        pred_phase.reshape(bsz * ts, c, d, h, w),
        target_cls.reshape(bsz * ts, d, h, w),
    )


def _masked_velocity_error(
    pred_velocity: torch.Tensor,
    gt_velocity: torch.Tensor,
    mask: torch.Tensor,
    norm: str,
) -> torch.Tensor:
    # pred/gt: [B,T,3,D,H,W], mask: [B,T,1,D,H,W]
    if norm == "l2":
        diff = (pred_velocity - gt_velocity) ** 2
    else:
        diff = torch.abs(pred_velocity - gt_velocity)
    w = mask.expand_as(diff)
    denom = w.sum().clamp_min(1e-8)
    return (diff * w).sum() / denom


def _flow_length_error(pred_velocity: torch.Tensor, gt_velocity: torch.Tensor) -> torch.Tensor:
    pred_mag = torch.norm(pred_velocity, dim=2)
    gt_mag = torch.norm(gt_velocity, dim=2)
    return torch.mean(torch.abs(pred_mag - gt_mag))


def _flow_direction_cosine(pred_velocity: torch.Tensor, gt_velocity: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    dot = torch.sum(pred_velocity * gt_velocity, dim=2)
    pred_mag = torch.norm(pred_velocity, dim=2)
    gt_mag = torch.norm(gt_velocity, dim=2)
    denom = (pred_mag * gt_mag).clamp_min(eps)
    cos = (dot / denom).clamp(-1.0, 1.0)
    valid = (pred_mag > eps) & (gt_mag > eps)
    if valid.any():
        return torch.mean(cos[valid])
    return torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)


def _t1_near_weight(
    target_sdf: torch.Tensor | None,
    fluid_mask: torch.Tensor | None,
    cfg: Dict[str, Any],
) -> torch.Tensor | None:
    if target_sdf is None:
        return None
    if not bool(cfg.get("enabled", False)):
        return None
    mode = str(cfg.get("mode", "object_sdf"))
    if mode not in ("sdf", "object_sdf"):
        return None
    sigma = float(cfg.get("sigma", 0.15))
    alpha = float(cfg.get("alpha", 3.0))
    clip = float(cfg.get("clip", 8.0))
    weight = 1.0 + alpha * torch.exp(-torch.abs(target_sdf) / max(sigma, 1e-6))
    if bool(cfg.get("use_fluid_mask", True)) and fluid_mask is not None:
        weight = weight * (fluid_mask > 0.5).to(dtype=weight.dtype, device=weight.device)
    if float(weight.sum().detach().item()) <= 0.0:
        return None
    return weight.clamp(0.0, clip)


def _slice_temporal_batch(batch: Dict[str, torch.Tensor], steps: int) -> Dict[str, torch.Tensor]:
    steps = int(max(1, steps))
    sliced: Dict[str, torch.Tensor] = dict(batch)
    for key in TIME_SERIES_KEYS:
        if key in sliced and isinstance(sliced[key], torch.Tensor):
            sliced[key] = sliced[key][:, :steps]
    if "particle_pos_seq" in sliced and isinstance(sliced["particle_pos_seq"], torch.Tensor):
        # particle positions include start frame -> steps + 1.
        sliced["particle_pos_seq"] = sliced["particle_pos_seq"][:, : (steps + 1)]
    return sliced


def sample_seed_points(mask_bdhw: torch.Tensor, num_points: int) -> torch.Tensor:
    # mask_bdhw: [B,D,H,W] -> points [B,N,3] in voxel coords (x,y,z)
    bsz, d, h, w = mask_bdhw.shape
    out = []
    for b in range(bsz):
        idx = torch.nonzero(mask_bdhw[b] > 0.5, as_tuple=False)
        if idx.shape[0] == 0:
            z = torch.randint(0, d, (num_points,), device=mask_bdhw.device)
            y = torch.randint(0, h, (num_points,), device=mask_bdhw.device)
            x = torch.randint(0, w, (num_points,), device=mask_bdhw.device)
            pts = torch.stack([x.float(), y.float(), z.float()], dim=1)
            out.append(pts)
            continue

        if idx.shape[0] >= num_points:
            perm = torch.randperm(idx.shape[0], device=mask_bdhw.device)[:num_points]
            sel = idx[perm]
        else:
            rep = torch.randint(0, idx.shape[0], (num_points,), device=mask_bdhw.device)
            sel = idx[rep]
        # idx columns: z,y,x -> convert to x,y,z
        pts = torch.stack([sel[:, 2].float(), sel[:, 1].float(), sel[:, 0].float()], dim=1)
        out.append(pts)
    return torch.stack(out, dim=0)


def sample_velocity_at_points(vel_bcdhw: torch.Tensor, pts_bnx: torch.Tensor) -> torch.Tensor:
    # vel: [B,3,D,H,W], pts: [B,N,3] voxel coords (x,y,z)
    bsz, _, d, h, w = vel_bcdhw.shape
    n = pts_bnx.shape[1]

    x = pts_bnx[..., 0].clamp(0, max(1, w - 1))
    y = pts_bnx[..., 1].clamp(0, max(1, h - 1))
    z = pts_bnx[..., 2].clamp(0, max(1, d - 1))

    gx = 2.0 * x / max(1.0, float(w - 1)) - 1.0
    gy = 2.0 * y / max(1.0, float(h - 1)) - 1.0
    gz = 2.0 * z / max(1.0, float(d - 1)) - 1.0
    grid = torch.stack([gx, gy, gz], dim=-1).view(bsz, n, 1, 1, 3)

    sampled = F.grid_sample(
        vel_bcdhw,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    # [B,3,N,1,1] -> [B,N,3]
    return sampled[:, :, :, 0, 0].transpose(1, 2)


def trajectory_loss_particle_gt(
    pred_velocity_norm: torch.Tensor,
    particle_pos_start: torch.Tensor,
    particle_pos_seq_gt: torch.Tensor,
    particle_valid: torch.Tensor,
    velocity_scale: float,
    step_mode: str,
    dt_value: float,
    norm: str = "l2",
) -> torch.Tensor:
    per_step = trajectory_error_particle_per_step(
        pred_velocity_norm=pred_velocity_norm,
        particle_pos_start=particle_pos_start,
        particle_pos_seq_gt=particle_pos_seq_gt,
        particle_valid=particle_valid,
        velocity_scale=velocity_scale,
        step_mode=step_mode,
        dt_value=dt_value,
        norm=norm,
    )
    if per_step.numel() == 0:
        return torch.tensor(0.0, dtype=pred_velocity_norm.dtype, device=pred_velocity_norm.device)
    return torch.mean(per_step)


def trajectory_error_particle_per_step(
    pred_velocity_norm: torch.Tensor,
    particle_pos_start: torch.Tensor,
    particle_pos_seq_gt: torch.Tensor,
    particle_valid: torch.Tensor,
    velocity_scale: float,
    step_mode: str,
    dt_value: float,
    norm: str = "l2",
) -> torch.Tensor:
    # pred_velocity_norm: [B,T,3,D,H,W]
    # particle_pos_start: [B,P,3], particle_pos_seq_gt: [B,T+1,P,3], particle_valid: [B,P]
    bsz, t, _, d, h, w = pred_velocity_norm.shape
    p_pred = particle_pos_start.clone()
    valid = particle_valid.float().clamp(0.0, 1.0)  # [B,P]
    vden = valid.sum().clamp_min(1e-8)
    losses: List[torch.Tensor] = []

    for i in range(t):
        # Convert normalized prediction back to voxel displacement per step.
        step = pred_velocity_norm[:, i] * float(velocity_scale)
        if step_mode == "velocity":
            step = step * float(dt_value)
        elif step_mode != "displacement":
            raise ValueError(f"trajectory_step_mode must be 'displacement' or 'velocity', got {step_mode}")

        vp = sample_velocity_at_points(step, p_pred)
        p_pred = p_pred + vp
        p_pred[..., 0].clamp_(0.0, float(max(1, w - 1)))
        p_pred[..., 1].clamp_(0.0, float(max(1, h - 1)))
        p_pred[..., 2].clamp_(0.0, float(max(1, d - 1)))

        gt_pos = particle_pos_seq_gt[:, i + 1]
        delta = p_pred - gt_pos
        if norm == "l1":
            step_loss = torch.sum(torch.abs(delta), dim=-1)
        else:
            step_loss = torch.norm(delta, dim=-1, p=2)
        losses.append((step_loss * valid).sum() / vden)

    if not losses:
        return torch.zeros((0,), dtype=pred_velocity_norm.dtype, device=pred_velocity_norm.device)
    return torch.stack(losses)


def dual_space_consistency_loss(
    pred_velocity_norm: torch.Tensor,
    gt_velocity_norm: torch.Tensor,
    particle_pos_seq_gt: torch.Tensor,
    particle_valid: torch.Tensor,
    velocity_scale: float,
    step_mode: str,
    dt_value: float,
    norm: str = "l1",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Eulerian-Lagrangian dual consistency:
    - E2L: predicted field sampled at GT particle positions should match GT particle transport.
    - L2E: predicted transport sampled along predicted trajectories should match GT Eulerian field.
    """
    bsz, t, _, d, h, w = pred_velocity_norm.shape
    valid = particle_valid.float().clamp(0.0, 1.0)  # [B,P]
    vden = valid.sum().clamp_min(1e-8)
    eps = 1e-8

    pred_step_field = pred_velocity_norm * float(velocity_scale)
    gt_step_field = gt_velocity_norm * float(velocity_scale)
    if step_mode == "velocity":
        pred_step_field = pred_step_field * float(dt_value)
        gt_step_field = gt_step_field * float(dt_value)
    elif step_mode != "displacement":
        raise ValueError(f"trajectory_step_mode must be 'displacement' or 'velocity', got {step_mode}")

    p_gt = particle_pos_seq_gt[:, :-1]  # [B,T,P,3]
    gt_disp = particle_pos_seq_gt[:, 1:] - particle_pos_seq_gt[:, :-1]  # [B,T,P,3]

    # E2L consistency at GT particle locations.
    e2l_steps: List[torch.Tensor] = []
    for i in range(t):
        pred_step_at_gt = sample_velocity_at_points(pred_step_field[:, i], p_gt[:, i])
        delta = pred_step_at_gt - gt_disp[:, i]
        if norm == "l2":
            step_loss = torch.norm(delta, dim=-1, p=2)
        else:
            step_loss = torch.sum(torch.abs(delta), dim=-1)
        e2l_steps.append((step_loss * valid).sum() / vden)
    l_e2l = torch.mean(torch.stack(e2l_steps)) if e2l_steps else torch.tensor(0.0, device=pred_velocity_norm.device)

    # L2E consistency along predicted trajectories.
    p_pred = particle_pos_seq_gt[:, 0].clone()
    l2e_steps: List[torch.Tensor] = []
    for i in range(t):
        pred_step = sample_velocity_at_points(pred_step_field[:, i], p_pred)
        gt_step_at_pred = sample_velocity_at_points(gt_step_field[:, i], p_pred)
        delta = pred_step - gt_step_at_pred
        if norm == "l2":
            step_loss = torch.norm(delta, dim=-1, p=2)
        else:
            step_loss = torch.sum(torch.abs(delta), dim=-1)
        l2e_steps.append((step_loss * valid).sum() / vden)

        p_pred = p_pred + pred_step
        p_pred[..., 0].clamp_(0.0, float(max(1, w - 1)))
        p_pred[..., 1].clamp_(0.0, float(max(1, h - 1)))
        p_pred[..., 2].clamp_(0.0, float(max(1, d - 1)))
    l_l2e = torch.mean(torch.stack(l2e_steps)) if l2e_steps else torch.tensor(0.0, device=pred_velocity_norm.device)

    l_dual = l_e2l + l_l2e
    if torch.isnan(l_dual) or torch.isinf(l_dual):
        z = torch.tensor(0.0, dtype=pred_velocity_norm.dtype, device=pred_velocity_norm.device)
        return z, z, z
    return l_dual, l_e2l.clamp_min(eps), l_l2e.clamp_min(eps)


def interface_consistency_loss(
    pred_velocity: torch.Tensor,
    pred_phase: torch.Tensor | None,
    target_phase: torch.Tensor,
    boundary_velocity: torch.Tensor,
    boundary_mask: torch.Tensor,
    interface_normal: torch.Tensor,
    int_cfg: Dict[str, Any],
) -> torch.Tensor:
    # all tensors use [B,T,C,D,H,W]
    eps = 1e-8
    w_bv = float(int_cfg.get("boundary_velocity_weight", 1.0))
    w_np = float(int_cfg.get("no_penetration_weight", 1.0))
    w_n = float(int_cfg.get("normal_consistency_weight", 1.0))
    grad_focus_threshold = float(int_cfg.get("phase_grad_focus_threshold", 0.02))

    mask = boundary_mask.clamp(0.0, 1.0)
    denom = mask.sum().clamp_min(eps)

    # 1) boundary velocity matching
    l_bv = (torch.abs(pred_velocity - boundary_velocity) * mask).sum() / (denom * 3.0)

    # 2) no-penetration (normal component of relative velocity)
    n = F.normalize(interface_normal, dim=2, eps=eps)
    rel = pred_velocity - boundary_velocity
    rel_n = torch.sum(rel * n, dim=2, keepdim=True)
    l_np = (torch.abs(rel_n) * mask).sum() / denom

    # 3) interface normal consistency (if phase head exists)
    l_nrm = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if pred_phase is not None:
        # pred_phase is logits.
        pred_prob = torch.softmax(pred_phase, dim=2)
        pred_liq = pred_prob[:, :, 0:1]
        gt_liq = target_phase[:, :, 0:1]
        p_dx, p_dy, p_dz = _grad_xyz(pred_liq)
        g_dx, g_dy, g_dz = _grad_xyz(gt_liq)
        pred_n = F.normalize(torch.cat([p_dx, p_dy, p_dz], dim=2), dim=2, eps=eps)
        gt_n = F.normalize(torch.cat([g_dx, g_dy, g_dz], dim=2), dim=2, eps=eps)
        # Interface focus mask from raw (pre-normalization) GT gradient magnitude.
        gt_grad_mag = torch.sqrt(g_dx.pow(2) + g_dy.pow(2) + g_dz.pow(2) + eps)
        m = (gt_grad_mag > grad_focus_threshold).float()
        m_denom = m.sum().clamp_min(eps)
        cos = torch.sum(pred_n * gt_n, dim=2, keepdim=True).clamp(-1.0, 1.0)
        l_nrm = ((1.0 - cos) * m).sum() / m_denom

    return w_bv * l_bv + w_np * l_np + w_n * l_nrm


def compute_losses(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred_velocity = out["velocity"]
    gt_velocity = batch["target_velocity"]
    gt_phase = batch["target_phase"]
    pred_phase = out.get("phase", None)
    pred_particle = out.get("particle_pos", None)

    loss_cfg = cfg.get("loss", {})
    weights_cfg = cfg.get("loss_weights", {})

    vel_norm = str(loss_cfg.get("velocity_norm", "l1"))
    grad_norm = str(loss_cfg.get("gradient_norm", "l1"))
    traj_norm = str(loss_cfg.get("trajectory_norm", "l2"))
    traj_step_mode = str(loss_cfg.get("trajectory_step_mode", "displacement"))
    rollout_gamma = float(loss_cfg.get("rollout_gamma", 1.0))
    vort_weight = float(loss_cfg.get("vorticity_weight", 0.0))
    velocity_scale = float(cfg.get("dataset", {}).get("velocity_scale", 1.0))
    dt_value = float(batch.get("target_dt", torch.tensor(1.0, device=pred_velocity.device)).mean().item())
    reweight_cfg = loss_cfg.get("interface_reweight", {})

    spatial_w_all = _interface_focus_weight(
        gt_phase=gt_phase,
        target_interface_sdf=batch.get("target_object_sdf", batch.get("target_interface_sdf", None)),
        reweight_cfg=reweight_cfg,
    )
    spatial_w_t0 = spatial_w_all[:, 0] if spatial_w_all is not None else None

    l11 = float(weights_cfg.get("lambda_vel_near_t1", 0.0))
    l12 = float(weights_cfg.get("lambda_mag_t1", 0.0))
    l13 = float(weights_cfg.get("lambda_resp_t1", 0.0))

    # L_vel (1-step main recon)
    l_vel = _recon_loss(pred_velocity[:, 0], gt_velocity[:, 0], norm=vel_norm, spatial_weight=spatial_w_t0)

    l_vel_near_t1 = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    l_mag_t1 = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    l_resp_t1 = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    pred_speed_mean_t1 = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    gt_speed_mean_t1 = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    pred_gt_speed_ratio_t1 = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if (l11 > 0.0 or l12 > 0.0 or l13 > 0.0) and "fluid_init" in batch:
        target_sdf_t1 = batch.get("target_object_sdf", batch.get("target_interface_sdf", None))
        if isinstance(target_sdf_t1, torch.Tensor):
            target_sdf_t1 = target_sdf_t1[:, 0]
        target_fluid_mask_t1 = batch.get("target_fluid_mask", None)
        if isinstance(target_fluid_mask_t1, torch.Tensor):
            target_fluid_mask_t1 = target_fluid_mask_t1[:, 0]
        near_weight_t1 = _t1_near_weight(
            target_sdf=target_sdf_t1,
            fluid_mask=target_fluid_mask_t1,
            cfg=loss_cfg.get("t1_near_supervision", {}),
        )
        if near_weight_t1 is not None:
            pred_t1 = pred_velocity[:, 0]
            gt_t1 = gt_velocity[:, 0]
            if l11 > 0.0:
                l_vel_near_t1 = _recon_loss(pred_t1, gt_t1, norm=vel_norm, spatial_weight=near_weight_t1)
            pred_speed_t1 = torch.norm(pred_t1, dim=1, keepdim=True)
            gt_speed_t1 = torch.norm(gt_t1, dim=1, keepdim=True)
            if l12 > 0.0:
                l_mag_t1 = _recon_loss(pred_speed_t1, gt_speed_t1, norm=vel_norm, spatial_weight=near_weight_t1)
            current_velocity = batch["fluid_init"][:, :3]
            pred_resp_t1 = pred_t1 - current_velocity
            gt_resp_t1 = gt_t1 - current_velocity
            if l13 > 0.0:
                l_resp_t1 = _recon_loss(pred_resp_t1, gt_resp_t1, norm=vel_norm, spatial_weight=near_weight_t1)
            weight_sum = near_weight_t1.sum().clamp_min(1e-8)
            pred_speed_mean_t1 = (pred_speed_t1 * near_weight_t1).sum() / weight_sum
            gt_speed_mean_t1 = (gt_speed_t1 * near_weight_t1).sum() / weight_sum
            pred_gt_speed_ratio_t1 = pred_speed_mean_t1 / gt_speed_mean_t1.clamp_min(1e-8)

    # L_grad (+ optional vorticity)
    p_dx, p_dy, p_dz = _grad_xyz(pred_velocity)
    g_dx, g_dy, g_dz = _grad_xyz(gt_velocity)
    if grad_norm == "l2":
        l_grad_core = (
            F.mse_loss(p_dx, g_dx) + F.mse_loss(p_dy, g_dy) + F.mse_loss(p_dz, g_dz)
        ) / 3.0
    else:
        l_grad_core = (
            torch.mean(torch.abs(p_dx - g_dx))
            + torch.mean(torch.abs(p_dy - g_dy))
            + torch.mean(torch.abs(p_dz - g_dz))
        ) / 3.0
    l_grad = l_grad_core
    if vort_weight > 0.0:
        p_w = vorticity_3d(pred_velocity)
        g_w = vorticity_3d(gt_velocity)
        if grad_norm == "l2":
            l_vort = F.mse_loss(p_w, g_w)
        else:
            l_vort = torch.mean(torch.abs(p_w - g_w))
        l_grad = l_grad + vort_weight * l_vort

    # L_div
    l_div = torch.mean(torch.abs(divergence_3d(pred_velocity)))

    # L_traj (particle GT supervision)
    traj_per_step = trajectory_error_particle_per_step(
        pred_velocity_norm=pred_velocity,
        particle_pos_start=batch["particle_pos_start"],
        particle_pos_seq_gt=batch["particle_pos_seq"],
        particle_valid=batch["particle_valid"],
        velocity_scale=velocity_scale,
        step_mode=traj_step_mode,
        dt_value=dt_value,
        norm=traj_norm,
    )
    if traj_per_step.numel() > 0:
        l_traj = torch.mean(traj_per_step)
        trajectory_endpoint_error = traj_per_step[-1]
    else:
        l_traj = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
        trajectory_endpoint_error = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)

    # Optional explicit particle branch supervision (for dual-space rollout module).
    if (
        isinstance(pred_particle, torch.Tensor)
        and pred_particle.ndim == 4
        and pred_particle.shape[1] == (batch["particle_pos_seq"].shape[1])
    ):
        gt_particle = batch["particle_pos_seq"]
        valid = batch["particle_valid"].float().clamp(0.0, 1.0)  # [B,P]
        delta_particle = pred_particle[:, 1:] - gt_particle[:, 1:]  # [B,T,P,3]
        per_particle = torch.norm(delta_particle, dim=-1, p=2)  # [B,T,P]
        denom = (valid.sum() * max(1, delta_particle.shape[1])).clamp_min(1e-8)
        l_particle_module = (per_particle * valid[:, None, :]).sum() / denom
    else:
        l_particle_module = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)

    # L_dual (Eulerian-Lagrangian consistency)
    l_dual, _, _ = dual_space_consistency_loss(
        pred_velocity_norm=pred_velocity,
        gt_velocity_norm=gt_velocity,
        particle_pos_seq_gt=batch["particle_pos_seq"],
        particle_valid=batch["particle_valid"],
        velocity_scale=velocity_scale,
        step_mode=traj_step_mode,
        dt_value=dt_value,
        norm=vel_norm,
    )

    # L_int
    l_int = interface_consistency_loss(
        pred_velocity=pred_velocity,
        pred_phase=pred_phase,
        target_phase=gt_phase,
        boundary_velocity=batch["target_boundary_velocity"],
        boundary_mask=batch["target_boundary_mask"],
        interface_normal=batch["target_interface_normal"],
        int_cfg=loss_cfg.get("interface", {}),
    )

    # L_roll (multi-step weighted error)
    t = pred_velocity.shape[1]
    idx = torch.arange(t, device=pred_velocity.device, dtype=pred_velocity.dtype)
    w = torch.pow(torch.tensor(rollout_gamma, device=pred_velocity.device, dtype=pred_velocity.dtype), idx)
    l_roll = rollout_weighted_loss(
        pred_velocity,
        gt_velocity,
        weights=w,
        spatial_weight=spatial_w_all,
        norm=vel_norm,
    )
    spatial_w_last = spatial_w_all[:, -1] if spatial_w_all is not None else None
    rollout_endpoint_error = _recon_loss(
        pred_velocity[:, -1], gt_velocity[:, -1], norm=vel_norm, spatial_weight=spatial_w_last
    )

    # Optional phase CE (logits vs class index)
    l_phase = _phase_ce_loss(pred_phase=pred_phase, gt_phase=gt_phase)

    # Interface-near / bulk-far decomposition for interaction-centric evaluation.
    target_obj_sdf = batch.get("target_object_sdf", batch.get("target_interface_sdf", None))
    near_vel_error = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    far_vel_error = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    near_voxel_count = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    far_voxel_count = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    near_voxel_frac = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    far_voxel_frac = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    mask_valid_voxel_count = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    mask_valid_voxel_frac = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    near_sdf_threshold_used = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    far_sdf_threshold_used = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    mask_quantile_fallback_used = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    near_sdf_mean = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    far_sdf_mean = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    near_far_overlap_frac = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(target_obj_sdf, torch.Tensor):
        abs_sdf = torch.abs(target_obj_sdf)
        near_thresh_cfg = float(loss_cfg.get("near_interface_threshold", 1.5))
        far_thresh_cfg = float(loss_cfg.get("far_interface_threshold", 4.0))
        near_sdf_threshold_used = torch.tensor(near_thresh_cfg, dtype=pred_velocity.dtype, device=pred_velocity.device)
        far_sdf_threshold_used = torch.tensor(far_thresh_cfg, dtype=pred_velocity.dtype, device=pred_velocity.device)

        valid_mask = torch.ones_like(abs_sdf)
        if bool(loss_cfg.get("use_fluid_mask_for_near_far", True)):
            fluid_mask = batch.get("target_fluid_mask", None)
            if isinstance(fluid_mask, torch.Tensor):
                fm = fluid_mask.to(device=abs_sdf.device, dtype=abs_sdf.dtype)
                if fm.ndim == abs_sdf.ndim - 2:  # [B,D,H,W]
                    fm = fm[:, None, None]
                elif fm.ndim == abs_sdf.ndim - 1:  # [B,T,D,H,W] or [B,1,D,H,W]
                    if fm.shape[1] == abs_sdf.shape[1]:
                        fm = fm.unsqueeze(2)
                    else:
                        fm = fm.unsqueeze(1)
                elif fm.ndim < abs_sdf.ndim:
                    while fm.ndim < abs_sdf.ndim:
                        fm = fm.unsqueeze(1)
                if fm.ndim == abs_sdf.ndim and fm.shape[1] == 1 and abs_sdf.shape[1] > 1:
                    fm = fm.expand(-1, abs_sdf.shape[1], -1, -1, -1, -1)
                if fm.ndim == abs_sdf.ndim and fm.shape[2] != abs_sdf.shape[2]:
                    fm = fm[:, :, : abs_sdf.shape[2]]
                if fm.shape != abs_sdf.shape:
                    try:
                        fm = fm.expand_as(abs_sdf)
                    except RuntimeError:
                        fm = torch.ones_like(abs_sdf)
                valid_mask = (fm > 0.5).float()

        total_voxel_count = torch.tensor(float(abs_sdf.numel()), dtype=pred_velocity.dtype, device=pred_velocity.device).clamp_min(1.0)
        mask_valid_voxel_count = valid_mask.sum()
        if float(mask_valid_voxel_count.detach().item()) <= 0.0:
            valid_mask = torch.ones_like(abs_sdf)
            mask_valid_voxel_count = valid_mask.sum()
        mask_valid_voxel_frac = mask_valid_voxel_count / total_voxel_count

        near_mask = (abs_sdf <= near_thresh_cfg).float() * valid_mask
        far_mask = (abs_sdf >= far_thresh_cfg).float() * valid_mask
        near_voxel_count = near_mask.sum()
        far_voxel_count = far_mask.sum()

        overlap_count = (near_mask * far_mask).sum()
        overlap_frac = overlap_count / torch.minimum(
            near_voxel_count.clamp_min(1.0),
            far_voxel_count.clamp_min(1.0),
        )

        # If one side collapses or near/far overlap too much, fallback to quantile/rank split.
        max_overlap = float(loss_cfg.get("near_far_max_overlap", 0.2))
        need_fallback = (
            float(near_voxel_count.detach().item()) <= 0.0
            or float(far_voxel_count.detach().item()) <= 0.0
            or float(overlap_frac.detach().item()) > max_overlap
        )
        if bool(loss_cfg.get("auto_quantile_fallback", True)) and need_fallback:
            mask_quantile_fallback_used = torch.tensor(1.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
            valid_vals = abs_sdf[valid_mask > 0.5]
            near_q = float(loss_cfg.get("near_quantile", 0.25))
            far_q = float(loss_cfg.get("far_quantile", 0.75))
            near_q = min(1.0, max(0.0, near_q))
            far_q = min(1.0, max(near_q, far_q))
            if valid_vals.numel() > 0:
                near_thr_q = torch.quantile(valid_vals, near_q)
                far_thr_q = torch.quantile(valid_vals, far_q)
                near_sdf_threshold_used = near_thr_q.to(dtype=pred_velocity.dtype)
                far_sdf_threshold_used = far_thr_q.to(dtype=pred_velocity.dtype)
                near_mask = (abs_sdf <= near_thr_q).float() * valid_mask
                far_mask = (abs_sdf >= far_thr_q).float() * valid_mask
                near_voxel_count = near_mask.sum()
                far_voxel_count = far_mask.sum()
                overlap_count = (near_mask * far_mask).sum()
                overlap_frac = overlap_count / torch.minimum(
                    near_voxel_count.clamp_min(1.0),
                    far_voxel_count.clamp_min(1.0),
                )

            # Last-resort robust split on sorted valid voxels (disjoint near/far sets).
            if (
                float(near_voxel_count.detach().item()) <= 0.0
                or float(far_voxel_count.detach().item()) <= 0.0
                or float(overlap_frac.detach().item()) > max_overlap
            ):
                flat_abs = abs_sdf.reshape(-1)
                flat_valid = (valid_mask.reshape(-1) > 0.5)
                valid_idx = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1)
                if valid_idx.numel() > 0:
                    sorted_idx = valid_idx[torch.argsort(flat_abs[valid_idx])]
                    n_valid = int(sorted_idx.numel())
                    k_near = max(1, int(round(near_q * n_valid)))
                    k_far = max(1, int(round((1.0 - far_q) * n_valid)))
                    if k_near + k_far > n_valid:
                        k_near = max(1, n_valid // 2)
                        k_far = max(1, n_valid - k_near)
                    near_idx = sorted_idx[:k_near]
                    far_idx = sorted_idx[n_valid - k_far :]
                    near_flat = torch.zeros_like(flat_abs, dtype=pred_velocity.dtype)
                    far_flat = torch.zeros_like(flat_abs, dtype=pred_velocity.dtype)
                    near_flat[near_idx] = 1.0
                    far_flat[far_idx] = 1.0
                    near_mask = near_flat.view_as(abs_sdf)
                    far_mask = far_flat.view_as(abs_sdf)
                    near_voxel_count = near_mask.sum()
                    far_voxel_count = far_mask.sum()
                    overlap_count = (near_mask * far_mask).sum()
                    overlap_frac = overlap_count / torch.minimum(
                        near_voxel_count.clamp_min(1.0),
                        far_voxel_count.clamp_min(1.0),
                    )

        near_far_overlap_frac = overlap_frac
        denom_valid = mask_valid_voxel_count.clamp_min(1.0)
        near_voxel_frac = near_voxel_count / denom_valid
        far_voxel_frac = far_voxel_count / denom_valid

        if float(near_voxel_count.detach().item()) > 0.0:
            near_vel_error = _masked_velocity_error(
                pred_velocity=pred_velocity,
                gt_velocity=gt_velocity,
                mask=near_mask,
                norm=vel_norm,
            )
            near_sdf_mean = (abs_sdf * near_mask).sum() / near_voxel_count.clamp_min(1.0)
        if float(far_voxel_count.detach().item()) > 0.0:
            far_vel_error = _masked_velocity_error(
                pred_velocity=pred_velocity,
                gt_velocity=gt_velocity,
                mask=far_mask,
                norm=vel_norm,
            )
            far_sdf_mean = (abs_sdf * far_mask).sum() / far_voxel_count.clamp_min(1.0)

    l1 = float(weights_cfg.get("lambda_vel", 1.0))
    l2 = float(weights_cfg.get("lambda_grad", 0.2))
    l3 = float(weights_cfg.get("lambda_div", 0.1))
    l4 = float(weights_cfg.get("lambda_traj", 0.5))
    l5 = float(weights_cfg.get("lambda_int", 0.2))
    l6 = float(weights_cfg.get("lambda_roll", 0.5))
    l7 = float(weights_cfg.get("lambda_phase", 0.0))
    l8 = float(weights_cfg.get("lambda_dual", 0.5))
    l9 = float(weights_cfg.get("lambda_exchange", 0.1))
    l10 = float(weights_cfg.get("lambda_particle_module", 0.2))

    diagnostics = out.get("diagnostics", None)
    if isinstance(diagnostics, dict) and ("interaction_intensity_mean" in diagnostics):
        interaction_intensity_mean = torch.mean(diagnostics["interaction_intensity_mean"])
    else:
        interaction_intensity_mean = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("direct_obj_delta_norm" in diagnostics):
        direct_obj_delta_norm = torch.mean(diagnostics["direct_obj_delta_norm"])
    else:
        direct_obj_delta_norm = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("exchange_balance_error" in diagnostics):
        exchange_balance_error = torch.mean(diagnostics["exchange_balance_error"])
    else:
        exchange_balance_error = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("pressure_correction_norm" in diagnostics):
        pressure_correction_norm = torch.mean(diagnostics["pressure_correction_norm"])
    else:
        pressure_correction_norm = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("pressure_divergence_pre" in diagnostics):
        pressure_divergence_pre = torch.mean(diagnostics["pressure_divergence_pre"])
    else:
        pressure_divergence_pre = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("pressure_divergence_post" in diagnostics):
        pressure_divergence_post = torch.mean(diagnostics["pressure_divergence_post"])
    else:
        pressure_divergence_post = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("pressure_stage1_correction_norm" in diagnostics):
        pressure_stage1_correction_norm = torch.mean(diagnostics["pressure_stage1_correction_norm"])
    else:
        pressure_stage1_correction_norm = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("pressure_stage1_divergence_pre" in diagnostics):
        pressure_stage1_divergence_pre = torch.mean(diagnostics["pressure_stage1_divergence_pre"])
    else:
        pressure_stage1_divergence_pre = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("pressure_stage1_divergence_post" in diagnostics):
        pressure_stage1_divergence_post = torch.mean(diagnostics["pressure_stage1_divergence_post"])
    else:
        pressure_stage1_divergence_post = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("pressure_stage2_correction_norm" in diagnostics):
        pressure_stage2_correction_norm = torch.mean(diagnostics["pressure_stage2_correction_norm"])
    else:
        pressure_stage2_correction_norm = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("pressure_stage2_divergence_pre" in diagnostics):
        pressure_stage2_divergence_pre = torch.mean(diagnostics["pressure_stage2_divergence_pre"])
    else:
        pressure_stage2_divergence_pre = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("pressure_stage2_divergence_post" in diagnostics):
        pressure_stage2_divergence_post = torch.mean(diagnostics["pressure_stage2_divergence_post"])
    else:
        pressure_stage2_divergence_post = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("pressure_stage2_divergence_delta" in diagnostics):
        pressure_stage2_divergence_delta = torch.mean(diagnostics["pressure_stage2_divergence_delta"])
    else:
        pressure_stage2_divergence_delta = pressure_stage2_divergence_post - pressure_stage2_divergence_pre
    if isinstance(diagnostics, dict) and ("output_pressure_correction_norm" in diagnostics):
        output_pressure_correction_norm = torch.mean(diagnostics["output_pressure_correction_norm"])
    else:
        output_pressure_correction_norm = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("output_pressure_divergence_pre" in diagnostics):
        output_pressure_divergence_pre = torch.mean(diagnostics["output_pressure_divergence_pre"])
    else:
        output_pressure_divergence_pre = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("output_pressure_divergence_post" in diagnostics):
        output_pressure_divergence_post = torch.mean(diagnostics["output_pressure_divergence_post"])
    else:
        output_pressure_divergence_post = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("output_pressure_divergence_delta" in diagnostics):
        output_pressure_divergence_delta = torch.mean(diagnostics["output_pressure_divergence_delta"])
    else:
        output_pressure_divergence_delta = output_pressure_divergence_post - output_pressure_divergence_pre
    if isinstance(diagnostics, dict) and ("output_boundary_mismatch_pre" in diagnostics):
        output_boundary_mismatch_pre = torch.mean(diagnostics["output_boundary_mismatch_pre"])
    else:
        output_boundary_mismatch_pre = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("output_boundary_mismatch_post" in diagnostics):
        output_boundary_mismatch_post = torch.mean(diagnostics["output_boundary_mismatch_post"])
    else:
        output_boundary_mismatch_post = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("dual_particle_step_norm" in diagnostics):
        dual_particle_step_norm = torch.mean(diagnostics["dual_particle_step_norm"])
    else:
        dual_particle_step_norm = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("dual_p2g_norm" in diagnostics):
        dual_p2g_norm = torch.mean(diagnostics["dual_p2g_norm"])
    else:
        dual_p2g_norm = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("obj_dyn_norm" in diagnostics):
        obj_dyn_norm = torch.mean(diagnostics["obj_dyn_norm"])
    else:
        obj_dyn_norm = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("obj_geo_norm" in diagnostics):
        obj_geo_norm = torch.mean(diagnostics["obj_geo_norm"])
    else:
        obj_geo_norm = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("obj_geo_gate_mean" in diagnostics):
        obj_geo_gate_mean = torch.mean(diagnostics["obj_geo_gate_mean"])
    else:
        obj_geo_gate_mean = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("obj_geo_to_int_raw_norm" in diagnostics):
        obj_geo_to_int_raw_norm = torch.mean(diagnostics["obj_geo_to_int_raw_norm"])
    else:
        obj_geo_to_int_raw_norm = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("obj_geo_to_int_gated_norm" in diagnostics):
        obj_geo_to_int_gated_norm = torch.mean(diagnostics["obj_geo_to_int_gated_norm"])
    else:
        obj_geo_to_int_gated_norm = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)
    if isinstance(diagnostics, dict) and ("obj_geo_effect_ratio" in diagnostics):
        obj_geo_effect_ratio = torch.mean(diagnostics["obj_geo_effect_ratio"])
    else:
        obj_geo_effect_ratio = torch.tensor(0.0, dtype=pred_velocity.dtype, device=pred_velocity.device)

    pressure_divergence_delta = pressure_divergence_post - pressure_divergence_pre
    obj_geo_dyn_ratio = obj_geo_norm / obj_dyn_norm.clamp_min(1e-8)
    l_exchange = exchange_balance_error

    total = (
        l1 * l_vel
        + l11 * l_vel_near_t1
        + l12 * l_mag_t1
        + l13 * l_resp_t1
        + l2 * l_grad
        + l3 * l_div
        + l4 * l_traj
        + l5 * l_int
        + l6 * l_roll
        + l7 * l_phase
        + l8 * l_dual
        + l9 * l_exchange
        + l10 * l_particle_module
    )

    stats = {
        "total_loss": float(total.detach().item()),
        "loss": float(total.detach().item()),  # backward-compatible alias
        "l_vel": float(l_vel.detach().item()),
        "l_vel_near_t1": float(l_vel_near_t1.detach().item()),
        "l_mag_t1": float(l_mag_t1.detach().item()),
        "l_resp_t1": float(l_resp_t1.detach().item()),
        "l_grad": float(l_grad.detach().item()),
        "l_div": float(l_div.detach().item()),
        "l_traj": float(l_traj.detach().item()),
        "l_dual": float(l_dual.detach().item()),
        "l_particle_module": float(l_particle_module.detach().item()),
        "l_exchange": float(l_exchange.detach().item()),
        "l_int": float(l_int.detach().item()),
        "l_roll": float(l_roll.detach().item()),
        "l_phase": float(l_phase.detach().item()),
        "rollout_endpoint_error": float(rollout_endpoint_error.detach().item()),
        "trajectory_endpoint_error": float(trajectory_endpoint_error.detach().item()),
        "near_vel_error": float(near_vel_error.detach().item()),
        "far_vel_error": float(far_vel_error.detach().item()),
        "near_voxel_count": float(near_voxel_count.detach().item()),
        "far_voxel_count": float(far_voxel_count.detach().item()),
        "near_voxel_frac": float(near_voxel_frac.detach().item()),
        "far_voxel_frac": float(far_voxel_frac.detach().item()),
        "mask_valid_voxel_count": float(mask_valid_voxel_count.detach().item()),
        "mask_valid_voxel_frac": float(mask_valid_voxel_frac.detach().item()),
        "near_sdf_threshold_used": float(near_sdf_threshold_used.detach().item()),
        "far_sdf_threshold_used": float(far_sdf_threshold_used.detach().item()),
        "mask_quantile_fallback_used": float(mask_quantile_fallback_used.detach().item()),
        "near_sdf_mean": float(near_sdf_mean.detach().item()),
        "far_sdf_mean": float(far_sdf_mean.detach().item()),
        "near_far_overlap_frac": float(near_far_overlap_frac.detach().item()),
        "pred_speed_mean_t1": float(pred_speed_mean_t1.detach().item()),
        "gt_speed_mean_t1": float(gt_speed_mean_t1.detach().item()),
        "pred_gt_speed_ratio_t1": float(pred_gt_speed_ratio_t1.detach().item()),
        "interaction_intensity_mean": float(interaction_intensity_mean.detach().item()),
        "exchange_balance_error": float(exchange_balance_error.detach().item()),
        "pressure_correction_norm": float(pressure_correction_norm.detach().item()),
        "pressure_divergence_pre": float(pressure_divergence_pre.detach().item()),
        "pressure_divergence_post": float(pressure_divergence_post.detach().item()),
        "pressure_divergence_delta": float(pressure_divergence_delta.detach().item()),
        "pressure_stage1_correction_norm": float(pressure_stage1_correction_norm.detach().item()),
        "pressure_stage1_divergence_pre": float(pressure_stage1_divergence_pre.detach().item()),
        "pressure_stage1_divergence_post": float(pressure_stage1_divergence_post.detach().item()),
        "pressure_stage2_correction_norm": float(pressure_stage2_correction_norm.detach().item()),
        "pressure_stage2_divergence_pre": float(pressure_stage2_divergence_pre.detach().item()),
        "pressure_stage2_divergence_post": float(pressure_stage2_divergence_post.detach().item()),
        "pressure_stage2_divergence_delta": float(pressure_stage2_divergence_delta.detach().item()),
        "output_pressure_correction_norm": float(output_pressure_correction_norm.detach().item()),
        "output_pressure_divergence_pre": float(output_pressure_divergence_pre.detach().item()),
        "output_pressure_divergence_post": float(output_pressure_divergence_post.detach().item()),
        "output_pressure_divergence_delta": float(output_pressure_divergence_delta.detach().item()),
        "output_boundary_mismatch_pre": float(output_boundary_mismatch_pre.detach().item()),
        "output_boundary_mismatch_post": float(output_boundary_mismatch_post.detach().item()),
        "dual_particle_step_norm": float(dual_particle_step_norm.detach().item()),
        "dual_p2g_norm": float(dual_p2g_norm.detach().item()),
        "obj_dyn_norm": float(obj_dyn_norm.detach().item()),
        "obj_geo_norm": float(obj_geo_norm.detach().item()),
        "obj_geo_dyn_ratio": float(obj_geo_dyn_ratio.detach().item()),
        "obj_geo_gate_mean": float(obj_geo_gate_mean.detach().item()),
        "obj_geo_to_int_raw_norm": float(obj_geo_to_int_raw_norm.detach().item()),
        "obj_geo_to_int_gated_norm": float(obj_geo_to_int_gated_norm.detach().item()),
        "obj_geo_effect_ratio": float(obj_geo_effect_ratio.detach().item()),
        "direct_obj_delta_norm": float(direct_obj_delta_norm.detach().item()),
    }
    return total, stats


def _prepare_batch_inputs(
    batch: Dict[str, Any], device: torch.device
) -> Tuple[
    torch.Tensor,
    Optional[Dict[str, Any]],
    Dict[str, Any],
    Dict[str, Any],
    torch.Tensor,
    torch.Tensor,
    Dict[str, Any],
]:
    fluid_init = batch["fluid_init"].to(device, non_blocking=True)
    solid_sequence = move_nested_to_device(batch.get("solid_sequence"), device)
    object_inputs = move_nested_to_device(batch["object_inputs"], device)
    interface_inputs = move_nested_to_device(batch["interface_inputs"], device)
    control_seq = batch["control_sequence"].to(device, non_blocking=True)
    particle_pos_start = batch["particle_pos_start"].to(device, non_blocking=True)
    batch_dev = {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
    }
    return fluid_init, solid_sequence, object_inputs, interface_inputs, control_seq, particle_pos_start, batch_dev


def _filter_interface_inputs_grid_only(
    interface_inputs: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if interface_inputs is None:
        return None
    grid_keys = {
        "fluid_solid_interface",
        "liquid_liquid_interface",
        "interface_sdf",
        "interface_normal",
        "boundary_velocity",
        "boundary_mask",
    }
    out: Dict[str, Any] = {}
    for k, v in interface_inputs.items():
        if k in grid_keys:
            out[k] = v
    return out


def _apply_condition_variant(
    *,
    solid_sequence: Optional[Dict[str, Any]],
    object_inputs: Optional[Dict[str, Any]],
    interface_inputs: Optional[Dict[str, Any]],
    control_seq: Optional[torch.Tensor],
    cond_variant: str,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[torch.Tensor],
]:
    variant = str(cond_variant).strip().lower()
    if variant == "hybrid":
        return solid_sequence, object_inputs, interface_inputs, control_seq
    if variant == "euler":
        # Euler-cond: use grid-aligned interface cues only, no object-centric conditioning.
        control_seq_zero = torch.zeros_like(control_seq) if isinstance(control_seq, torch.Tensor) else control_seq
        return solid_sequence, None, _filter_interface_inputs_grid_only(interface_inputs), control_seq_zero
    if variant == "object":
        # Object-cond: use object-centric conditioning only (no explicit interface branch input).
        return None, object_inputs, None, control_seq
    if variant == "bulk_only":
        # Bulk-only: no object/interface conditioning and no control signal.
        control_seq_zero = torch.zeros_like(control_seq) if isinstance(control_seq, torch.Tensor) else control_seq
        return None, None, None, control_seq_zero
    raise ValueError(f"cond_variant must be one of ['hybrid','euler','object','bulk_only'], got {cond_variant!r}")


def _forward_full_rollout(
    model: torch.nn.Module,
    fluid_init: torch.Tensor,
    solid_sequence: Optional[Dict[str, Any]],
    object_inputs: Optional[Dict[str, Any]],
    interface_inputs: Optional[Dict[str, Any]],
    control_seq: Optional[torch.Tensor],
    particle_pos_init: torch.Tensor,
    rollout_steps: int,
) -> Dict[str, torch.Tensor]:
    return model(
        fluid_init=fluid_init,
        solid_sequence=solid_sequence,
        object_inputs=object_inputs,
        interface_inputs=interface_inputs,
        control_sequence=control_seq,
        particle_pos_init=particle_pos_init,
        rollout_steps=rollout_steps,
        return_phase=True,
        return_pressure=False,
        return_latents=False,
        return_diagnostics=True,
    )


def _compute_one_step_metrics(
    out_rollout: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    out_one = {"velocity": out_rollout["velocity"][:, :1]}
    if "phase" in out_rollout:
        out_one["phase"] = out_rollout["phase"][:, :1]
    if "particle_pos" in out_rollout:
        out_one["particle_pos"] = out_rollout["particle_pos"][:, :2]
    if "diagnostics" in out_rollout:
        out_one["diagnostics"] = {k: v[:, :1] for k, v in out_rollout["diagnostics"].items()}
    batch_one = _slice_temporal_batch(batch, steps=1)
    _, one_stats = compute_losses(out=out_one, batch=batch_one, cfg=cfg)

    pred_v = out_one["velocity"]
    gt_v = batch_one["target_velocity"]
    one_stats["flow_len"] = float(_flow_length_error(pred_v, gt_v).detach().item())
    one_stats["flow_dir_cos"] = float(_flow_direction_cosine(pred_v, gt_v).detach().item())
    return one_stats


def _compute_rollout_metrics(
    out_rollout: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    _, roll_stats = compute_losses(out=out_rollout, batch=batch, cfg=cfg)
    pred_velocity = out_rollout["velocity"]
    gt_velocity = batch["target_velocity"]
    loss_cfg = cfg.get("loss", {})
    vel_norm = str(loss_cfg.get("velocity_norm", "l1"))
    traj_norm = str(loss_cfg.get("trajectory_norm", "l2"))
    traj_step_mode = str(loss_cfg.get("trajectory_step_mode", "displacement"))
    velocity_scale = float(cfg.get("dataset", {}).get("velocity_scale", 1.0))
    dt_value = float(batch.get("target_dt", torch.tensor(1.0, device=pred_velocity.device)).mean().item())

    if vel_norm == "l2":
        per_step_vel = torch.mean((pred_velocity - gt_velocity) ** 2, dim=(0, 2, 3, 4, 5))
    else:
        per_step_vel = torch.mean(torch.abs(pred_velocity - gt_velocity), dim=(0, 2, 3, 4, 5))
    per_step_div = torch.mean(torch.abs(divergence_3d(pred_velocity)), dim=(0, 2, 3, 4, 5))
    per_step_traj = trajectory_error_particle_per_step(
        pred_velocity_norm=pred_velocity,
        particle_pos_start=batch["particle_pos_start"],
        particle_pos_seq_gt=batch["particle_pos_seq"],
        particle_valid=batch["particle_valid"],
        velocity_scale=velocity_scale,
        step_mode=traj_step_mode,
        dt_value=dt_value,
        norm=traj_norm,
    )

    pred_phase = out_rollout.get("phase", None)
    t = pred_velocity.shape[1]
    per_step_int: List[torch.Tensor] = []
    int_cfg = loss_cfg.get("interface", {})
    for i in range(t):
        pred_phase_i = None if pred_phase is None else pred_phase[:, i : i + 1]
        l_int_i = interface_consistency_loss(
            pred_velocity=pred_velocity[:, i : i + 1],
            pred_phase=pred_phase_i,
            target_phase=batch["target_phase"][:, i : i + 1],
            boundary_velocity=batch["target_boundary_velocity"][:, i : i + 1],
            boundary_mask=batch["target_boundary_mask"][:, i : i + 1],
            interface_normal=batch["target_interface_normal"][:, i : i + 1],
            int_cfg=int_cfg,
        )
        per_step_int.append(l_int_i)
    per_step_int_curve = torch.stack(per_step_int) if per_step_int else torch.zeros((0,), device=pred_velocity.device)

    roll_stats["flow_len"] = float(_flow_length_error(pred_velocity, gt_velocity).detach().item())
    roll_stats["flow_dir_cos"] = float(_flow_direction_cosine(pred_velocity, gt_velocity).detach().item())

    curves = {
        "vel_curve": per_step_vel.detach().cpu().numpy(),
        "div_curve": per_step_div.detach().cpu().numpy(),
        "traj_curve": per_step_traj.detach().cpu().numpy(),
        "int_curve": per_step_int_curve.detach().cpu().numpy(),
    }
    return roll_stats, curves


def _running_mean_stats(totals: Dict[str, float], n: int) -> Dict[str, float]:
    denom = max(1, int(n))
    return {k: (float(v) / float(denom)) for k, v in totals.items()}


def _format_train_postfix(stats: Dict[str, float]) -> str:
    return (
        f"tot={stats['total_loss']:.4f} vel={stats['l_vel']:.4f} roll={stats['l_roll']:.4f} "
        f"traj={stats['l_traj']:.4f} pmod={stats['l_particle_module']:.4f} "
        f"dual={stats['l_dual']:.4f} exch={stats['l_exchange']:.4f} "
        f"div={stats['l_div']:.4f} grad={stats['l_grad']:.4f} "
        f"int={stats['l_int']:.4f} iint={stats['interaction_intensity_mean']:.4f} "
        f"pcor={stats['pressure_correction_norm']:.4f} pdiv={stats['pressure_divergence_post']:.4f} "
        f"pdd={stats['pressure_divergence_delta']:+.4f} "
        f"opcor={stats['output_pressure_correction_norm']:.4f} opdiv={stats['output_pressure_divergence_post']:.4f} "
        f"opdd={stats['output_pressure_divergence_delta']:+.4f} "
        f"g2p={stats['dual_particle_step_norm']:.4f} p2g={stats['dual_p2g_norm']:.4f} "
        f"og={stats['obj_geo_gate_mean']:.4f} gdr={stats['obj_geo_dyn_ratio']:.4f} ger={stats['obj_geo_effect_ratio']:.4f} "
        f"dir={stats['direct_obj_delta_norm']:.4f} "
        f"phase={stats['l_phase']:.4f} "
        f"near={stats['near_vel_error']:.4f} far={stats['far_vel_error']:.4f} "
        f"nct={stats['near_voxel_count']:.0f} fct={stats['far_voxel_count']:.0f} "
        f"vfct={stats['mask_valid_voxel_count']:.0f} mfb={stats['mask_quantile_fallback_used']:.0f} ovl={stats['near_far_overlap_frac']:.2f} "
        f"end_v={stats['rollout_endpoint_error']:.4f} end_traj={stats['trajectory_endpoint_error']:.4f}"
    )


def resolve_rollout_steps_for_epoch(
    run_cfg: Dict[str, Any], epoch: int, default_rollout_steps: int
) -> int:
    curriculum = run_cfg.get("rollout_curriculum", None)
    if not isinstance(curriculum, dict) or not bool(curriculum.get("enabled", False)):
        return int(max(1, default_rollout_steps))
    schedule = curriculum.get("schedule", [])
    if not isinstance(schedule, list) or len(schedule) == 0:
        return int(max(1, default_rollout_steps))

    parsed: List[Tuple[int, int]] = []
    for row in schedule:
        if not isinstance(row, dict):
            continue
        until_epoch = int(row.get("until_epoch", 0))
        rollout_steps = int(row.get("rollout_steps", 0))
        if until_epoch > 0 and rollout_steps > 0:
            parsed.append((until_epoch, rollout_steps))
    if not parsed:
        return int(max(1, default_rollout_steps))

    parsed.sort(key=lambda x: x[0])
    selected = parsed[-1][1]
    for until_epoch, rollout_steps in parsed:
        if int(epoch) <= until_epoch:
            selected = rollout_steps
            break
    return int(max(1, min(int(default_rollout_steps), int(selected))))


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    cfg: Dict[str, Any],
    device: torch.device,
    max_steps: int = 0,
    active_rollout_steps: int = 0,
    log_interval: int = 20,
    show_progress: bool = True,
    progress_prefix: str = "train",
    progress_width: int = 28,
    cond_variant: str = "hybrid",
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    totals = {k: 0.0 for k in LOSS_TRACK_KEYS}
    n = 0
    total_steps = len(loader) if max_steps <= 0 else min(len(loader), max_steps)
    progress = TerminalProgressBar(
        total=total_steps,
        prefix=str(progress_prefix),
        width=int(progress_width),
        enabled=bool(show_progress),
    )

    for step, batch in enumerate(loader):
        fluid_init, solid_sequence, object_inputs, interface_inputs, control_seq, particle_pos_start, batch_dev = _prepare_batch_inputs(
            batch, device
        )

        max_rollout_steps = int(batch_dev["target_velocity"].shape[1])
        rollout_steps = (
            min(max_rollout_steps, int(active_rollout_steps))
            if int(active_rollout_steps) > 0
            else max_rollout_steps
        )
        batch_loss = _slice_temporal_batch(batch_dev, steps=rollout_steps)
        control_seq_cut = control_seq[:, :rollout_steps] if control_seq.ndim == 3 else control_seq
        solid_sequence_eff, object_inputs_eff, interface_inputs_eff, control_seq_eff = _apply_condition_variant(
            solid_sequence=solid_sequence,
            object_inputs=object_inputs,
            interface_inputs=interface_inputs,
            control_seq=control_seq_cut,
            cond_variant=cond_variant,
        )
        out = _forward_full_rollout(
            model=model,
            fluid_init=fluid_init,
            solid_sequence=solid_sequence_eff,
            object_inputs=object_inputs_eff,
            interface_inputs=interface_inputs_eff,
            control_seq=control_seq_eff,
            particle_pos_init=particle_pos_start,
            rollout_steps=rollout_steps,
        )

        loss, stats = compute_losses(out=out, batch=batch_loss, cfg=cfg)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        for k in LOSS_TRACK_KEYS:
            totals[k] += float(stats[k])
        n += 1

        running = _running_mean_stats(totals, n)
        progress.update(step + 1, postfix=_format_train_postfix(running))
        if not show_progress and log_interval > 0 and ((step + 1) % log_interval == 0):
            print(f"  step {step + 1:5d}/{total_steps:5d} {_format_train_postfix(running)}", flush=True)

        if max_steps > 0 and (step + 1) >= max_steps:
            break

    progress.close()
    out_stats = _running_mean_stats(totals, n)
    out_stats["loss"] = out_stats["total_loss"]  # backward-compatible alias
    return out_stats


def run_validation(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
    max_steps: int = 0,
    active_rollout_steps: int = 0,
    show_progress: bool = True,
    progress_prefix: str = "val",
    progress_width: int = 28,
    pressure_correction_scale_override: float | None = None,
    object_ablation_mode: str = "none",
    cond_variant: str = "hybrid",
) -> Dict[str, Any]:
    model.eval()
    total_steps = len(loader) if max_steps <= 0 else min(len(loader), max_steps)
    progress = TerminalProgressBar(
        total=total_steps,
        prefix=str(progress_prefix),
        width=int(progress_width),
        enabled=bool(show_progress),
    )

    obj_mode = str(object_ablation_mode).strip().lower()
    if obj_mode not in ("none", "zero", "shuffle"):
        raise ValueError(f"object_ablation_mode must be one of ['none','zero','shuffle'], got {object_ablation_mode!r}")
    prev_pressure = None
    if pressure_correction_scale_override is not None:
        prev_pressure = _set_pressure_correction_scale(model, float(pressure_correction_scale_override))

    one_totals: Dict[str, float] = {}
    roll_totals: Dict[str, float] = {}
    curve_sums: Dict[str, np.ndarray] = {}
    n = 0

    try:
        with torch.no_grad():
            for step, batch in enumerate(loader):
                fluid_init, solid_sequence, object_inputs, interface_inputs, control_seq, particle_pos_start, batch_dev = _prepare_batch_inputs(
                    batch, device
                )
                max_rollout_steps = int(batch_dev["target_velocity"].shape[1])
                rollout_steps = (
                    min(max_rollout_steps, int(active_rollout_steps))
                    if int(active_rollout_steps) > 0
                    else max_rollout_steps
                )
                batch_eval = _slice_temporal_batch(batch_dev, steps=rollout_steps)
                control_seq_cut = control_seq[:, :rollout_steps] if control_seq.ndim == 3 else control_seq
                solid_sequence_eff, object_inputs_var, interface_inputs_eff, control_seq_eff = _apply_condition_variant(
                    solid_sequence=solid_sequence,
                    object_inputs=object_inputs,
                    interface_inputs=interface_inputs,
                    control_seq=control_seq_cut,
                    cond_variant=cond_variant,
                )
                object_inputs_eff = (
                    None
                    if object_inputs_var is None
                    else _ablate_object_geometry_inputs(object_inputs_var, mode=obj_mode)
                )
                out = _forward_full_rollout(
                    model=model,
                    fluid_init=fluid_init,
                    solid_sequence=solid_sequence_eff,
                    object_inputs=object_inputs_eff,
                    interface_inputs=interface_inputs_eff,
                    control_seq=control_seq_eff,
                    particle_pos_init=particle_pos_start,
                    rollout_steps=rollout_steps,
                )

                one_stats = _compute_one_step_metrics(out_rollout=out, batch=batch_eval, cfg=cfg)
                roll_stats, curves = _compute_rollout_metrics(out_rollout=out, batch=batch_eval, cfg=cfg)

                for k, v in one_stats.items():
                    one_totals[k] = one_totals.get(k, 0.0) + float(v)
                for k, v in roll_stats.items():
                    roll_totals[k] = roll_totals.get(k, 0.0) + float(v)
                for k, arr in curves.items():
                    if k not in curve_sums:
                        curve_sums[k] = np.array(arr, dtype=np.float64)
                    else:
                        curve_sums[k] += np.array(arr, dtype=np.float64)
                n += 1

                one_mean = {k: (v / max(1, n)) for k, v in one_totals.items()}
                roll_mean = {k: (v / max(1, n)) for k, v in roll_totals.items()}
                progress.update(
                    step + 1,
                    postfix=(
                        f"1s_vel={one_mean.get('l_vel', 0.0):.4f} "
                        f"1s_traj={one_mean.get('l_traj', 0.0):.4f} "
                        f"1s_pmod={one_mean.get('l_particle_module', 0.0):.4f} "
                        f"1s_dual={one_mean.get('l_dual', 0.0):.4f} "
                        f"1s_exch={one_mean.get('l_exchange', 0.0):.4f} "
                        f"roll_vel={roll_mean.get('l_roll', 0.0):.4f} "
                        f"roll_traj={roll_mean.get('l_traj', 0.0):.4f} "
                        f"roll_pmod={roll_mean.get('l_particle_module', 0.0):.4f} "
                        f"roll_dual={roll_mean.get('l_dual', 0.0):.4f} "
                        f"roll_exch={roll_mean.get('l_exchange', 0.0):.4f} "
                        f"roll_div={roll_mean.get('l_div', 0.0):.4f} "
                        f"iint={roll_mean.get('interaction_intensity_mean', 0.0):.4f} "
                        f"pcor={roll_mean.get('pressure_correction_norm', 0.0):.4f} "
                        f"pdiv={roll_mean.get('pressure_divergence_post', 0.0):.4f} "
                        f"opcor={roll_mean.get('output_pressure_correction_norm', 0.0):.4f} "
                        f"opdiv={roll_mean.get('output_pressure_divergence_post', 0.0):.4f} "
                        f"g2p={roll_mean.get('dual_particle_step_norm', 0.0):.4f} "
                        f"p2g={roll_mean.get('dual_p2g_norm', 0.0):.4f} "
                        f"gdr={roll_mean.get('obj_geo_dyn_ratio', 0.0):.4f} "
                        f"ger={roll_mean.get('obj_geo_effect_ratio', 0.0):.4f} "
                        f"dir={roll_mean.get('direct_obj_delta_norm', 0.0):.4f} "
                        f"near={roll_mean.get('near_vel_error', 0.0):.4f} "
                        f"far={roll_mean.get('far_vel_error', 0.0):.4f} "
                        f"nct={roll_mean.get('near_voxel_count', 0.0):.0f} "
                        f"fct={roll_mean.get('far_voxel_count', 0.0):.0f} "
                        f"vfct={roll_mean.get('mask_valid_voxel_count', 0.0):.0f} "
                        f"mfb={roll_mean.get('mask_quantile_fallback_used', 0.0):.0f} ovl={roll_mean.get('near_far_overlap_frac', 0.0):.2f} "
                        f"roll_end={roll_mean.get('rollout_endpoint_error', 0.0):.4f} "
                        f"traj_end={roll_mean.get('trajectory_endpoint_error', 0.0):.4f}"
                    ),
                )

                if max_steps > 0 and (step + 1) >= max_steps:
                    break
    finally:
        _restore_pressure_correction_scale(model, prev_pressure)

    progress.close()
    denom = max(1, n)
    one_avg = {k: (v / denom) for k, v in one_totals.items()}
    roll_avg = {k: (v / denom) for k, v in roll_totals.items()}
    step_curves = {k: (arr / denom).astype(np.float32).tolist() for k, arr in curve_sums.items()}
    one_avg["loss"] = one_avg.get("total_loss", one_avg.get("loss", 0.0))
    roll_avg["loss"] = roll_avg.get("total_loss", roll_avg.get("loss", 0.0))
    return {
        "one_step": one_avg,
        "rollout": roll_avg,
        "step_curves": step_curves,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    run_cfg = cfg["train"]
    ds_cfg = cfg["dataset"]
    model_cfg = dict(cfg["model"])

    set_seed(int(run_cfg.get("seed", 20260319)))
    device_name = (
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() and bool(run_cfg.get("use_cuda", True)) else "cpu")
    )
    device = torch.device(device_name)
    print(f"[device] {device}")

    train_loader = build_loader(cfg, split=str(ds_cfg.get("train_split", "train")), shuffle=True)
    val_loader = build_loader(cfg, split=str(ds_cfg.get("val_split", "val")), shuffle=False)
    print(f"[data] train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    model = build_model(model_cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(run_cfg.get("lr", 1e-4)),
        weight_decay=float(run_cfg.get("weight_decay", 1e-6)),
    )
    scheduler = None
    step_size = int(run_cfg.get("scheduler_step_size", 0))
    if step_size > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=float(run_cfg.get("scheduler_gamma", 0.95)),
        )

    epochs = int(run_cfg.get("epochs", 10))
    max_train_steps = int(run_cfg.get("max_train_steps", 0))
    max_val_steps = int(run_cfg.get("max_val_steps", 0))
    log_interval = int(run_cfg.get("log_interval", 20))
    show_progress = bool(run_cfg.get("show_progress_bar", True))
    progress_width = int(run_cfg.get("progress_bar_width", 28))
    default_rollout_steps = int(ds_cfg.get("rollout_steps", 8))
    checkpoint_every_steps = int(run_cfg.get("checkpoint_every_steps", 0))
    curriculum_apply_to_val = bool(
        run_cfg.get("rollout_curriculum", {}).get("apply_to_val", False)
        if isinstance(run_cfg.get("rollout_curriculum", None), dict)
        else False
    )
    run_val_ablation_pressure_off = bool(run_cfg.get("val_ablation_pressure_off", False))
    run_val_ablation_geo_zero = bool(run_cfg.get("val_ablation_geo_zero", False))
    run_val_ablation_geo_shuffle = bool(run_cfg.get("val_ablation_geo_shuffle", False))
    cond_variant = str(run_cfg.get("cond_variant", "hybrid")).strip().lower()
    if cond_variant not in ("hybrid", "euler", "object", "bulk_only"):
        raise ValueError(
            f"train.cond_variant must be one of ['hybrid','euler','object','bulk_only'], got {cond_variant!r}"
        )
    train_steps_per_epoch = len(train_loader) if max_train_steps <= 0 else min(len(train_loader), max_train_steps)
    if train_steps_per_epoch <= 0:
        raise RuntimeError("train_steps_per_epoch must be > 0")

    resume_ckpt_raw = str(run_cfg.get("resume_checkpoint", "")).strip()
    resume_ckpt_path: Path | None = None
    if resume_ckpt_raw:
        resume_ckpt_path = Path(resume_ckpt_raw).expanduser()
        if not resume_ckpt_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_ckpt_path}")

    base_out_dir = Path(run_cfg.get("output_dir", PROJECT_ROOT / "outputs" / "interaction_aware"))
    if resume_ckpt_path is not None:
        # In resume mode, default to writing alongside the checkpoint unless output_dir is explicitly given.
        if args.output_dir is None:
            out_dir = resume_ckpt_path.parent
            print(f"[output] resume mode, using checkpoint dir: {out_dir}")
        else:
            out_dir = base_out_dir
            print(f"[output] resume mode, using explicit dir: {out_dir}")
            ckpt_parent = resume_ckpt_path.parent.resolve()
            out_parent = out_dir.resolve()
            if out_parent != ckpt_parent:
                print(
                    f"[warn] resume checkpoint dir ({ckpt_parent}) != output dir ({out_parent}); "
                    "new metrics/checkpoints will be written to output dir."
                )
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = resolve_run_output_dir(base_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_dir != base_out_dir:
            print(f"[output] base exists, using new dir: {out_dir}")
        else:
            print(f"[output] using dir: {out_dir}")
    ckpt_last = out_dir / "checkpoint_last.pt"
    ckpt_best = out_dir / "checkpoint_best.pt"
    metrics_path = out_dir / "metrics.jsonl"
    resolved_cfg_path = out_dir / "resolved_config.json"
    with resolved_cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[output] config saved: {resolved_cfg_path}")

    best_val = float("inf")
    prev_global_step = 0
    start_epoch = 1
    if resume_ckpt_path is not None:
        ckpt = torch.load(str(resume_ckpt_path), map_location=device)
        model_state = ckpt.get("model", None)
        if model_state is None:
            raise RuntimeError(f"invalid checkpoint (missing model state): {resume_ckpt_path}")
        model.load_state_dict(model_state, strict=True)

        opt_state = ckpt.get("optimizer", None)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)

        sched_state = ckpt.get("scheduler", None)
        if scheduler is not None and sched_state is not None:
            scheduler.load_state_dict(sched_state)

        resumed_epoch = int(ckpt.get("epoch", 0))
        resumed_step = int(ckpt.get("global_step", resumed_epoch * train_steps_per_epoch))
        start_epoch = resumed_epoch + 1
        prev_global_step = resumed_step

        ckpt_val = ckpt.get("val", {})
        if isinstance(ckpt_val, dict):
            ckpt_roll = ckpt_val.get("rollout", {})
            if isinstance(ckpt_roll, dict):
                best_guess = float(ckpt_roll.get("l_roll", ckpt_roll.get("total_loss", float("inf"))))
                if math.isfinite(best_guess):
                    best_val = best_guess

        print(
            f"[resume] loaded checkpoint: {resume_ckpt_path} "
            f"(epoch={resumed_epoch}, global_step={resumed_step}, best_val={best_val:.6f})"
        )
        if start_epoch > epochs:
            print(
                f"[resume] start_epoch ({start_epoch}) is beyond configured epochs ({epochs}); "
                "nothing to run."
            )
            print(f"\n[done] outputs at: {out_dir}")
            return

    for epoch in range(start_epoch, epochs + 1):
        e0 = time.time()
        global_step = epoch * train_steps_per_epoch
        train_rollout_steps = resolve_rollout_steps_for_epoch(
            run_cfg=run_cfg,
            epoch=epoch,
            default_rollout_steps=default_rollout_steps,
        )
        val_rollout_steps = train_rollout_steps if curriculum_apply_to_val else 0
        val_rollout_label = (
            f"{val_rollout_steps}"
            if val_rollout_steps > 0
            else f"{default_rollout_steps}(full)"
        )
        print(f"\n[epoch {epoch}/{epochs}] rollout(train={train_rollout_steps}, val={val_rollout_label})")
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            max_steps=max_train_steps,
            active_rollout_steps=train_rollout_steps,
            log_interval=log_interval,
            show_progress=show_progress,
            progress_prefix=f"train e{epoch:03d}",
            progress_width=progress_width,
            cond_variant=cond_variant,
        )
        val_stats = run_validation(
            model=model,
            loader=val_loader,
            cfg=cfg,
            device=device,
            max_steps=max_val_steps,
            active_rollout_steps=val_rollout_steps,
            show_progress=show_progress,
            progress_prefix=f"val   e{epoch:03d}",
            progress_width=progress_width,
            cond_variant=cond_variant,
        )

        val_ablations: Dict[str, Any] = {}
        if run_val_ablation_pressure_off:
            val_ablations["pressure_off"] = run_validation(
                model=model,
                loader=val_loader,
                cfg=cfg,
                device=device,
                max_steps=max_val_steps,
                active_rollout_steps=val_rollout_steps,
                show_progress=show_progress,
                progress_prefix=f"valP0 e{epoch:03d}",
                progress_width=progress_width,
                pressure_correction_scale_override=0.0,
                object_ablation_mode="none",
                cond_variant=cond_variant,
            )
        if run_val_ablation_geo_zero:
            val_ablations["geo_zero"] = run_validation(
                model=model,
                loader=val_loader,
                cfg=cfg,
                device=device,
                max_steps=max_val_steps,
                active_rollout_steps=val_rollout_steps,
                show_progress=show_progress,
                progress_prefix=f"valG0 e{epoch:03d}",
                progress_width=progress_width,
                object_ablation_mode="zero",
                cond_variant=cond_variant,
            )
        if run_val_ablation_geo_shuffle:
            val_ablations["geo_shuffle"] = run_validation(
                model=model,
                loader=val_loader,
                cfg=cfg,
                device=device,
                max_steps=max_val_steps,
                active_rollout_steps=val_rollout_steps,
                show_progress=show_progress,
                progress_prefix=f"valGS e{epoch:03d}",
                progress_width=progress_width,
                object_ablation_mode="shuffle",
                cond_variant=cond_variant,
            )

        if scheduler is not None:
            scheduler.step()

        lr = float(optimizer.param_groups[0]["lr"])
        val_one = val_stats["one_step"]
        val_roll = val_stats["rollout"]
        row = {
            "epoch": epoch,
            "global_step": int(global_step),
            "lr": lr,
            "cond_variant": cond_variant,
            "train_rollout_steps": train_rollout_steps,
            "val_rollout_steps": (val_rollout_steps if val_rollout_steps > 0 else default_rollout_steps),
            "train": train_stats,
            "val": {
                "one_step": val_one,
                "rollout": val_roll,
                "step_curves": val_stats["step_curves"],
            },
            "epoch_sec": float(time.time() - e0),
        }
        val_ablation_deltas: Dict[str, Dict[str, float]] = {}
        if val_ablations:
            base_roll = float(val_roll.get("l_roll", val_roll.get("total_loss", 0.0)))
            base_traj_end = float(val_roll.get("trajectory_endpoint_error", 0.0))
            for name, stats_ab in val_ablations.items():
                ab_roll = stats_ab.get("rollout", {})
                ab_roll_l = float(ab_roll.get("l_roll", ab_roll.get("total_loss", 0.0)))
                ab_traj_end = float(ab_roll.get("trajectory_endpoint_error", 0.0))
                val_ablation_deltas[name] = {
                    "roll_delta": float(ab_roll_l - base_roll),
                    "traj_end_delta": float(ab_traj_end - base_traj_end),
                    "base_roll": float(base_roll),
                    "ablation_roll": float(ab_roll_l),
                    "base_traj_end": float(base_traj_end),
                    "ablation_traj_end": float(ab_traj_end),
                }
            row["val_ablations"] = val_ablations
            row["val_ablation_deltas"] = val_ablation_deltas
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(
            f"  [train] total={train_stats['total_loss']:.6f} vel={train_stats['l_vel']:.6f} "
            f"roll={train_stats['l_roll']:.6f} traj={train_stats['l_traj']:.6f} "
            f"pmod={train_stats['l_particle_module']:.6f} "
            f"dual={train_stats['l_dual']:.6f} exch={train_stats['l_exchange']:.6f} "
            f"div={train_stats['l_div']:.6f} grad={train_stats['l_grad']:.6f} "
            f"int={train_stats['l_int']:.6f} iint={train_stats['interaction_intensity_mean']:.6f} "
            f"pcor={train_stats['pressure_correction_norm']:.6f} "
            f"pdiv={train_stats['pressure_divergence_post']:.6f} "
            f"pdd={train_stats['pressure_divergence_delta']:+.6f} "
            f"opcor={train_stats.get('output_pressure_correction_norm', 0.0):.6f} "
            f"opdiv={train_stats.get('output_pressure_divergence_post', 0.0):.6f} "
            f"opdd={train_stats.get('output_pressure_divergence_delta', 0.0):+.6f} "
            f"g2p={train_stats['dual_particle_step_norm']:.6f} "
            f"p2g={train_stats['dual_p2g_norm']:.6f} "
            f"gdr={train_stats['obj_geo_dyn_ratio']:.6f} ger={train_stats['obj_geo_effect_ratio']:.6f} "
            f"dir={train_stats['direct_obj_delta_norm']:.6f} "
            f"phase={train_stats['l_phase']:.6f} "
            f"near={train_stats['near_vel_error']:.6f} far={train_stats['far_vel_error']:.6f} "
            f"nct={train_stats['near_voxel_count']:.0f} fct={train_stats['far_voxel_count']:.0f} "
            f"vfct={train_stats['mask_valid_voxel_count']:.0f} mfb={train_stats['mask_quantile_fallback_used']:.0f} ovl={train_stats['near_far_overlap_frac']:.2f} "
            f"roll_end={train_stats['rollout_endpoint_error']:.6f} "
            f"traj_end={train_stats['trajectory_endpoint_error']:.6f}"
        )
        print(
            f"  [val-1step] vel={val_one.get('l_vel', 0.0):.6f} phase={val_one.get('l_phase', 0.0):.6f} "
            f"div={val_one.get('l_div', 0.0):.6f} traj={val_one.get('l_traj', 0.0):.6f} "
            f"pmod={val_one.get('l_particle_module', 0.0):.6f} "
            f"dual={val_one.get('l_dual', 0.0):.6f} iint={val_one.get('interaction_intensity_mean', 0.0):.6f} "
            f"exch={val_one.get('l_exchange', 0.0):.6f} pcor={val_one.get('pressure_correction_norm', 0.0):.6f} "
            f"pdiv={val_one.get('pressure_divergence_post', 0.0):.6f} "
            f"pdd={val_one.get('pressure_divergence_delta', 0.0):+.6f} "
            f"opcor={val_one.get('output_pressure_correction_norm', 0.0):.6f} "
            f"opdiv={val_one.get('output_pressure_divergence_post', 0.0):.6f} "
            f"opdd={val_one.get('output_pressure_divergence_delta', 0.0):+.6f} "
            f"g2p={val_one.get('dual_particle_step_norm', 0.0):.6f} "
            f"p2g={val_one.get('dual_p2g_norm', 0.0):.6f} "
            f"dir={val_one.get('direct_obj_delta_norm', 0.0):.6f} "
            f"flow_len={val_one.get('flow_len', 0.0):.6f} flow_dir_cos={val_one.get('flow_dir_cos', 0.0):.6f}"
        )
        print(
            f"  [val-roll] total={val_roll.get('total_loss', 0.0):.6f} vel={val_roll.get('l_roll', 0.0):.6f} "
            f"traj={val_roll.get('l_traj', 0.0):.6f} pmod={val_roll.get('l_particle_module', 0.0):.6f} "
            f"dual={val_roll.get('l_dual', 0.0):.6f} "
            f"exch={val_roll.get('l_exchange', 0.0):.6f} "
            f"div={val_roll.get('l_div', 0.0):.6f} int={val_roll.get('l_int', 0.0):.6f} "
            f"iint={val_roll.get('interaction_intensity_mean', 0.0):.6f} "
            f"pcor={val_roll.get('pressure_correction_norm', 0.0):.6f} "
            f"pdiv={val_roll.get('pressure_divergence_post', 0.0):.6f} "
            f"pdd={val_roll.get('pressure_divergence_delta', 0.0):+.6f} "
            f"opcor={val_roll.get('output_pressure_correction_norm', 0.0):.6f} "
            f"opdiv={val_roll.get('output_pressure_divergence_post', 0.0):.6f} "
            f"opdd={val_roll.get('output_pressure_divergence_delta', 0.0):+.6f} "
            f"g2p={val_roll.get('dual_particle_step_norm', 0.0):.6f} "
            f"p2g={val_roll.get('dual_p2g_norm', 0.0):.6f} "
            f"gdr={val_roll.get('obj_geo_dyn_ratio', 0.0):.6f} ger={val_roll.get('obj_geo_effect_ratio', 0.0):.6f} "
            f"dir={val_roll.get('direct_obj_delta_norm', 0.0):.6f} "
            f"near={val_roll.get('near_vel_error', 0.0):.6f} "
            f"far={val_roll.get('far_vel_error', 0.0):.6f} "
            f"nct={val_roll.get('near_voxel_count', 0.0):.0f} fct={val_roll.get('far_voxel_count', 0.0):.0f} "
            f"vfct={val_roll.get('mask_valid_voxel_count', 0.0):.0f} mfb={val_roll.get('mask_quantile_fallback_used', 0.0):.0f} ovl={val_roll.get('near_far_overlap_frac', 0.0):.2f} "
            f"flow_len={val_roll.get('flow_len', 0.0):.6f} "
            f"flow_dir_cos={val_roll.get('flow_dir_cos', 0.0):.6f} "
            f"roll_end={val_roll.get('rollout_endpoint_error', 0.0):.6f} "
            f"traj_end={val_roll.get('trajectory_endpoint_error', 0.0):.6f} lr={lr:.6e}"
        )
        if val_ablations:
            base_roll = float(val_roll.get("l_roll", val_roll.get("total_loss", 0.0)))
            base_traj_end = float(val_roll.get("trajectory_endpoint_error", 0.0))
            for name, stats_ab in val_ablations.items():
                ab_roll = stats_ab.get("rollout", {})
                ab_roll_l = float(ab_roll.get("l_roll", ab_roll.get("total_loss", 0.0)))
                ab_traj_end = float(ab_roll.get("trajectory_endpoint_error", 0.0))
                print(
                    f"  [ablation {name}] roll={ab_roll_l:.6f} (delta={ab_roll_l - base_roll:+.6f}) "
                    f"traj_end={ab_traj_end:.6f} (delta={ab_traj_end - base_traj_end:+.6f})"
                )

        vel_curve = val_stats["step_curves"].get("vel_curve", [])
        if vel_curve:
            print(
                "  [val-curve] vel="
                + ",".join(f"{float(x):.4f}" for x in vel_curve)
            )

        payload = {
            "epoch": epoch,
            "global_step": int(global_step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "cfg": cfg,
            "train": train_stats,
            "val": row["val"],
        }
        should_save_last = True
        if checkpoint_every_steps > 0:
            crossed = (global_step // checkpoint_every_steps) > (prev_global_step // checkpoint_every_steps)
            should_save_last = crossed or (epoch == epochs)
        if should_save_last:
            torch.save(payload, ckpt_last)
            print(f"  [checkpoint] last saved at global_step={global_step}")
        model_select_metric = float(val_roll.get("l_roll", val_roll.get("total_loss", 1e18)))
        if model_select_metric < best_val:
            best_val = model_select_metric
            torch.save(payload, ckpt_best)
            print(f"  [checkpoint] best updated (val_roll): {best_val:.6f}")
        prev_global_step = global_step

    print(f"\n[done] outputs at: {out_dir}")


if __name__ == "__main__":
    main()
