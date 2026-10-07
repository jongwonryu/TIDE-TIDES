from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runtime import get_trainer, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a free-running TIDE/TIDES checkpoint.")
    parser.add_argument("--model", choices=("tide", "tides"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0, help="0 evaluates the complete split.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.max_batches < 0:
        parser.error("--steps must be positive and --max-batches nonnegative.")
    device = torch.device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, args.model, args.config, device)
    cfg["dataset"]["root"] = str(args.dataset_root.resolve())
    cfg["dataset"]["rollout_steps"] = args.steps
    cfg["dataset"][f"max_{args.split}_episodes"] = 0
    cfg["train"]["batch_size"] = 1
    trainer = get_trainer(args.model)
    trainer.set_seed(int(cfg["train"].get("seed", 0)))
    loader = trainer.build_loader(cfg, split=args.split, shuffle=False)
    if not len(loader.dataset):
        parser.error(f"No rollout windows found for split {args.split!r}.")
    stats = trainer.run_validation(
        model=model, loader=loader, cfg=cfg, device=device,
        max_steps=args.max_batches, active_rollout_steps=0,
        show_progress=True, progress_prefix="evaluate",
        cond_variant=str(cfg["train"].get("cond_variant", "hybrid")),
    )
    result = {
        "model": args.model, "split": args.split, "rollout_steps": args.steps,
        "available_windows": len(loader.dataset),
        "evaluated_batches": min(len(loader), args.max_batches) if args.max_batches else len(loader),
        "checkpoint": str(args.checkpoint), **stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metrics: {args.output}")


if __name__ == "__main__":
    main()
