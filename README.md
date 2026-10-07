# TIDE / TIDES

Framework code for **Interactive 4D Volumetric Liquid Forecasting under
Moving-Solid Interaction**, accepted at **NeurIPS 2026 (Poster)**.

**Jongwon Ryu, Sunjae Yoon, Junyeong Kim**

Department of Artificial Intelligence, Chung-Ang University

Given an observed 3D liquid velocity field and a known future solid-state
sequence, TIDE/TIDES predict a dense liquid velocity rollout. TIDE combines
bulk evolution with an interface-localized residual. TIDES adds recurrent
interface mediation, particle-grid transport feedback, and latent/output
correction. The model predicts liquid motion conditional on the supplied solid
motion; it does not advance solid dynamics in a closed fluid-solid loop.

## Release scope

This initial release includes the TIDE and TIDES implementations, training,
free-running inference, evaluation, configuration files, environment setup, and
a compact example extracted from an existing generated dataset.

The complete dataset and pretrained checkpoints will be distributed separately
on Hugging Face. Their download links will be added here when available.
External baselines, full dataset-generation tools, and the complete ablation
and rebuttal experiment suite are outside this initial framework release.
The compact example demonstrates execution; it is not the full benchmark and
does not reproduce the paper's quantitative results.

## Installation

Python 3.10 or newer is required. Run commands from the repository root.

```bash
git clone https://github.com/jongwonryu/TIDE-TIDES.git
cd TIDE-TIDES
conda env create -f environment.yml
conda activate tide-tides
```

Alternatively, use an existing environment:

```bash
python -m pip install -r requirements.txt
```

For GPU execution, install a CUDA-enabled PyTorch build compatible with your
driver before installing the other dependencies. The full-size configurations
require considerably more memory than the compact example configurations.

## Quick start

Test both models on the included example without downloading a dataset or
checkpoint:

```bash
bash bash/smoke_test.sh --device cpu
```

Run one training batch and one validation batch on the compact example:

```bash
bash bash/train_example.sh tide --device cpu
bash bash/train_example.sh tides --device cpu
```

These commands use reduced model widths and an example validation split equal
to the example training split. They check the execution pipeline, not
generalization. The checkpoints and resolved configurations are written under
`outputs/tide_example/` and `outputs/tides_example/`. If an output directory
already exists, the trainer creates a new numbered directory.

## Training

Place a compatible full dataset at `datasets/dualsph_liquid_v2_dense/`, or
change `dataset.root` in the configuration. Each dataset root contains a
`manifest.jsonl` with relative sample paths and split assignments.

```bash
bash bash/train.sh tide --device cuda
bash bash/train.sh tides --device cuda
```

The trainers support command-line overrides and checkpoint resume:

```bash
python train/train_bulk_interface.py --help
python train/train_interaction_aware.py --help
bash bash/train.sh tides --device cuda \
  --resume-checkpoint outputs/tides/checkpoint_last.pt
```

Configurations contain the model, dataset, optimizer, curriculum, and loss
settings. Use the exact resolved configuration that belongs to a checkpoint
when evaluating it. The full-size configurations are runnable starting points;
they do not claim to reproduce every training schedule reported in the paper
or the later extended-schedule control.

## Free-running inference

For a checkpoint trained by the compact example command:

```bash
python scripts/inference.py \
  --model tides \
  --checkpoint outputs/tides_example/checkpoint_best.pt \
  --dataset-root examples/mini_dataset \
  --split train --index 0 --steps 2 --device cpu \
  --output outputs/tides_rollout.npz
```

The checkpoint embeds its resolved configuration. Supply `--config` only when
loading a weights-only checkpoint without an embedded configuration. The
inference command strictly checks checkpoint compatibility and predicts all
steps from the single initial liquid state. It never uses future ground-truth
liquid velocity as an input.

For TIDE, use `--model tide` and a TIDE checkpoint. `--steps 1` requests a
one-step forecast. To use a prepared input instead of a manifest dataset:

```bash
python scripts/inference.py \
  --model tides --checkpoint checkpoints/tides.pt \
  --config configs/tides_train.json \
  --input input.npz --steps 8 --device cuda \
  --output outputs/rollout.npz
```

The prepared NPZ must contain arrays already expressed in the same coordinate
system and normalization as the checkpoint:

| Array | Unbatched shape | Meaning |
| --- | --- | --- |
| `fluid_init` | `[3,D,H,W]` | Observed liquid velocity at time t |
| `solid_sdf` | `[T,1,D,H,W]` | Solid signed-distance fields for t+1 through t+T |
| `solid_velocity` | `[T,3,D,H,W]` | Corresponding solid boundary velocities |
| `particle_pos_init` | `[P,3]` | Initial particle locations in x,y,z voxel coordinates; required for TIDES transport feedback |
| `solid_center_voxel` | `[T,3]` | Optional solid centers |
| `solid_radius_voxel` | `[T,1]` | Optional solid size metadata |

Batched inputs add a leading batch dimension. Outputs contain `velocity` with
shape `[B,T,3,D,H,W]`, an optional `particle_pos` sequence, and the configuration's
`velocity_scale`. Predictions retain the checkpoint's normalized units. For
the dataset loader's normalization, multiplying velocity by `velocity_scale`
recovers the stored dataset velocity units.

## Evaluation

```bash
python scripts/evaluate.py \
  --model tides \
  --checkpoint outputs/tides_example/checkpoint_best.pt \
  --dataset-root examples/mini_dataset --split train \
  --steps 2 --max-batches 1 --device cpu \
  --output outputs/tides_metrics.json
```

For the complete dataset, change the dataset root, use `--split test`, select
the matching pretrained checkpoint, and omit `--max-batches` to evaluate the
entire split. The output JSON records rollout metrics, one-step metrics, and
per-step curves produced by the model's training/evaluation implementation.
The compact-example metrics are pipeline checks and should not be cited as
benchmark results.

## Repository layout

```text
models/          TIDE/TIDES, shared bulk backbone, interface modules, corrections
data/            Manifest dataset loader and conditioning construction
train/           Model-specific training loops and losses
scripts/         Inference, evaluation, checkpoint loading, smoke test
configs/         Full-size and compact-example configurations
bash/            Training and smoke-test entrypoints
examples/        One compact dataset example
```

The example contains 12 frames on a 16 x 32 x 32 grid, obtained by temporal
truncation and spatial downsampling of an existing generated sample.
Full benchmark inputs use a 32 x 128 x 128 grid sampled at 8 Hz.

## Citation

```bibtex
@inproceedings{ryu2026interactive,
  title={Interactive 4D Volumetric Liquid Forecasting under Moving-Solid Interaction},
  author={Ryu, Jongwon and Yoon, Sunjae and Kim, Junyeong},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```
