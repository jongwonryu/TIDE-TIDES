# Dataset Example

`mini_dataset/` contains one compact example extracted from the already generated `dualsph_liquid_v2_dense` dataset.

The original full generated `.npz` sample is about 205 MB, so this example keeps the same field names and metadata contract but truncates/downsamples it for supplementary distribution:

- source dataset: `dualsph_liquid_v2_dense`
- source sample: `samples/train/0000004_rigidv1_tpl_straight_constant_0020.npz`
- source case: `rigidv1_tpl_straight_constant_0020`
- compact sample: `mini_dataset/samples/train/existing_sample_compact.npz`
- compact shape: 12 frames on a `16 x 32 x 32` Eulerian grid
- downsampling from the source grid: `D x H x W = 2 x 4 x 4`

The example includes the dense voxel tensors used by the method loader, plus a compact particle-state file referenced by `manifest.jsonl`.

Quick inspection from the supplementary root:

```bash
bash bash/inspect_dataset_example.sh
```
